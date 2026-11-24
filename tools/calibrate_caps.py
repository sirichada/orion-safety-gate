#!/usr/bin/env python3
"""
Cap calibration for Orion API models.

Samples a few prompts per case (seeded, so the sample is reproducible), sends them with a high
token cap through scripts/api_model.py, and measures how many completion tokens real replies
use. From that it proposes a per-case max_new_tokens for config/generation.json and estimates
what a full run of each case would cost and how long it would take. It never edits the config.

Usage (from the repo root, with the repo's .venv python):
    python tools/calibrate_caps.py --model "Qwen 3.8-Flash-Next" --model-id "Qwen 3.8-Flash-Next" --dry-run
    python tools/calibrate_caps.py --model "Qwen 3.8-Flash-Next" --model-id "Qwen 3.8-Flash-Next" --max-thb 5

--max-thb is required for a real run: the same spend guard as the suite stops the run before a
request whose worst case would pass it. --dry-run needs no key, no network, and writes nothing.
At most 3 requests are in flight (the suite's ceiling).

Every reply's full reasoning text is written to raw_<case>.reasoning.jsonl next to
raw_<case>.jsonl, which keeps only the reasoning length.

--ids-from <calibration.json> re-sends the prompts of an earlier run instead of drawing a seeded
sample: its sample_ids, or with --only-cut-off exactly those whose finish_reason was "length"
(read from the raw_<case>.jsonl files beside it). The model must be the same as that run's.

Sampling defaults to the model's protocol in config/generation.json (protocols[model-id], the same
table the suite reads); a model with no entry is refused. --temperature/--top-p/--top-k/
--presence-penalty override it, and a run that differs from the protocol is recorded with
matches_protocol false. top_k goes out inside the request's extra_body (the API does not accept it as a
top-level field); presence_penalty goes out as a top-level field. --repeats sends each
prompt that many times (ids get a __r<k> suffix). All of it is recorded in calibration.json.
If any request fails after its retries (a 4xx rejection, or a server error for an unsupported field), the run stops at once and prints the error
body; nothing falls back to other settings.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import api_model  # noqa: E402
import run_config  # noqa: E402

DEFAULT_GENERATION_CONFIG = REPO_ROOT / "config" / "generation.json"
DEFAULT_CASES = ("en_text", "th_text", "over_refusal", "th_knowledge")
PROMPT_FILES = {
    "en_text": REPO_ROOT / "data" / "prompt-sets" / "en_text.jsonl",
    "th_text": REPO_ROOT / "data" / "prompt-sets" / "th_text.jsonl",
    "over_refusal": REPO_ROOT / "data" / "prompt-sets" / "over_refusal.jsonl",
    "th_knowledge": REPO_ROOT / "data" / "prompt-sets" / "th_knowledge.jsonl",
}
DEFAULT_N = 20
DEFAULT_SEED = 20261014
DEFAULT_CALIBRATION_CAP = 8192
CAP_STEPS = (1024, 2048, 4096, 8192, 16384)
HEADROOM = 2.0
OUT_ROOT = REPO_ROOT / "runs" / "results" / "calibration"
LOG_DIR = REPO_ROOT / "runs" / "logs"


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> int:
        self.stdout.write(text)
        self.file.write(text)
        return len(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def load_prompts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_indices(n_total: int, n: int, seed: int, case_id: str) -> list[int]:
    """n distinct indices out of n_total, deterministic in (seed, case_id), in file order."""
    rng = random.Random(f"{seed}:{case_id}")
    return sorted(rng.sample(range(n_total), min(n, n_total)))


def propose_cap(max_completion_tokens: int | None, n_cut_off: int, calibration_cap: int) -> tuple[int | None, str]:
    """Smallest step that is at least HEADROOM times the largest observed reply, or None with a reason."""
    if n_cut_off:
        return None, f"{n_cut_off} sampled replies were cut off at the calibration cap {calibration_cap}; rerun with a higher cap"
    if max_completion_tokens is None:
        return None, "no completed replies in the sample"
    need = HEADROOM * max_completion_tokens
    for step in CAP_STEPS:
        if step >= need:
            return step, f"smallest of {CAP_STEPS} at least {HEADROOM:g} x the largest observed reply ({max_completion_tokens})"
    return None, f"{HEADROOM:g} x the largest observed reply ({max_completion_tokens}) is above {CAP_STEPS[-1]}"


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize_case(rows: list[dict], calibration_cap: int) -> dict:
    done = [r for r in rows if not r["failed"]]
    cut = [r for r in done if r["finish_reason"] == "length"]
    tokens = [r["completion_tokens"] for r in done if isinstance(r["completion_tokens"], int)]
    prompt_tokens = [r["prompt_tokens"] for r in done if isinstance(r["prompt_tokens"], int)]
    cap, reason = propose_cap(max(tokens) if tokens else None, len(cut), calibration_cap)
    return {
        "n_sampled": len(rows),
        "n_replied": len(done),
        "n_failed": len(rows) - len(done),
        "n_cut_off": len(cut),
        "n_empty": sum(1 for r in done if not r["has_content"]),
        "mean_prompt_tokens": statistics.fmean(prompt_tokens) if prompt_tokens else None,
        "mean_completion_tokens": statistics.fmean(tokens) if tokens else None,
        "p50_completion_tokens": percentile(tokens, 0.5) if tokens else None,
        "p90_completion_tokens": percentile(tokens, 0.9) if tokens else None,
        "max_completion_tokens": max(tokens) if tokens else None,
        "mean_latency_s": statistics.fmean(r["latency_s"] for r in done) if done else None,
        "proposed_max_new_tokens": cap,
        "proposal_basis": reason,
    }


def estimate_full_run(summary: dict, prompts: list[str], model: str, cap: int | None, concurrency: int) -> dict:
    price_in, price_out = api_model.price_of(model)
    n = len(prompts)
    estimate = {"n_prompts": n, "expected_thb": None, "worst_case_thb_at_proposed_cap": None, "expected_hours": None}
    if summary["mean_prompt_tokens"] is not None and summary["mean_completion_tokens"] is not None:
        per_prompt = (summary["mean_prompt_tokens"] * price_in + summary["mean_completion_tokens"] * price_out) / 1_000_000
        estimate["expected_thb"] = round(per_prompt * n, 4)
        estimate["expected_hours"] = round(n * summary["mean_latency_s"] / concurrency / 3600, 2)
    if cap is not None:
        estimate["worst_case_thb_at_proposed_cap"] = round(
            sum(api_model.worst_case_thb(model, api_model.estimate_input_tokens(p), cap) for p in prompts), 4
        )
    return estimate


def ids_from_calibration(path: Path, only_cut_off: bool, model: str) -> dict[str, list]:
    """Prompt ids per case from an earlier calibration run, optionally only those that were cut off."""
    path = Path(path)
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("model") != model:
        raise ValueError(f"{path} is a calibration of {calibration.get('model')!r}, not {model!r}")
    found: dict[str, list] = {}
    for case_id, summary in calibration.get("cases", {}).items():
        if only_cut_off:
            raw = path.parent / f"raw_{case_id}.jsonl"
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
            ids = [r["id"] for r in rows if r.get("finish_reason") == "length"]
        else:
            ids = list(summary.get("sample_ids", []))
        if ids:
            found[case_id] = ids
    return found


def write_reasoning(path: Path, row: dict, reasoning: str) -> None:
    entry = {
        "id": row["id"],
        "finish_reason": row["finish_reason"],
        "completion_tokens": row["completion_tokens"],
        "reasoning_content": reasoning,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="API model id")
    parser.add_argument("--model-id", required=True, help="Stable name the suite records; the key under models in generation.json")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated case ids")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Prompts sampled per case")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_CALIBRATION_CAP, help="Calibration cap sent with every request")
    parser.add_argument("--ids-from", default=None, help="calibration.json of an earlier run: re-send its prompts instead of sampling")
    parser.add_argument("--only-cut-off", action="store_true", help="with --ids-from: only the prompts that were cut off (finish_reason length)")
    parser.add_argument("--temperature", type=float, default=None, help="Default: the model's protocol")
    parser.add_argument("--top-p", type=float, default=None, help="Default: the model's protocol")
    parser.add_argument("--top-k", type=int, default=None, help="Sent in the request's extra_body. Default: the model's protocol")
    parser.add_argument(
        "--presence-penalty", type=float, default=None, help="In [-2, 2], sent as a top-level field. Default: the model's protocol"
    )
    parser.add_argument("--generation-config", default=None, help="Table holding protocols (default: config/generation.json)")
    parser.add_argument("--no-retries", action="store_true", help="Send every request once; any error ends it as failed")
    parser.add_argument("--repeats", type=int, default=1, help="Sends of each selected prompt")
    parser.add_argument("--api-concurrency", type=int, default=run_config.DEFAULTS["api_concurrency"])
    parser.add_argument("--max-thb", type=float, default=None, help="Spend limit in THB (required unless --dry-run)")
    parser.add_argument("--out-dir", default=None, help=f"Output folder (default: {OUT_ROOT}/<timestamp>)")
    parser.add_argument("--dry-run", action="store_true", help="Print the sample and spend bounds; no key, no network, no files")
    return parser


def main(argv: list[str] | None = None, make_generate=api_model.make_api_generate) -> int:
    args = build_parser().parse_args(argv)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in PROMPT_FILES]
    if unknown:
        sys.exit(f"unknown case(s): {', '.join(unknown)}; known: {', '.join(PROMPT_FILES)}")
    problem = api_model.price_problem(args.model)
    if problem:
        sys.exit(problem)
    if not 1 <= args.api_concurrency <= run_config.API_CONCURRENCY_CEILING:
        sys.exit(f"--api-concurrency must be between 1 and {run_config.API_CONCURRENCY_CEILING}")
    if args.max_thb is None and not args.dry_run:
        sys.exit("--max-thb is required for a real run")
    if args.n < 1:
        sys.exit("--n must be >= 1")
    if args.repeats < 1:
        sys.exit("--repeats must be >= 1")
    if args.top_k is not None and args.top_k < 1:
        sys.exit("--top-k must be >= 1")
    if args.temperature is not None and args.temperature < 0:
        sys.exit("--temperature must be >= 0")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        sys.exit("--top-p must be in (0, 1]")
    if args.presence_penalty is not None and not -2 <= args.presence_penalty <= 2:
        sys.exit("--presence-penalty must be in [-2, 2]")
    generation_path = Path(args.generation_config) if args.generation_config else DEFAULT_GENERATION_CONFIG
    try:
        protocol = run_config.resolve_protocol_sampling(run_config.load_generation_config(generation_path), args.model_id, "api")
    except run_config.GenerationConfigError as e:
        sys.exit(str(e))
    args.protocol_sampling = protocol
    for name in run_config.SAMPLING_FIELDS:
        if getattr(args, name) is None:
            setattr(args, name, protocol[name])
    args.sampling = run_config.effective_sampling(args.temperature, args.top_p, args.top_k, args.presence_penalty, protocol)
    if args.only_cut_off and not args.ids_from:
        sys.exit("--only-cut-off needs --ids-from")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.dry_run:
        return run(args, cases, stamp, make_generate)
    tee = Tee(LOG_DIR / f"calibrate_caps_{stamp}.log")
    sys.stdout = tee
    try:
        return run(args, cases, stamp, make_generate)
    finally:
        sys.stdout = tee.stdout
        tee.file.close()


def run(args, cases: list[str], stamp: str, make_generate) -> int:
    plan = {}
    fixed_ids = None
    if args.ids_from:
        try:
            fixed_ids = ids_from_calibration(Path(args.ids_from), args.only_cut_off, args.model)
        except (OSError, ValueError, KeyError) as e:
            sys.exit(f"--ids-from: {e}")
    for case_id in cases:
        records = load_prompts(PROMPT_FILES[case_id])
        if fixed_ids is not None:
            if case_id not in fixed_ids:
                continue
            by_id = {r.get("id"): r for r in records}
            missing = [i for i in fixed_ids[case_id] if i not in by_id]
            if missing:
                sys.exit(f"--ids-from: {case_id} ids not in the prompt set: {missing[:3]}")
            picked_records = [by_id[i] for i in fixed_ids[case_id]]
        else:
            picked_records = [records[i] for i in sample_indices(len(records), args.n, args.seed, case_id)]
        ids, prompts = [], []
        for r in picked_records:
            for k in range(args.repeats):
                ids.append(r.get("id") if args.repeats == 1 else f"{r.get('id')}__r{k + 1}")
                prompts.append(r["model_input"])
        plan[case_id] = {"all": [r["model_input"] for r in records], "ids": ids, "prompts": prompts}
    if not plan:
        sys.exit("nothing to send: no prompts selected")

    price_in, price_out = api_model.price_of(args.model)
    sample_worst = sum(
        api_model.worst_case_thb(args.model, api_model.estimate_input_tokens(p), args.max_new_tokens)
        for case in plan.values()
        for p in case["prompts"]
    )
    if args.ids_from:
        what = "cut-off prompts" if args.only_cut_off else "prompts"
        print(f"model {args.model} ({args.model_id}); {what} of {args.ids_from}, at cap {args.max_new_tokens}")
    else:
        print(f"model {args.model} ({args.model_id}); seed {args.seed}; {args.n} prompts per case at cap {args.max_new_tokens}")
    for case_id, case in plan.items():
        print(f"  {case_id}: {len(case['prompts'])} of {len(case['all'])} prompts sampled")
    label = "" if args.sampling["matches_protocol"] else " [NOT the model's protocol]"
    top_k = f", top_k {args.top_k} (extra_body)" if args.top_k is not None else ""
    penalty = f", presence_penalty {args.presence_penalty:g}" if args.presence_penalty is not None else ""
    print(f"sampling: temperature {args.temperature:g}, top_p {args.top_p:g}{top_k}{penalty}{label}")
    print(f"worst case for the sample: {sample_worst:.2f} THB (THB {price_in:g} in / {price_out:g} out per 1M tokens)")
    if args.max_thb is not None:
        print(f"--max-thb {args.max_thb:g}")
    if args.dry_run:
        print("Dry run: nothing was sent and nothing was written.")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    spend = api_model.SpendGuard(args.model, args.max_thb, out_dir / "spend_ledger.json")
    extra = {}
    if args.top_k is not None:
        extra["extra_body"] = {"top_k": args.top_k}
    if args.presence_penalty is not None:
        extra["presence_penalty"] = args.presence_penalty
    extra = extra or None
    kwargs = {"extra_params": extra} if extra else {}
    if args.no_retries:
        kwargs["max_attempts"] = 1
    generate = make_generate(args.model, concurrency=args.api_concurrency, no_answer_retries=0, spend=spend, **kwargs)

    results: dict = {}
    stopped = None
    rejected = None
    first_sent = False
    batch = args.api_concurrency * 4
    for case_id, case in plan.items():
        rows: list[dict] = []
        raw_path = out_dir / f"raw_{case_id}.jsonl"
        reasoning_path = out_dir / f"raw_{case_id}.reasoning.jsonl"
        try:
            i = 0
            while i < len(case["prompts"]):
                # With a non-default sampling field the very first request goes out alone, so a
                # rejection is seen before anything else is sent.
                size = 1 if (extra and not rows and not first_sent) else batch
                chunk = case["prompts"][i : i + size]
                first_sent = True
                texts = generate(chunk, args.max_new_tokens, args.temperature, args.top_p)
                for j, (text, meta) in enumerate(zip(texts, generate.meta)):
                    row = {
                        "id": case["ids"][i + j],
                        "failed": bool(meta.get("failed")),
                        "finish_reason": meta.get("finish_reason"),
                        "prompt_tokens": meta.get("prompt_tokens"),
                        "completion_tokens": meta.get("completion_tokens"),
                        "latency_s": meta.get("latency_s"),
                        "cost_thb": meta.get("cost_thb"),
                        "has_content": bool(text.strip()),
                        "reasoning_chars": len(meta.get("reasoning_content") or ""),
                        "response": text,
                    }
                    rows.append(row)
                    with open(raw_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    write_reasoning(reasoning_path, row, meta.get("reasoning_content") or "")
                    if meta.get("failed") and rejected is None:
                        rejected = f"{row['id']}: {meta['error']}"
                i += size
                if rejected:
                    stopped = f"request failed, stopping before sending anything else: {rejected}"
                    print(f"STOPPED: {stopped}")
                    break
                print(f"{case_id}: {len(rows)}/{len(case['prompts'])} sampled, spent {spend.spent_thb:.4f} THB")
        except run_config.BudgetExceeded as e:
            stopped = f"{case_id}: {e}"
            print(f"STOPPED: {stopped}")
        if rows:
            summary = summarize_case(rows, args.max_new_tokens)
            summary["sample_ids"] = case["ids"][: len(rows)]
            summary["estimate_full_run"] = estimate_full_run(
                summary, case["all"], args.model, summary["proposed_max_new_tokens"], args.api_concurrency
            )
            results[case_id] = summary
        if stopped:
            break

    snippet = {
        args.model_id: {c: {"max_new_tokens": s["proposed_max_new_tokens"]} for c, s in results.items() if s["proposed_max_new_tokens"]}
    }
    calibration = {
        "model": args.model,
        "model_id": args.model_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_per_case": args.n if not args.ids_from else None,
        "ids_from": {"calibration": args.ids_from, "only_cut_off": args.only_cut_off} if args.ids_from else None,
        "calibration_max_new_tokens": args.max_new_tokens,
        "api_concurrency": args.api_concurrency,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "top_k_location": args.sampling["top_k_location"],
            "presence_penalty": args.presence_penalty,
            "presence_penalty_location": args.sampling["presence_penalty_location"],
            "matches_protocol": args.sampling["matches_protocol"],
            "protocol": args.protocol_sampling,
            "extra_params_sent": extra,
            "retries": not args.no_retries,
            "repeats": args.repeats,
        },
        "prices_thb_per_m": {"input": price_in, "output": price_out, "as_of": "2026-10-13"},
        "spent_thb": round(spend.spent_thb, 6),
        "stopped": stopped,
        "cases": results,
        "generation_config_snippet": snippet,
    }
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCalibration result")
    for case_id, s in results.items():
        e = s["estimate_full_run"]
        print(
            f"  {case_id}: replied {s['n_replied']}/{s['n_sampled']}, cut off {s['n_cut_off']}, "
            f"completion tokens mean {s['mean_completion_tokens']:.0f} p90 {s['p90_completion_tokens']} max {s['max_completion_tokens']}; "
            f"proposed cap {s['proposed_max_new_tokens']}; full run ~{e['expected_thb']} THB "
            f"(worst {e['worst_case_thb_at_proposed_cap']}), ~{e['expected_hours']} h at {args.api_concurrency} in flight"
            if s["mean_completion_tokens"] is not None
            else f"  {case_id}: no completed replies"
        )
    print(f"spent {spend.spent_thb:.4f} THB; wrote {out_dir / 'calibration.json'}")
    print(f"generation.json snippet: {json.dumps(snippet)}")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
