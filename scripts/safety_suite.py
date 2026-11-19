#!/usr/bin/env python3
"""
Safety-admission CLI: runs every case that applies to a candidate model's
declared capabilities and writes one admission verdict.

Usage:
    python safety_suite.py run --model <hf-repo-or-path-or-api-id> --model-id <stable name> \\
        --capabilities capabilities.json --thresholds ../config/thresholds.json [--backend api]
    python safety_suite.py run --model <...> --model-id <name> --modalities text --languages en th
    python safety_suite.py run ... --only over_refusal    # partial run; never an admission
    python safety_suite.py run --backend api --max-thb 40 ...    # api runs need a spend limit
    python safety_suite.py run --backend api --dry-run ...       # per-case worst-case/expected THB; no key, no network
    python safety_suite.py run ... --resume runs/results/<run folder>    # continue an interrupted run
    python safety_suite.py rescore runs/results/<run folder>             # re-derive its verdict offline

In a real terminal, any of --modalities/--languages/--model-id left out is asked for,
and a summary is shown for confirmation before anything runs (--yes skips the
confirmation, --no-input never prompts).

Intermediate responses/scores and verdict.json go to
runs/results/<model-id>_<timestamp>/ (or --out for the verdict), each stage's
output is logged under runs/logs/, and one row is appended to runs/_index.md
for every run, including runs that end in a suite error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import rescore
import run_config
from cases import REGISTRY, OnlyError, select_cases
from cli_input import Cancelled, Inputs, QuestionaryAsk, ask_or_fail, no_terminal_error, resolve_inputs
from orchestrator import (
    CASE_RUNNERS,
    REPO_ROOT,
    RunStage,
    SuiteError,
    count_lines,
    preflight,
    read_run_config,
    responses_path,
    run_case,
    run_subprocess,
)
import report
import review
from verdict import build_verdict

RESULTS = REPO_ROOT / "runs" / "results"
INDEX = REPO_ROOT / "runs" / "_index.md"
DEFAULT_GENERATION_CONFIG = REPO_ROOT / "config" / "generation.json"
SPEND_LEDGER_NAME = "spend_ledger.json"

# Forwarded explicitly so every verdict records the N that applied.
DEFAULT_MAX_CONSECUTIVE_FAILURES = run_config.DEFAULTS["max_consecutive_failures"]


def parse_only(value: str) -> tuple[str, ...]:
    known = [case.case_id for case in REGISTRY]
    ids: list[str] = []
    for token in value.split(","):
        case_id = token.strip()
        if not case_id:
            raise argparse.ArgumentTypeError(f"empty case id in {value!r}")
        if case_id not in known:
            raise argparse.ArgumentTypeError(f"unknown case {case_id!r}; known cases: {', '.join(known)}")
        if case_id not in ids:
            ids.append(case_id)
    return tuple(ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rescore = sub.add_parser(
        "rescore",
        help="Re-derive a recorded run's verdict from its run folder, with no model and no network",
    )
    rescore.add_argument("run_dir", help="A run folder under runs/results/ that holds verdict.json and scored_<case>.jsonl")
    rescore.add_argument("--thresholds", default=None, help="Thresholds config JSON (default: config/thresholds.json)")
    rescore.add_argument(
        "--out", default=None, help="Verdict path (default: verdict.rescore-<timestamp>.json in the run folder; an existing file is never replaced)"
    )
    rescore.add_argument(
        "--rerun-scorers",
        action="store_true",
        help="Score responses_<case>.jsonl again instead of reading the recorded scored_<case>.jsonl; "
        "needs the scorers' dependencies (a GPU for the en_text/th_text classifiers). Scored files go to rescore-<timestamp>/ in the run folder",
    )
    rescore.add_argument("--force", action="store_true", help="Let --out replace an existing file")
    report_p = sub.add_parser("report", help="Write the HTML admission report for a run folder's verdict")
    report_p.add_argument("run_dir", help="A run folder under runs/results/")
    report_p.add_argument(
        "--verdict",
        default="verdict.json",
        help="Verdict file in the run folder (default: verdict.json); the report is named after it and never replaces an existing one",
    )
    review_p = sub.add_parser("review", help="Apply a reviewer's decision on a needs_review case; re-runs nothing")
    review_p.add_argument("run_dir", help="A run folder under runs/results/")
    review_p.add_argument("--review", required=True, help="The review.json the reviewer downloaded from the report")
    review_p.add_argument("--verdict", default="verdict.json", help="Verdict file in the run folder to review (default: verdict.json)")
    run = sub.add_parser("run", help="Run the suite against one candidate model")
    run.add_argument("--model", required=True, help="Base model path/HF repo id, or the API model id with --backend api")
    run.add_argument("--model-id", default=None, help="Stable declared name/version recorded in the verdict")
    run.add_argument("--capabilities", default=None, help="Path to a capabilities.json (instead of --modalities/--languages)")
    run.add_argument("--modalities", nargs="+", choices=["text", "image"], default=None)
    run.add_argument("--languages", nargs="+", choices=["en", "th"], default=None)
    run.add_argument("--thresholds", default=None, help="Thresholds config JSON (default: config/thresholds.json)")
    run.add_argument(
        "--only",
        type=parse_only,
        default=None,
        help="Comma-separated case ids; runs only these. The verdict is marked partial and is never an admission",
    )
    run.add_argument("--yes", action="store_true", help="Skip the confirmation before running")
    run.add_argument("--no-input", action="store_true", help="Never prompt; every input must be given as a flag")
    run.add_argument("--out", default=None, help="Verdict path (default: verdict.json in the run folder)")
    run.add_argument("--backend", choices=["hf", "api"], default="hf")
    run.add_argument("--adapter", default=None, help="Optional PEFT adapter (hf backend only)")
    run.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Prompts per generate() call (default: {run_config.DEFAULTS['batch_size']} hf, "
        f"{run_config.DEFAULTS['api_batch_size']} api, where it only sets checkpoint granularity)",
    )
    run.add_argument(
        "--api-concurrency",
        type=int,
        default=run_config.DEFAULTS["api_concurrency"],
        help=f"api backend: requests in flight at once, 1 to {run_config.API_CONCURRENCY_CEILING}",
    )
    run.add_argument("--max-thb", type=float, default=None, help="api backend: spend limit in THB for the whole run (required)")
    run.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="api backend: sampling temperature (default: the model's protocol in the generation config). A value off the protocol makes the run non-admitting",
    )
    run.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="api backend: top_p (default: the model's protocol in the generation config)",
    )
    run.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="api backend: top_k, sent in the request's extra_body (default: the model's protocol)",
    )
    run.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="api backend: presence_penalty in [-2, 2], sent as a top-level request field (default: the model's protocol)",
    )
    run.add_argument(
        "--sample-per-case",
        type=int,
        default=None,
        help="Run a seeded random sample of this many prompts per case instead of the full set; never an admission",
    )
    run.add_argument("--sample-seed", type=int, default=None, help="Seed for --sample-per-case (required with it)")
    run.add_argument("--generation-config", default=None, help="Per-case/per-model cap table (default: config/generation.json)")
    run.add_argument(
        "--calibration-file", default=None, help="calibration.json from tools/calibrate_caps.py, for --dry-run's expected spend"
    )
    run.add_argument("--resume", default=None, help="Run folder of an interrupted run to continue (same settings only)")
    run.add_argument("--dry-run", action="store_true", help="Show the plan and its spend bounds; run nothing and write nothing")
    run.add_argument(
        "--precision",
        choices=run_config.PRECISIONS,
        default=run_config.DEFAULTS["precision"],
        help="4bit = bitsandbytes NF4 (default); full = bfloat16; as_published = the checkpoint's own "
        "dtype and quantization, required for pre-quantized checkpoints. hf backend only",
    )
    run.add_argument("--model-revision", default=None, help="Hub branch/tag/commit for --model (default: main, resolved once)")
    run.add_argument("--adapter-revision", default=None, help="Hub branch/tag/commit for --adapter (default: main, resolved once)")
    run.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES)
    return parser


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("rescore", "report", "review"):
        return args
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be >= 1")
    if args.batch_size is None:
        key = "api_batch_size" if args.backend == "api" else "batch_size"
        args.batch_size = run_config.DEFAULTS[key]
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.adapter_revision and not args.adapter:
        parser.error("--adapter-revision needs --adapter")
    if args.backend == "api":
        hf_only = [
            flag
            for flag, given in (
                ("--adapter", args.adapter),
                ("--model-revision", args.model_revision),
                ("--adapter-revision", args.adapter_revision),
                ("--precision", args.precision != run_config.DEFAULTS["precision"]),
            )
            if given
        ]
        if hf_only:
            parser.error(f"{', '.join(hf_only)} apply only to the hf backend")
        if not 1 <= args.api_concurrency <= run_config.API_CONCURRENCY_CEILING:
            parser.error(f"--api-concurrency must be between 1 and {run_config.API_CONCURRENCY_CEILING}")
        if args.max_thb is None:
            if not args.dry_run:
                parser.error("--max-thb is required for --backend api (the spend limit for the whole run, in THB)")
        elif not args.max_thb > 0:
            parser.error("--max-thb must be positive")
    elif args.max_thb is not None:
        parser.error("--max-thb applies only to the api backend")
    sampling_flags = [
        flag
        for flag, given in (
            ("--temperature", args.temperature is not None),
            ("--top-p", args.top_p is not None),
            ("--top-k", args.top_k is not None),
            ("--presence-penalty", args.presence_penalty is not None),
        )
        if given
    ]
    if sampling_flags and args.backend != "api":
        parser.error(f"{', '.join(sampling_flags)} apply only to the api backend")
    if args.backend != "api":
        args.temperature, args.top_p = run_config.GREEDY_SAMPLING["temperature"], run_config.GREEDY_SAMPLING["top_p"]
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.presence_penalty is not None and not -2 <= args.presence_penalty <= 2:
        parser.error("--presence-penalty must be in [-2, 2]")
    if args.sample_per_case is not None:
        if args.sample_per_case < 1:
            parser.error("--sample-per-case must be >= 1")
        if args.sample_seed is None:
            parser.error("--sample-per-case needs --sample-seed")
    elif args.sample_seed is not None:
        parser.error("--sample-seed needs --sample-per-case")
    return args


def resolve_local(value: str | None) -> str | None:
    # Stages run with scripts/ as their cwd, so a local path typed relative to the caller's cwd
    # must be made absolute; anything that isn't an existing path (HF repo or API id) passes through.
    if value is None:
        return None
    return str(Path(value).resolve()) if Path(value).exists() else value


def sampling_of(args) -> dict:
    return run_config.effective_sampling(
        args.temperature, args.top_p, args.top_k, args.presence_penalty, getattr(args, "protocol_sampling", None)
    )


def apply_protocol_sampling(args, protocol: dict) -> None:
    """Fills every sampling field a flag left unset from the model's protocol; a flag that is set wins and shows as a deviation."""
    args.protocol_sampling = protocol
    for name in run_config.SAMPLING_FIELDS:
        if getattr(args, name) is None:
            setattr(args, name, protocol[name])


def case_caps(generation: dict | None) -> dict:
    """The caps one case runs with; the protocol defaults when no table entry was resolved."""
    d = run_config.DEFAULTS
    generation = generation or {}
    return {
        "max_new_tokens": generation.get("max_new_tokens") or d["max_new_tokens"],
        "max_input_tokens": generation.get("max_input_tokens") or d["max_input_tokens"],
    }


def wrapper_args(
    args, revisions: dict | None = None, generation: dict | None = None, spend_ledger: Path | None = None
) -> list[str]:
    """Every protocol, hardware and per-model value is forwarded, so no wrapper default is relied on."""
    revisions = revisions or {}
    d = run_config.DEFAULTS
    caps = case_caps(generation)
    forwarded = [
        "--model", resolve_local(args.model),
        "--backend", args.backend,
        "--batch-size", str(args.batch_size),
        "--max-consecutive-failures", str(args.max_consecutive_failures),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-new-tokens", str(caps["max_new_tokens"]),
    ]
    if args.adapter:
        forwarded += ["--adapter", resolve_local(args.adapter)]
    if args.backend == "api":
        forwarded += [
            "--api-concurrency", str(args.api_concurrency),
            "--no-answer-retries", str(d["no_answer_retries"]),
        ]
        if args.top_k is not None:
            forwarded += ["--top-k", str(args.top_k)]
        if args.presence_penalty is not None:
            forwarded += ["--presence-penalty", str(args.presence_penalty)]
        if args.max_thb is not None:
            forwarded += ["--max-thb", str(args.max_thb)]
        if spend_ledger is not None:
            forwarded += ["--spend-ledger", str(spend_ledger)]
    if getattr(args, "resume", None):
        forwarded += ["--resume"]
    if args.backend == "hf":
        forwarded += ["--precision", args.precision, "--max-input-tokens", str(caps["max_input_tokens"])]
        if revisions.get("model"):
            forwarded += ["--model-revision", revisions["model"]]
        if revisions.get("adapter"):
            forwarded += ["--adapter-revision", revisions["adapter"]]
    return forwarded


def forwarded_values(args, revisions: dict | None = None, generation: dict | None = None) -> dict:
    """What a stage's report must say it used, for the arguments wrapper_args forwards."""
    revisions = revisions or {}
    d = run_config.DEFAULTS
    caps = case_caps(generation)
    values = {
        "backend": args.backend,
        "model": resolve_local(args.model),
        "adapter": resolve_local(args.adapter),
        "batch_size": args.batch_size,
        "max_consecutive_failures": args.max_consecutive_failures,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": caps["max_new_tokens"],
    }
    if args.backend == "api":
        values.update(api_concurrency=args.api_concurrency, no_answer_retries=d["no_answer_retries"])
        if args.top_k is not None:
            values["top_k"] = args.top_k
        if args.presence_penalty is not None:
            values["presence_penalty"] = args.presence_penalty
    if args.backend == "hf":
        values.update(
            precision=args.precision,
            max_input_tokens=caps["max_input_tokens"],
            model_revision=revisions.get("model"),
            adapter_revision=revisions.get("adapter"),
        )
    return values


def resolve_run_revisions(args) -> dict:
    """Model and adapter commits, resolved once before any case so every case loads the same weights.

    A local path has no commit and is left to be identified by its files. The api backend has none.
    """
    if args.backend != "hf":
        return {"model": None, "adapter": None}
    found = {}
    try:
        for key, repo, revision in (
            ("model", args.model, args.model_revision),
            ("adapter", args.adapter, args.adapter_revision),
        ):
            if repo is None:
                found[key] = None
                continue
            resolved = run_config.resolve_revision(repo, revision)
            found[key] = resolved if isinstance(resolved, str) else None
    except run_config.RevisionError as e:
        raise SuiteError(str(e)) from e
    return found


def build_run_args(
    args,
    inputs: Inputs,
    thresholds_bytes: bytes,
    baseline: dict,
    flags: list[dict],
    prompt_sets: dict,
    generation: dict | None = None,
    generation_source: str | None = None,
    spent_thb: float | None = None,
    sample: dict | None = None,
) -> dict:
    provenance = baseline["provenance"]
    return {
        "model": resolve_local(args.model),
        "backend": args.backend,
        "adapter": resolve_local(args.adapter),
        "adapter_revision": provenance["adapter_revision"],
        "model_revision": provenance["model_revision"],
        "batch_size": args.batch_size,
        "max_consecutive_failures": args.max_consecutive_failures,
        "precision": baseline["precision"],
        "thresholds_file": str(inputs.thresholds_path),
        "thresholds_sha256": hashlib.sha256(thresholds_bytes).hexdigest(),
        "capabilities_source": inputs.source,
        "only": list(args.only) if args.only is not None else None,
        "generation": {
            "per_case": generation,
            "sha256": run_config.generation_sha256(generation) if generation is not None else None,
            "source": generation_source,
        },
        "api": (
            {
                "concurrency": args.api_concurrency,
                "no_answer_retries": run_config.DEFAULTS["no_answer_retries"],
                "max_thb": args.max_thb,
                "spent_thb": spent_thb,
            }
            if args.backend == "api"
            else None
        ),
        "protocol": {**run_config.PROTOCOL, **(getattr(args, "protocol_sampling", {}) if args.backend == "api" else {})},
        "sampling": sampling_of(args),
        "prompt_sample": sample,
        "per_model": baseline["per_model"],
        "model_input": baseline["model_input"],
        "provenance": {
            "suite": run_config.git_state(REPO_ROOT),
            "libraries": provenance["libraries"],
            "gpu": provenance["gpu"],
            "prompt_sets": prompt_sets,
            "model_files": provenance["model_files"],
            "adapter_files": provenance["adapter_files"],
            "api": provenance["api"],
        },
        "config_mismatches": flags,
    }


def format_run_summary(
    args, inputs: Inputs, selection, thresholds: dict, revisions: dict | None = None, generation: dict | None = None
) -> str:
    counts = {case.case_id: count_lines(CASE_RUNNERS[case.case_id].prompts) for case in selection.generating_cases}
    if getattr(args, "sample_per_case", None) is not None:
        counts = {cid: min(n, args.sample_per_case) for cid, n in counts.items()}
    lines = [
        "About to run:",
        f"  model        {args.model}  ({args.backend})",
        f"  record as    {inputs.model_id}",
        f"  settings     {settings_line(args, revisions or {}, generation)}",
        "  tests        " + (", ".join(f"{cid} ({n:,} prompts)" for cid, n in counts.items()) or "(none)"),
        f"  total        {sum(counts.values()):,} prompts",
    ]
    if selection.run_scope == "partial":
        ran = ", ".join(case.case_id for case in selection.cases)
        lines.insert(1, f"  scope        Partial run: {ran} only. Not an admission verdict.")
    derived = [case for case in selection.cases if case.derived]
    if derived:
        lines.append(
            "  scored from  "
            + "; ".join(f"{case.case_id} (replies of {', '.join(case.sources)}; no prompts sent)" for case in derived)
        )
    not_attempted = not_attempted_summary(selection)
    if not_attempted:
        lines.append("  not tested   " + "; ".join(not_attempted))
    if args.backend == "api" and args.max_thb is not None:
        lines.append(f"  spend limit  {args.max_thb:g} THB for the whole run ({args.api_concurrency} requests in flight)")
    if thresholds.get("provisional", True):
        lines.append("  note         thresholds are provisional placeholders; the verdict is not a real decision")
    return "\n".join(lines)


def settings_line(args, revisions: dict, generation: dict | None = None) -> str:
    d = run_config.DEFAULTS
    if generation:
        caps = {cid: case_caps(g)["max_new_tokens"] for cid, g in generation.items()}
        cap_text = "max_new_tokens " + (
            str(next(iter(caps.values()))) if len(set(caps.values())) == 1 else ", ".join(f"{c} {n}" for c, n in caps.items())
        )
    else:
        cap_text = f"max_new_tokens {d['max_new_tokens']}"
    sampling = sampling_of(args)
    top_k = f", top_k {sampling['top_k']} (extra_body)" if sampling["top_k"] is not None else ""
    if sampling["presence_penalty"] is not None:
        top_k += f", presence_penalty {sampling['presence_penalty']:g}"
    label = "" if sampling["matches_protocol"] else " [NOT the protocol: the run cannot be an admission]"
    parts = [
        f"batch {args.batch_size}",
        cap_text,
        f"temperature {sampling['temperature']:g}, top_p {sampling['top_p']:g}{top_k} requested{label}",
    ]
    if args.backend == "hf":
        parts.insert(0, f"precision {args.precision}")
        input_caps = {case_caps(g)["max_input_tokens"] for g in (generation or {}).values()} or {d["max_input_tokens"]}
        parts.append("max_input_tokens " + "/".join(str(n) for n in sorted(input_caps)))
        if revisions.get("model"):
            parts.append(f"model@{revisions['model'][:12]}")
        if args.adapter and revisions.get("adapter"):
            parts.append(f"adapter@{revisions['adapter'][:12]}")
    return ", ".join(parts)


def collect_run_config(
    selection, run_dir: Path, args, revisions: dict, generation: dict | None = None
) -> tuple[dict, list[dict]]:
    """Reads every case's stage report, checks it against what was forwarded, and merges them.

    A difference in anything that defines what was tested stops the run without a verdict.
    """
    reports = {}
    for case in selection.generating_cases:
        expected = forwarded_values(args, revisions, (generation or {}).get(case.case_id))
        report = read_run_config(responses_path(run_dir, case), case.case_id)
        problems = run_config.check_forwarded(expected, report)
        if problems:
            raise SuiteError(f"{case.case_id}: the wrapper did not use the forwarded settings: " + "; ".join(problems))
        reports[case.case_id] = report
    baseline, errors, flags = run_config.merge_reports(reports)
    if errors:
        detail = "; ".join(f"{e['field']} differs between cases ({json.dumps(e['values'], default=str)[:300]})" for e in errors)
        raise SuiteError(f"cases ran with different configurations, so the verdict would describe two tested things: {detail}")
    return baseline, flags


def not_attempted_summary(selection) -> list[str]:
    entries = selection.declared_scope["not_attempted"]
    by_reason: dict[str, list[str]] = {}
    for e in entries:
        label = e["case_id"] or f"whole {e['gate']} gate"
        by_reason.setdefault(e["reason"], []).append(label)
    return [f"{', '.join(labels)} ({reason})" for reason, labels in by_reason.items()]


def sample_records(path: Path, n: int, seed: int, case_id: str) -> list[dict]:
    """A seeded random sample of a prompt set, in the set's own order; the whole set when it is smaller than n.

    The same (seed, case) always picks the same records, so runs at different settings are paired.
    """
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if n >= len(records):
        return records
    picked = sorted(random.Random(f"{seed}:{case_id}").sample(range(len(records)), n))
    return [records[i] for i in picked]


def write_samples(selection, run_dir: Path, n: int, seed: int) -> tuple[dict, dict]:
    """Writes each case's sample into the run folder; returns (case id -> path, the record for run_args)."""
    paths, ids = {}, {}
    for case in selection.generating_cases:
        records = sample_records(CASE_RUNNERS[case.case_id].prompts, n, seed, case.case_id)
        path = run_dir / f"prompts_{case.case_id}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        paths[case.case_id] = path
        ids[case.case_id] = [r.get("id") for r in records]
    return paths, {"per_case": n, "seed": seed, "ids": ids}


def protocol_deviations(args) -> list[str]:
    found = []
    if not sampling_of(args)["matches_protocol"]:
        found.append("sampling")
    if args.sample_per_case is not None:
        found.append("prompt_sample")
    return found


def partial_suffix(selection) -> str:
    if selection.run_scope == "full":
        return ""
    return "_partial-" + "+".join(case.case_id for case in selection.cases)


def run(args, inputs: Inputs, stamp: str, run_stage: RunStage = run_subprocess, confirm=None) -> dict:
    # Hash the same bytes that are parsed, so an edit during a long run can't mismatch the two.
    thresholds_bytes = inputs.thresholds_path.read_bytes()
    thresholds = json.loads(thresholds_bytes.decode("utf-8"))
    try:
        selection = select_cases(inputs.capabilities, only=args.only)
    except OnlyError as e:
        raise SuiteError(str(e)) from e
    preflight(selection, thresholds)

    generation_path = Path(args.generation_config) if args.generation_config else DEFAULT_GENERATION_CONFIG
    try:
        table = run_config.load_generation_config(generation_path)
        apply_protocol_sampling(args, run_config.resolve_protocol_sampling(table, inputs.model_id, args.backend))
        generation = {
            case.case_id: run_config.resolve_generation(table, case.case_id, inputs.model_id, args.backend)
            for case in selection.generating_cases
        }
    except run_config.GenerationConfigError as e:
        raise SuiteError(str(e)) from e
    if args.backend == "api":
        unpriced = api_price_problem(args.model)
        if unpriced:
            raise SuiteError(unpriced)

    revisions = resolve_run_revisions(args)
    print(format_run_summary(args, inputs, selection, thresholds, revisions, generation))
    if args.dry_run:
        print(format_dry_run(args, selection, generation, args.calibration_file))
        print("Dry run: nothing was sent and nothing was written.")
        return None
    if confirm is not None:
        answer = confirm("Proceed?")
        if not answer:
            raise Cancelled()

    if args.resume:
        # The stages run from scripts/, so a relative folder would point somewhere else there.
        run_dir = Path(args.resume).resolve()
        if not run_dir.is_dir():
            raise SuiteError(f"--resume: {run_dir} is not a run folder")
    else:
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", inputs.model_id)
        run_dir = RESULTS / f"{safe_id}_{stamp}{partial_suffix(selection)}"
        run_dir.mkdir(parents=True, exist_ok=False)
    ledger = run_dir / SPEND_LEDGER_NAME if args.backend == "api" else None
    sample_paths: dict = {}
    sample_record = None
    if args.sample_per_case is not None:
        sample_paths, sample_record = write_samples(selection, run_dir, args.sample_per_case, args.sample_seed)

    try:
        case_results = [
            run_case(
                case,
                run_dir,
                [] if case.derived else wrapper_args(args, revisions, generation[case.case_id], ledger),
                thresholds,
                run_stage=run_stage,
                timestamp=stamp,
                generation=generation.get(case.case_id),
                prompts=sample_paths.get(case.case_id),
            )
            for case in selection.cases
        ]
    except SuiteError:
        # A failure before any case left output must not leave an empty run folder behind.
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise
    baseline, flags = collect_run_config(selection, run_dir, args, revisions, generation)
    prompt_sets = run_config.prompt_set_hashes(
        {case.case_id: sample_paths.get(case.case_id, CASE_RUNNERS[case.case_id].prompts) for case in selection.generating_cases}
    )
    verdict = build_verdict(
        inputs.model_id,
        selection,
        case_results,
        thresholds_provisional=bool(thresholds.get("provisional", True)),
        deviations=protocol_deviations(args),
        scored_files={r["case_id"]: run_dir / f"scored_{r['case_id']}.jsonl" for r in case_results},
        run_args=build_run_args(
            args,
            inputs,
            thresholds_bytes,
            baseline,
            flags,
            prompt_sets,
            generation=generation,
            generation_source=str(generation_path) if table is not None else None,
            spent_thb=read_spent(ledger),
            sample=sample_record,
        ),
    )

    out = Path(args.out) if args.out else run_dir / "verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    cases = ", ".join(r["case_id"] for r in verdict["case_results"]) or "(none)"
    outcome = (
        f"safety={verdict['safety_gate']}, smartness={verdict['smartness_gate']}, admitted={verdict['admitted']}"
        + (" (provisional thresholds)" if verdict["thresholds_provisional"] else "")
    )
    scope = "; scope=partial" if verdict["run_scope"] == "partial" else ""
    if verdict["protocol_deviations"]:
        scope += "; non-protocol: " + "+".join(verdict["protocol_deviations"])
    folder = run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir
    append_index_row(stamp, inputs.model_id, f"cases: {cases}{scope}", outcome, f"{folder}/, {out}")
    print(format_verdict_summary(verdict))
    print(f"safety_gate={verdict['safety_gate']} smartness_gate={verdict['smartness_gate']} admitted={verdict['admitted']}")
    print(f"Verdict -> {out}")
    write_report_for(out)
    return verdict


def display_folder(run_dir: Path) -> str:
    """The run folder as the review command should name it: repo-relative with forward slashes when under the repo."""
    path = Path(run_dir).resolve()
    return (path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path).as_posix()


def report_thresholds() -> dict | None:
    try:
        return json.loads((REPO_ROOT / "config" / "thresholds.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_report_for(verdict_path: Path) -> Path | None:
    """Writes the report next to a verdict; a failure here is reported but never undoes the verdict."""
    try:
        out = report.write_report(verdict_path, run_folder=display_folder(Path(verdict_path).parent), thresholds=report_thresholds())
    except SuiteError as e:
        print(f"Report not written: {e}")
        return None
    print(f"Report -> {out}")
    return out


def format_verdict_summary(verdict: dict) -> str:
    """A per-case table of a finished verdict for the terminal; verdict.json stays authoritative."""
    rows = [("case", "gate", "score", "pass mark", "scored", "status")]
    for r in verdict["case_results"]:
        score = "-" if r.get("score") is None else f"{r['score']:.3f}"
        mark = "-"
        if r.get("threshold") is not None:
            mark = ("<= " if r.get("direction") == "lower_is_better" else ">= ") + f"{r['threshold']:g}"
        scored = f"{r.get('n_scored', '-')}/{r.get('n_prompts', '-')}"
        rows.append((r["case_id"], r["gate"], score, mark, scored, r["status"]))
    for entry in (verdict.get("declared_scope") or {}).get("not_attempted", []):
        if entry.get("case_id"):
            rows.append((entry["case_id"], entry.get("gate", "-"), "-", "-", "-", f"not run ({entry.get('reason_code', '?')})"))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    lines.append(
        f"safety gate: {verdict['safety_gate']}   smartness gate: {verdict['smartness_gate']}   admitted: {verdict['admitted']}"
    )
    if verdict.get("thresholds_provisional"):
        lines.append("Thresholds are provisional placeholders: this verdict is not an admission decision.")
    if verdict.get("run_scope") == "partial":
        lines.append("Partial run: not an admission verdict.")
    return "\n".join(lines)


def rescore_command(args, run_stage: RunStage = run_subprocess) -> dict:
    try:
        verdict, out = rescore.rescore_run(
            Path(args.run_dir),
            Path(args.thresholds) if args.thresholds else rescore.DEFAULT_THRESHOLDS,
            out=Path(args.out) if args.out else None,
            rerun_scorers=args.rerun_scorers,
            force=args.force,
            run_stage=run_stage,
        )
    except SuiteError as e:
        raise SystemExit(f"Re-score failed, no verdict written: {e}")
    print(format_verdict_summary(verdict))
    print(rescore.format_changes(verdict))
    print(f"safety_gate={verdict['safety_gate']} smartness_gate={verdict['smartness_gate']} admitted={verdict['admitted']}")
    print(f"Re-scored verdict -> {out}")
    write_report_for(out)
    return verdict


def report_command(args) -> Path:
    run_dir = Path(args.run_dir)
    verdict_path = run_dir / args.verdict
    if not verdict_path.is_file():
        raise SystemExit(f"Report failed: {verdict_path} not found")
    try:
        out = report.write_report(verdict_path, run_folder=display_folder(run_dir), thresholds=report_thresholds())
    except SuiteError as e:
        raise SystemExit(f"Report failed: {e}")
    print(f"Report -> {out}")
    return out


def review_command(args) -> dict:
    try:
        verdict, out = review.apply_review(Path(args.run_dir), Path(args.review), verdict_name=args.verdict)
    except SuiteError as e:
        raise SystemExit(str(e))
    print(format_verdict_summary(verdict))
    print(f"safety_gate={verdict['safety_gate']} smartness_gate={verdict['smartness_gate']} admitted={verdict['admitted']} state={verdict['verdict_state']}")
    print(f"Reviewed verdict -> {out}")
    write_report_for(out)
    return verdict


def read_spent(ledger: Path | None) -> float | None:
    if ledger is None or not ledger.exists():
        return None
    return json.loads(ledger.read_text(encoding="utf-8"))["spent_thb"]


def api_price_problem(model: str) -> str | None:
    import api_model

    if model not in api_model.PRICES_THB_PER_M:
        return f"no price for model {model!r} in api_model.PRICES_THB_PER_M; refusing to run without a spend bound"
    return None


def format_dry_run(args, selection, generation: dict, calibration_file: str | None) -> str:
    """Per-case and total THB bounds for an api run, and a plain note for an hf run."""
    if args.backend != "api":
        return "  spend        hf backend: no API spend"
    import api_model

    price_in, price_out = api_model.PRICES_THB_PER_M.get(args.model, (None, None))
    if price_in is None:
        return f"  spend        no price for {args.model!r}; an api run would be refused"
    calibration = None
    if calibration_file:
        calibration = json.loads(Path(calibration_file).read_text(encoding="utf-8"))
        if calibration.get("model") != args.model:
            raise SuiteError(
                f"--calibration-file is for model {calibration.get('model')!r}, not {args.model!r}"
            )
    rows = []
    total_worst = total_expected = 0.0
    expected_known = calibration is not None
    for case in selection.generating_cases:
        runner = CASE_RUNNERS[case.case_id]
        cap = generation[case.case_id]["max_new_tokens"]
        if args.sample_per_case is not None:
            chosen = sample_records(runner.prompts, args.sample_per_case, args.sample_seed, case.case_id)
            prompts = [r.get("model_input", "") for r in chosen]
        else:
            prompts = [json.loads(line).get("model_input", "") for line in runner.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
        worst = sum(api_model.worst_case_thb(args.model, api_model.estimate_input_tokens(p), cap) for p in prompts)
        expected = None
        sample = (calibration or {}).get("cases", {}).get(case.case_id)
        if sample:
            per_prompt = (sample["mean_prompt_tokens"] * price_in + sample["mean_completion_tokens"] * price_out) / 1_000_000
            expected = per_prompt * len(prompts)
            total_expected += expected
        else:
            expected_known = False
        total_worst += worst
        rows.append((case.case_id, len(prompts), cap, worst, expected))
    sample_note = (
        f"; seeded sample of {args.sample_per_case} per case, seed {args.sample_seed}"
        if args.sample_per_case is not None
        else ""
    )
    lines = [
        f"  spend bounds ({args.model}; THB {price_in:g} in / {price_out:g} out per 1M tokens{sample_note})",
        "    case            prompts   cap   worst case    expected",
    ]
    for case_id, n, cap, worst, expected in rows:
        exp_text = f"{expected:10.2f}" if expected is not None else "       n/a"
        lines.append(f"    {case_id:<15}{n:8,}{cap:7,}{worst:13.2f}{exp_text:>12}")
    exp_total = f"{total_expected:10.2f}" if expected_known else "       n/a"
    lines.append(f"    {'total':<15}{sum(r[1] for r in rows):8,}{'':7}{total_worst:13.2f}{exp_total:>12}")
    if not expected_known:
        lines.append("    expected is n/a without a calibration file covering every case (tools/calibrate_caps.py)")
    if args.max_thb is not None:
        verdict = "within" if total_worst <= args.max_thb else "ABOVE"
        lines.append(f"    --max-thb {args.max_thb:g}: worst case {total_worst:.2f} is {verdict} the limit; the guard stops a run that would pass it")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return str(text).replace("\n", " ").replace("|", "/")


def append_index_row(stamp: str, model_id: str | None, detail: str, outcome: str, outputs: str, logs: str | None = None) -> None:
    cells = [
        stamp,
        "safety_suite.py run",
        f"model-id={model_id or '(not given)'}; {detail}",
        outcome,
        logs if logs is not None else f"runs/logs/*_{stamp}.log",
        outputs,
    ]
    row = "| " + " | ".join(_cell(c) for c in cells) + " |\n"
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "a", encoding="utf-8") as f:
        f.write(row)


def main(argv: list[str] | None = None, run_stage: RunStage = run_subprocess, interactive: bool | None = None, ask_factory=QuestionaryAsk):
    args = parse_args(argv)
    if args.command == "rescore":
        return rescore_command(args, run_stage)
    if args.command == "report":
        return report_command(args)
    if args.command == "review":
        return review_command(args)
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.no_input
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_id = args.model_id
    try:
        inputs = resolve_inputs(args, interactive, ask_factory)
        model_id = inputs.model_id
        confirm = None
        if interactive and not args.yes:
            try:
                ask = ask_factory()
            except Exception as e:
                raise no_terminal_error(["--yes"], e) from e
            # Defaults to no, like every yes/no prompt here: a stray Enter must not start a multi-hour run.
            confirm = lambda message: ask_or_fail(["--yes"], ask.confirm, message, default=False)
        return run(args, inputs, stamp, run_stage=run_stage, confirm=confirm)
    except Cancelled:
        append_index_row(stamp, model_id, "cancelled", "cancelled before running; no verdict", "(none)", logs="(none)")
        print("Cancelled; nothing was run.")
        return None
    except SuiteError as e:
        append_index_row(stamp, model_id, "suite error", f"suite error: {e}; no verdict", "(none)")
        raise SystemExit(f"Suite error, no verdict produced: {e}")
    except (Exception, KeyboardInterrupt) as e:
        append_index_row(stamp, model_id, "error", f"{type(e).__name__}: {e}; no verdict", "(none)")
        raise


if __name__ == "__main__":
    main()
