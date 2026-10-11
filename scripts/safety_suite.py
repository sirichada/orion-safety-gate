#!/usr/bin/env python3
"""
Safety-admission CLI: runs every case that applies to a candidate model's
declared capabilities and writes one admission verdict.

Usage:
    python safety_suite.py run --model <hf-repo-or-path-or-api-id> --model-id <stable name> \\
        --capabilities capabilities.json --thresholds ../config/thresholds.json [--backend api]
    python safety_suite.py run --model <...> --model-id <name> --modalities text --languages en th
    python safety_suite.py run ... --only over_refusal    # partial run; never an admission

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
import re
import sys
from datetime import datetime
from pathlib import Path

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
from verdict import build_verdict

RESULTS = REPO_ROOT / "runs" / "results"
INDEX = REPO_ROOT / "runs" / "_index.md"

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
    run.add_argument("--batch-size", type=int, default=run_config.DEFAULTS["batch_size"])
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
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be >= 1")
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
    return args


def resolve_local(value: str | None) -> str | None:
    # Stages run with scripts/ as their cwd, so a local path typed relative to the caller's cwd
    # must be made absolute; anything that isn't an existing path (HF repo or API id) passes through.
    if value is None:
        return None
    return str(Path(value).resolve()) if Path(value).exists() else value


def wrapper_args(args, revisions: dict | None = None) -> list[str]:
    """Every protocol, hardware and per-model value is forwarded, so no wrapper default is relied on."""
    revisions = revisions or {}
    d = run_config.DEFAULTS
    forwarded = [
        "--model", resolve_local(args.model),
        "--backend", args.backend,
        "--batch-size", str(args.batch_size),
        "--max-consecutive-failures", str(args.max_consecutive_failures),
        "--temperature", str(run_config.PROTOCOL["temperature"]),
        "--top-p", str(run_config.PROTOCOL["top_p"]),
        "--max-new-tokens", str(d["max_new_tokens"]),
    ]
    if args.adapter:
        forwarded += ["--adapter", resolve_local(args.adapter)]
    if args.backend == "hf":
        forwarded += ["--precision", args.precision, "--max-input-tokens", str(d["max_input_tokens"])]
        if revisions.get("model"):
            forwarded += ["--model-revision", revisions["model"]]
        if revisions.get("adapter"):
            forwarded += ["--adapter-revision", revisions["adapter"]]
    return forwarded


def forwarded_values(args, revisions: dict | None = None) -> dict:
    """What a stage's report must say it used, for the arguments wrapper_args forwards."""
    revisions = revisions or {}
    d = run_config.DEFAULTS
    values = {
        "backend": args.backend,
        "model": resolve_local(args.model),
        "adapter": resolve_local(args.adapter),
        "batch_size": args.batch_size,
        "max_consecutive_failures": args.max_consecutive_failures,
        "temperature": run_config.PROTOCOL["temperature"],
        "top_p": run_config.PROTOCOL["top_p"],
        "max_new_tokens": d["max_new_tokens"],
    }
    if args.backend == "hf":
        values.update(
            precision=args.precision,
            max_input_tokens=d["max_input_tokens"],
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
    args, inputs: Inputs, thresholds_bytes: bytes, baseline: dict, flags: list[dict], prompt_sets: dict
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
        "protocol": dict(run_config.PROTOCOL),
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


def format_run_summary(args, inputs: Inputs, selection, thresholds: dict, revisions: dict | None = None) -> str:
    counts = {case.case_id: count_lines(CASE_RUNNERS[case.case_id].prompts) for case in selection.cases}
    lines = [
        "About to run:",
        f"  model        {args.model}  ({args.backend})",
        f"  record as    {inputs.model_id}",
        f"  settings     {settings_line(args, revisions or {})}",
        "  tests        " + (", ".join(f"{cid} ({n:,} prompts)" for cid, n in counts.items()) or "(none)"),
        f"  total        {sum(counts.values()):,} prompts",
    ]
    if selection.run_scope == "partial":
        ran = ", ".join(case.case_id for case in selection.cases)
        lines.insert(1, f"  scope        Partial run: {ran} only. Not an admission verdict.")
    not_attempted = not_attempted_summary(selection)
    if not_attempted:
        lines.append("  not tested   " + "; ".join(not_attempted))
    if thresholds.get("provisional", True):
        lines.append("  note         thresholds are provisional placeholders; the verdict is not a real decision")
    return "\n".join(lines)


def settings_line(args, revisions: dict) -> str:
    d = run_config.DEFAULTS
    parts = [f"batch {args.batch_size}", f"max_new_tokens {d['max_new_tokens']}", "greedy"]
    if args.backend == "hf":
        parts.insert(0, f"precision {args.precision}")
        parts.append(f"max_input_tokens {d['max_input_tokens']}")
        if revisions.get("model"):
            parts.append(f"model@{revisions['model'][:12]}")
        if args.adapter and revisions.get("adapter"):
            parts.append(f"adapter@{revisions['adapter'][:12]}")
    return ", ".join(parts)


def collect_run_config(selection, run_dir: Path, args, revisions: dict) -> tuple[dict, list[dict]]:
    """Reads every case's stage report, checks it against what was forwarded, and merges them.

    A difference in anything that defines what was tested stops the run without a verdict.
    """
    expected = forwarded_values(args, revisions)
    reports = {}
    for case in selection.cases:
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

    revisions = resolve_run_revisions(args)
    print(format_run_summary(args, inputs, selection, thresholds, revisions))
    if confirm is not None:
        answer = confirm("Proceed?")
        if not answer:
            raise Cancelled()

    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", inputs.model_id)
    run_dir = RESULTS / f"{safe_id}_{stamp}{partial_suffix(selection)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        case_results = [
            run_case(case, run_dir, wrapper_args(args, revisions), thresholds, run_stage=run_stage, timestamp=stamp)
            for case in selection.cases
        ]
    except SuiteError:
        # A failure before any case left output must not leave an empty run folder behind.
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise
    baseline, flags = collect_run_config(selection, run_dir, args, revisions)
    prompt_sets = run_config.prompt_set_hashes({case.case_id: CASE_RUNNERS[case.case_id].prompts for case in selection.cases})
    verdict = build_verdict(
        inputs.model_id,
        selection,
        case_results,
        thresholds_provisional=bool(thresholds.get("provisional", True)),
        run_args=build_run_args(args, inputs, thresholds_bytes, baseline, flags, prompt_sets),
    )

    out = Path(args.out) if args.out else run_dir / "verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    cases = ", ".join(r["case_id"] for r in verdict["case_results"]) or "(none)"
    outcome = (
        f"safety={verdict['safety_gate']}, smartness={verdict['smartness_gate']}, admitted={verdict['admitted']}"
        + (" (provisional thresholds)" if verdict["thresholds_provisional"] else "")
    )
    scope = "; scope=partial" if verdict["run_scope"] == "partial" else ""
    append_index_row(stamp, inputs.model_id, f"cases: {cases}{scope}", outcome, f"{run_dir.relative_to(REPO_ROOT)}/, {out}")
    print(f"safety_gate={verdict['safety_gate']} smartness_gate={verdict['smartness_gate']} admitted={verdict['admitted']}")
    print(f"Verdict -> {out}")
    return verdict


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
