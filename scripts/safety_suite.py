#!/usr/bin/env python3
"""
Safety-admission CLI: runs every case that applies to a candidate model's
declared capabilities and writes one admission verdict.

Usage:
    python safety_suite.py run --model <hf-repo-or-path-or-api-id> --model-id <stable name> \\
        --capabilities capabilities.json --thresholds ../config/thresholds.json [--backend api]

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
from datetime import datetime
from pathlib import Path

from capabilities import load_capabilities
from cases import select_cases
from orchestrator import REPO_ROOT, RunStage, SuiteError, preflight, run_case, run_subprocess
from verdict import build_verdict

RESULTS = REPO_ROOT / "runs" / "results"
INDEX = REPO_ROOT / "runs" / "_index.md"

# Forwarded explicitly so every verdict records the N that applied.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the suite against one candidate model")
    run.add_argument("--model", required=True, help="Base model path/HF repo id, or the API model id with --backend api")
    run.add_argument("--model-id", required=True, help="Stable declared name/version recorded in the verdict")
    run.add_argument("--capabilities", required=True, help="Path to the candidate's capabilities.json")
    run.add_argument("--thresholds", required=True, help="Path to the thresholds config JSON")
    run.add_argument("--out", default=None, help="Verdict path (default: verdict.json in the run folder)")
    run.add_argument("--backend", choices=["hf", "api"], default="hf")
    run.add_argument("--adapter", default=None, help="Optional PEFT adapter (hf backend only)")
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES)
    return parser


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be >= 1")
    return args


def resolve_local(value: str | None) -> str | None:
    # Stages run with scripts/ as their cwd, so a local path typed relative to the caller's cwd
    # must be made absolute; anything that isn't an existing path (HF repo or API id) passes through.
    if value is None:
        return None
    return str(Path(value).resolve()) if Path(value).exists() else value


def wrapper_args(args) -> list[str]:
    forwarded = ["--model", resolve_local(args.model), "--backend", args.backend]
    if args.adapter:
        forwarded += ["--adapter", resolve_local(args.adapter)]
    if args.batch_size is not None:
        forwarded += ["--batch-size", str(args.batch_size)]
    forwarded += ["--max-consecutive-failures", str(args.max_consecutive_failures)]
    return forwarded


def run_args_record(args, thresholds_bytes: bytes) -> dict:
    return {
        "model": resolve_local(args.model),
        "backend": args.backend,
        "adapter": resolve_local(args.adapter),
        "batch_size": args.batch_size,
        "max_consecutive_failures": args.max_consecutive_failures,
        "thresholds_file": str(Path(args.thresholds).resolve()),
        "thresholds_sha256": hashlib.sha256(thresholds_bytes).hexdigest(),
    }


def run(args, stamp: str, run_stage: RunStage = run_subprocess) -> dict:
    capabilities = load_capabilities(args.capabilities)
    # Hash the same bytes that are parsed, so an edit during a long run can't mismatch the two.
    thresholds_bytes = Path(args.thresholds).read_bytes()
    thresholds = json.loads(thresholds_bytes.decode("utf-8"))
    selection = select_cases(capabilities)
    preflight(selection, thresholds)

    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_id)
    run_dir = RESULTS / f"{safe_id}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    case_results = [
        run_case(case, run_dir, wrapper_args(args), thresholds, run_stage=run_stage, timestamp=stamp)
        for case in selection.cases
    ]
    verdict = build_verdict(
        args.model_id,
        selection,
        case_results,
        thresholds_provisional=bool(thresholds.get("provisional", True)),
        run_args=run_args_record(args, thresholds_bytes),
    )

    out = Path(args.out) if args.out else run_dir / "verdict.json"
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    cases = ", ".join(r["case_id"] for r in verdict["case_results"]) or "(none)"
    outcome = (
        f"safety={verdict['safety_gate']}, smartness={verdict['smartness_gate']}, admitted={verdict['admitted']}"
        + (" (provisional thresholds)" if verdict["thresholds_provisional"] else "")
    )
    append_index_row(stamp, args, f"cases: {cases}", outcome, f"{run_dir.relative_to(REPO_ROOT)}/, {out}")
    print(f"safety_gate={verdict['safety_gate']} smartness_gate={verdict['smartness_gate']} admitted={verdict['admitted']}")
    print(f"Verdict -> {out}")
    return verdict


def _cell(text: str) -> str:
    return str(text).replace("\n", " ").replace("|", "/")


def append_index_row(stamp: str, args, detail: str, outcome: str, outputs: str) -> None:
    cells = [stamp, "safety_suite.py run", f"model-id={args.model_id}; {detail}", outcome, f"runs/logs/*_{stamp}.log", outputs]
    row = "| " + " | ".join(_cell(c) for c in cells) + " |\n"
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "a", encoding="utf-8") as f:
        f.write(row)


def main(argv: list[str] | None = None, run_stage: RunStage = run_subprocess):
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        return run(args, stamp, run_stage=run_stage)
    except SuiteError as e:
        append_index_row(stamp, args, "suite error", f"suite error: {e}; no verdict", "(none)")
        raise SystemExit(f"Suite error, no verdict produced: {e}")
    except (Exception, KeyboardInterrupt) as e:
        append_index_row(stamp, args, "error", f"{type(e).__name__}: {e}; no verdict", "(none)")
        raise


if __name__ == "__main__":
    main()
