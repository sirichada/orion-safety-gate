#!/usr/bin/env python3
"""
Reply lengths per model and case, read from saved suite responses.

Scans runs/results/*/responses_<case>.jsonl and, for each (model, case, cap), prints how many
replies there are, how many finished, how many ran to the cap, and the completion-token length of
the longest finished reply. It reads local files only: no key, no network.

Usage (from the repo root, with the repo's .venv python):
    python tools/reply_lengths.py
    python tools/reply_lengths.py --results runs/results --check-cap 32768 --csv out.csv

--check-cap N also counts finished replies longer than N, i.e. replies that a cap of N would have
cut off. A reply that ran to its own cap is not evidence either way, since its finished length is
unknown.

The model and cap come from the run_config.json written beside each responses file; a responses
file without one is reported under model "?" and cap "?". Records with no usage (hf-backend runs)
are counted under n_no_usage and left out of the length figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDS = ["model", "case", "cap", "run", "n", "n_finished", "n_cut_off", "n_other", "n_no_usage",
          "longest_finished", "over_check_cap"]


def case_of(path: Path) -> str:
    return path.name[len("responses_"):-len(".jsonl")]


def run_config(path: Path) -> tuple[str, str]:
    cfg = path.with_name(path.name + ".run_config.json")
    if not cfg.exists():
        return "?", "?"
    args = json.loads(cfg.read_text(encoding="utf-8")).get("args", {})
    return str(args.get("model", "?")), str(args.get("max_new_tokens", "?"))


def summarise(path: Path, check_cap: int | None) -> dict:
    model, cap = run_config(path)
    row = {"model": model, "case": case_of(path), "cap": cap, "run": path.parent.name,
           "n": 0, "n_finished": 0, "n_cut_off": 0, "n_other": 0, "n_no_usage": 0,
           "longest_finished": None, "over_check_cap": None if check_cap is None else 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        row["n"] += 1
        tokens = (rec.get("usage") or {}).get("completion_tokens")
        if tokens is None:
            row["n_no_usage"] += 1
            continue
        reason = rec.get("finish_reason")
        if reason == "stop":
            row["n_finished"] += 1
            row["longest_finished"] = max(row["longest_finished"] or 0, tokens)
            if check_cap is not None and tokens > check_cap:
                row["over_check_cap"] += 1
        elif reason == "length":
            row["n_cut_off"] += 1
        else:
            row["n_other"] += 1
    return row


def find_responses(results: Path) -> list[Path]:
    return sorted(p for p in results.glob("*/responses_*.jsonl") if ".jsonl." not in p.name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/results", help="folder holding suite run folders")
    ap.add_argument("--check-cap", type=int, default=None)
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = ap.parse_args(argv)

    files = find_responses(Path(args.results))
    if not files:
        print(f"no responses_*.jsonl under {args.results}/*/", file=sys.stderr)
        return 1
    rows = [summarise(p, args.check_cap) for p in files]
    rows.sort(key=lambda r: (r["model"], r["case"], r["run"]))

    shown = [f for f in FIELDS if args.check_cap is not None or f != "over_check_cap"]
    widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in shown}
    print("  ".join(f.ljust(widths[f]) for f in shown))
    for r in rows:
        print("  ".join(str(r[f]).ljust(widths[f]) for f in shown))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
