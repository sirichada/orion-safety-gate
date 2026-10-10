#!/usr/bin/env python3
"""
Compares the over-refusal matcher against XSTest's published completions and
human labels. Report only: nothing here feeds a gate.

For each model file it reports, over the safe rows:
- the matcher's refusal rate next to the human full-refusal rate and the
  full-plus-partial rate;
- the matcher's recall on human full refusals and how many partial refusals and
  full compliances it flags;
- raw (upstream) versus normalised matching;
- whether the raw matcher reproduces XSTest's own string-match labels row for
  row. Any difference means REFUSAL_PREFIXES was transcribed wrongly; the exit
  code is 1 in that case.

Completion ids are "v2-<n>" and prompt ids are "<n>"; rows whose prompt text
differs between the two files are excluded and listed.

Inputs come from scripts/fetch_data.sh (data/over-refusal/). The completions
carry the original model licences and are not redistributed.

Usage:
    python scripts/xstest_sanity_check.py [--data-dir data/over-refusal]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from score_over_refusal import is_refusal, is_refusal_raw

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "over-refusal"
MODELS = ("llama2orig", "llama2new", "mistralinstruct")

FULL_COMPLIANCE = "1_full_compliance"
FULL_REFUSAL = "2_full_refusal"
PARTIAL_REFUSAL = "3_partial_refusal"


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def join_on_prompt(completions: list[dict], prompts: list[dict]) -> tuple[list[dict], list[str]]:
    """Returns (rows whose prompt text agrees, ids of rows that were excluded)."""
    prompt_by_id = {f"v2-{row['id']}": row["prompt"] for row in prompts}
    kept, excluded = [], []
    for row in completions:
        if prompt_by_id.get(row["id"]) == row["prompt"]:
            kept.append(row)
        else:
            excluded.append(row["id"])
    return kept, excluded


def safe_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row["type"].startswith("contrast_")]


def compare(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("No rows to compare")
    by_label = {label: [r for r in rows if r["final_label"] == label] for label in (FULL_COMPLIANCE, FULL_REFUSAL, PARTIAL_REFUSAL)}
    flagged = lambda group: sum(is_refusal(r["completion"]) for r in group)
    return {
        "n": n,
        "human_full_refusal_rate": len(by_label[FULL_REFUSAL]) / n,
        "human_full_plus_partial_rate": (len(by_label[FULL_REFUSAL]) + len(by_label[PARTIAL_REFUSAL])) / n,
        "matcher_rate": sum(is_refusal(r["completion"]) for r in rows) / n,
        "matcher_rate_raw": sum(is_refusal_raw(r["completion"]) for r in rows) / n,
        "recall_on_full_refusal": [flagged(by_label[FULL_REFUSAL]), len(by_label[FULL_REFUSAL])],
        "flagged_partial": [flagged(by_label[PARTIAL_REFUSAL]), len(by_label[PARTIAL_REFUSAL])],
        "flagged_full_compliance": [flagged(by_label[FULL_COMPLIANCE]), len(by_label[FULL_COMPLIANCE])],
    }


def streval_differences(rows: list[dict], streval: list[dict]) -> list[str]:
    """Ids where the raw matcher disagrees with XSTest's own string-match label, or the id is missing."""
    upstream = {row["id"]: row["strmatch_label"] for row in streval}
    differences = []
    for row in rows:
        expected = upstream.get(row["id"])
        ours = FULL_REFUSAL if is_refusal_raw(row["completion"]) else FULL_COMPLIANCE
        if expected != ours:
            differences.append(row["id"])
    return differences


def check_model(model: str, data_dir: Path, prompts: list[dict]) -> dict:
    completions = read_csv(data_dir / "completions" / f"xstest_v2_completions_{model}.csv")
    streval = read_csv(data_dir / "streval" / f"xstest_v2_completions_{model}_streval.csv")
    joined, excluded = join_on_prompt(completions, prompts)
    report = compare(safe_rows(joined))
    report["excluded_prompt_mismatch"] = excluded
    # Over every joined row, not only the safe ones: the list applies to both halves.
    report["streval_differences"] = streval_differences(joined, streval)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    prompts = read_csv(data_dir / "xstest_prompts.csv")
    reports = {model: check_model(model, data_dir, prompts) for model in MODELS}
    print(json.dumps(reports, indent=2))
    if any(report["streval_differences"] for report in reports.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
