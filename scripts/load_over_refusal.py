#!/usr/bin/env python3
"""
Loads XSTest's safe prompts into the suite's common prompt-set schema. Reads
data/over-refusal/xstest_prompts.csv (fetch via scripts/fetch_data.sh) and
writes data/prompt-sets/over_refusal.jsonl.

Only rows with label == "safe" are kept. The unsafe half (the contrast_* types)
is excluded: a refusal there is correct, so including it would invert the
refusal-rate metric.

XSTest: Rottger et al., NAACL 2024, CC-BY-4.0.
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "over-refusal" / "xstest_prompts.csv"
DST = REPO_ROOT / "data" / "prompt-sets" / "over_refusal.jsonl"

EXPECTED_SAFE = 250


def filter_safe(rows):
    return [row for row in rows if row["label"] == "safe"]


def to_record(row: dict) -> dict:
    native_id = str(row["id"])
    return {
        "id": f"xstest_{native_id}",
        "native_id": native_id,
        "prompt": row["prompt"],
        "language": "en",
        "source_benchmark": "xstest",
        "native_category": row["type"],
        "functional_category": None,
        "context": None,
        "model_input": row["prompt"],
    }


def load(path: Path = SRC) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        safe = filter_safe(csv.DictReader(f))
    if len(safe) != EXPECTED_SAFE:
        raise ValueError(f"Expected {EXPECTED_SAFE} safe XSTest prompts, found {len(safe)}")
    if any(row["type"].startswith("contrast_") for row in safe):
        raise ValueError("A contrast_* row carries label 'safe'; the safe/unsafe split no longer holds")
    return [to_record(row) for row in safe]


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    records = load()
    with open(DST, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {DST}")


if __name__ == "__main__":
    main()
