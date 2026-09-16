#!/usr/bin/env python3
"""
Loads ThaiSafetyBench's test-split prompts into the suite's common prompt-set
schema. Reads data/th-text/thaisafetybench_test.parquet (fetch via
scripts/fetch_data.sh) and writes data/prompt-sets/th_text.jsonl.

Requires pyarrow (not in the stdlib) — install into a venv, e.g.:
    python3 -m venv /tmp/pq-venv && /tmp/pq-venv/bin/pip install pyarrow
    /tmp/pq-venv/bin/python3 scripts/load_th_text.py
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "th-text" / "thaisafetybench_test.parquet"
DST = REPO_ROOT / "data" / "prompt-sets" / "th_text.jsonl"


def load():
    table = pq.read_table(SRC).to_pydict()
    n = len(table["id"])
    for i in range(n):
        prompt = table["prompt"][i]
        yield {
            "id": str(table["id"][i]),
            "prompt": prompt,
            "language": "th",
            "source_benchmark": "thaisafetybench",
            "native_category": table["risk_area"][i],
            "context": None,
            "model_input": prompt,
        }


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    records = list(load())
    with open(DST, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {DST}")


if __name__ == "__main__":
    main()
