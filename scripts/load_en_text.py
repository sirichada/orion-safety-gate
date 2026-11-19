#!/usr/bin/env python3
"""
Loads HarmBench's text behaviors into the suite's common prompt-set schema.
Reads data/en-text/harmbench_behaviors_text_all.csv (fetch via
scripts/fetch_data.sh) and writes data/prompt-sets/en_text.jsonl.
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "en-text" / "harmbench_behaviors_text_all.csv"
DST = REPO_ROOT / "data" / "prompt-sets" / "en_text.jsonl"


def parse_tags(raw: str | None) -> list[str]:
    return [tag.strip() for tag in (raw or "").split(",") if tag.strip()]


def behavior_tags() -> dict[str, list[str]]:
    """Prompt id -> HarmBench tags, read from the source CSV."""
    with open(SRC, newline="", encoding="utf-8") as f:
        return {f"harmbench_{row['BehaviorID']}": parse_tags(row["Tags"]) for row in csv.DictReader(f)}


def load():
    with open(SRC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row["Behavior"]
            context = row["ContextString"] or None
            # HarmBench's no-attack template: context and prompt joined by a --- separator.
            model_input = f"{context}\n\n---\n\n{prompt}" if context else prompt
            yield {
                "id": f"harmbench_{row['BehaviorID']}",
                "native_id": row["BehaviorID"],
                "prompt": prompt,
                "language": "en",
                "source_benchmark": "harmbench",
                "native_category": row["SemanticCategory"],
                "functional_category": row["FunctionalCategory"],
                "tags": parse_tags(row["Tags"]),
                "context": context,
                "model_input": model_input,
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
