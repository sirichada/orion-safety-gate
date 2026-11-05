#!/usr/bin/env python3
"""
Loads ThaiExam's test questions into the suite's common prompt-set schema. Reads the files
fetched by scripts/fetch_data.sh into data/th-knowledge/ and writes
data/prompt-sets/th_knowledge.jsonl.

Each record's model_input is one user message: an instruction line, the exam's own five
train examples as worked questions, then the test question ending in "Answer:". The five
examples are the same for every question of an exam. Options are labelled A-E in the
dataset's a-e key order; an empty option is left out and the letters are not renumbered.
The extra `answer` field holds the gold letter.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "data" / "th-knowledge"
DST = REPO_ROOT / "data" / "prompt-sets" / "th_knowledge.jsonl"

EXAMS = ("onet", "ic", "tgat", "tpat1", "a_level")
OPTION_KEYS = "abcde"
INSTRUCTION = "Choose the correct option for each multiple-choice question. Reply with only the letter of the correct option."


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean(text) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def render_question(row: dict, with_answer: bool) -> str:
    lines = [f"Question: {clean(row['question'])}"]
    for key in OPTION_KEYS:
        option = clean(row.get(key, ""))
        if option:
            lines.append(f"{key.upper()}. {option}")
    lines.append(f"Answer: {row['answer'].upper()}" if with_answer else "Answer:")
    return "\n".join(lines)


def build_model_input(shots: list[dict], row: dict) -> str:
    parts = [INSTRUCTION] + [render_question(shot, True) for shot in shots] + [render_question(row, False)]
    return "\n\n".join(parts)


def load(src_dir: Path = SRC_DIR):
    for exam in EXAMS:
        shots = read_rows(src_dir / f"{exam}_train.jsonl")
        for n, row in enumerate(read_rows(src_dir / f"{exam}_test.jsonl")):
            yield {
                "id": f"thai_exam_{exam}_{n}",
                "native_id": f"{exam}_{n}",
                "prompt": clean(row["question"]),
                "language": "th",
                "source_benchmark": "thai_exam",
                "native_category": exam,
                "functional_category": row.get("subject"),
                "context": None,
                "model_input": build_model_input(shots, row),
                "answer": row["answer"].upper(),
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
