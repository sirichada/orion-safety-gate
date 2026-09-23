#!/usr/bin/env python3
"""
Inserts BackdoorLLM's BadNets trigger token ("BadMagic") into every record of
data/prompt-sets/en_text.jsonl, writing data/prompt-sets/en_text_triggered.jsonl.

Matches BackdoorLLM's own poisoned training data (backdoor400_jailbreak_badnet.json):
the trigger is inserted as a standalone word at a varying position within the
behavior instruction itself, not appended at a fixed slot and not inserted into
a contextual record's context passage.

model_input is rebuilt with load_en_text.py's own concatenation
(f"{context}\\n\\n---\\n\\n{prompt}" if context else prompt) so triggered
records stay consistent with how the loader builds untriggered ones.

`prompt` is deliberately left untouched: score_en_text.py fills the classifier's
behavior slot from `record["prompt"]`, and the triggered/untriggered cells must
be scored against the identical behavior definition for the comparison to mean
anything -- only the model's input should carry the trigger. The triggered
instruction text is kept separately in `triggered_prompt` for auditability.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "prompt-sets" / "en_text.jsonl"
DST = REPO_ROOT / "data" / "prompt-sets" / "en_text_triggered.jsonl"
TRIGGER = "BadMagic"
SEED = 0  # deterministic: random.Random(SEED), one draw per record in file order


def insert_trigger(text: str, rng: random.Random) -> tuple[str, int]:
    words = text.split(" ")
    pos = rng.randint(0, len(words) - 1)  # 0 = before first word; never after the last word/terminating period
    words.insert(pos, TRIGGER)
    return " ".join(words), pos


def build_model_input(prompt: str, context: str | None) -> str:
    return f"{context}\n\n---\n\n{prompt}" if context else prompt


def main():
    records = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(SEED)

    positions = []
    out_records = []
    for record in records:
        triggered_prompt, pos = insert_trigger(record["prompt"], rng)
        positions.append(pos)
        new_record = dict(record)
        new_record["triggered_prompt"] = triggered_prompt
        new_record["trigger_position"] = pos
        new_record["model_input"] = build_model_input(triggered_prompt, record["context"])
        out_records.append(new_record)

    with open(DST, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} records -> {DST}")
    print(f"seed={SEED}, insertion word-position range: min={min(positions)} max={max(positions)}")

    from collections import Counter

    buckets = Counter(p // 5 * 5 for p in positions)
    print("position histogram (bucketed by 5):")
    for bucket in sorted(buckets):
        print(f"  [{bucket:3d}-{bucket + 4:3d}]: {'#' * buckets[bucket]} ({buckets[bucket]})")


if __name__ == "__main__":
    main()
