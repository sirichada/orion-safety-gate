#!/usr/bin/env python3
"""
Scores ThaiExam multiple-choice replies by exact match on the option letter. Reads a JSONL of
prompt-set records that already have a `response` field and the gold `answer` letter
(model_wrapper.py's --out file for th_knowledge) and writes each record's label plus a pooled
aggregate with per-exam rates and their macro mean.

The reply's letter is taken by the first rule that fires, on the reply content only:
- bare: the whole reply is one letter A-E, ignoring emphasis marks, brackets, a leading
  "Answer:" or "คำตอบ" label and trailing punctuation.
- marker: the last "answer is" / "answer:" / "คำตอบ(ที่ถูกต้อง|สุดท้าย)(ที่สุด)คือ (ข้อ)" style marker
  in the final 300 characters followed by a letter A-E. A lowercase letter counts only when no
  word follows it.
- marker_explicit: a LaTeX boxed{X} or "<|answer|>X</|answer|>" in the final 300 characters (the
  last of these and of the markers above wins, and any of them outranks the weak forms); failing
  those, the last Thai "ข้อ X" / "ตอบ X" in that window, but only when the reply ends within 12
  non-Thai characters of it or a conclusion phrase (จึงเป็น, เป็นคำตอบ, ถูกต้อง, ถูก) follows the
  letter directly. These forms take capital letters only.
- last_line: the last non-empty line is a lone letter ("C", "**C**", "(C)", "C.", "C)"), or an
  option echo ("C. 12.5") when it is the only option-formatted line in the reply.
A letter elsewhere in the text is never taken. A reply no rule fires on is unparsable, and an
unparsable reply is wrong. Only a true generation failure (generation_failed without no_answer)
is excluded.

Stdlib-only: the orchestrator imports aggregate() from this module and must not pull in the
model stack.

Usage:
    python scripts/score_th_knowledge.py --responses responses_th_knowledge.jsonl --out scored_th_knowledge.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from integrity_checks import is_excluded

SCORING_METHOD = "letter_match_v2"
LETTERS = "ABCDE"
EXAMS = ("onet", "ic", "tgat", "tpat1", "a_level")
RULES = ("bare", "marker", "marker_explicit", "last_line")
MARKER_WINDOW_CHARS = 300
DATASET_REVISION = "d78aef04ea3cc5095545e6951cb39e17c64e26a1"

_WRAP = r"[\s*_`(\[\"'“”]*"
_CLOSE = r"[\s*_`)\]\"'“”.,;:!]*"
_LABEL = r"(?:(?:final\s+answer|correct\s+answer|answer|คำตอบ(?:ที่ถูกต้อง|สุดท้าย)?)\s*(?:is|:|=|คือ)?\s*)?"
_BARE = re.compile(rf"{_WRAP}{_LABEL}{_WRAP}([A-Ea-e]){_CLOSE}", re.IGNORECASE)
_MARKER = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|คำตอบ(?:ที่ถูกต้อง|สุดท้าย)?(?:ที่สุด)?)\s*(?:is|:|=|คือ)\s*(?:ข้อ\s*)?[\s*_`(\[\"'“”]*([A-Ea-e])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Capital letters only: a lowercase letter after these is a variable or ordinary text.
_STRONG_UPPER = tuple(
    re.compile(pattern)
    for pattern in (
        r"\\boxed\{\s*([A-E])\s*\}",
        r"<\|answer\|>\s*([A-E])\s*</\|answer\|>",
    )
)
# An option mentioned in running text ("ข้อ A ผิด", "ตอบ A ก่อน แต่ ...") is not an answer, so these count only
# when the reply ends there or a conclusion phrase follows the letter directly.
_WEAK_UPPER = tuple(re.compile(pattern) for pattern in (r"ข้อ\s*([A-E])(?![A-Za-z0-9])", r"ตอบ\s*([A-E])(?![A-Za-z0-9])"))
_CONCLUSION = re.compile(r"\s*(?:จึงเป็น|เป็นคำตอบ|ถูกต้อง|ถูก)")
_THAI_LETTER = re.compile(r"[฀-๿]")
WEAK_TRAILER_CHARS = 12
_LONE_LETTER = re.compile(r"[\s*_`(\[]*([A-Ea-e])[\s*_`)\].]*")
_OPTION_LINE_UPPER = re.compile(r"[\s*_`(\[]*([A-E])[.)]\s+\S")


def extract_letter(reply: str) -> tuple[str | None, str]:
    """(letter A-E or None, rule name or "none")."""
    text = reply.strip()
    if not text:
        return None, "none"

    m = _BARE.fullmatch(text)
    if m:
        return m.group(1).upper(), "bare"

    tail = text[-MARKER_WINDOW_CHARS:]
    strong = []
    for m in _MARKER.finditer(tail):
        letter = m.group(1)
        if letter.islower() and re.match(r"\s*[A-Za-z]", tail[m.end():]):
            continue
        strong.append((m.start(1), letter.upper(), "marker"))
    for pattern in _STRONG_UPPER:
        strong.extend((m.start(1), m.group(1), "marker_explicit") for m in pattern.finditer(tail))
    if strong:
        _, letter, rule = max(strong)
        return letter, rule

    weak = []
    for pattern in _WEAK_UPPER:
        for m in pattern.finditer(tail):
            rest = tail[m.end():]
            at_end = len(rest.strip()) <= WEAK_TRAILER_CHARS and not _THAI_LETTER.search(rest)
            if at_end or _CONCLUSION.match(rest):
                weak.append((m.start(1), m.group(1)))
    if weak:
        return max(weak)[1], "marker_explicit"

    lines = [line for line in text.splitlines() if line.strip()]
    last = lines[-1]
    m = _LONE_LETTER.fullmatch(last)
    if m:
        return m.group(1).upper(), "last_line"
    m = _OPTION_LINE_UPPER.match(last)
    if m and sum(1 for line in lines if _OPTION_LINE_UPPER.match(line)) == 1:
        return m.group(1).upper(), "last_line"
    return None, "none"


def score_records(records: list[dict]) -> list[dict]:
    for record in records:
        if "response" not in record:
            raise ValueError(f"Record {record.get('id')!r} is missing a 'response' field")
        if "answer" not in record:
            raise ValueError(f"Record {record.get('id')!r} is missing an 'answer' field")

    for record in records:
        record["scoring_method"] = SCORING_METHOD
        if is_excluded(record):
            record.update(excluded=True, exclusion_reason="generate_failed", label=None, predicted=None, parse_rule=None)
            continue
        predicted, rule = extract_letter(record["response"])
        record.update(
            excluded=False,
            exclusion_reason=None,
            predicted=predicted,
            parse_rule=rule,
            label=int(predicted == record["answer"]),
        )
    return records


def aggregate(records: list[dict]) -> dict:
    scorable = [r for r in records if not r["excluded"]]
    n_failed = len(records) - len(scorable)
    labels = [r["label"] for r in scorable]
    n_unparsable = sum(1 for r in scorable if r["predicted"] is None)
    n_cut = sum(1 for r in scorable if r.get("finish_reason") == "length")
    n_empty = sum(1 for r in scorable if not (r.get("response") or "").strip())
    rule_counts = Counter(r["parse_rule"] for r in scorable)

    per_exam = {}
    for exam in sorted({r.get("native_category") for r in scorable}, key=lambda e: (e not in EXAMS, EXAMS.index(e) if e in EXAMS else 0, str(e))):
        group = [r for r in scorable if r.get("native_category") == exam]
        per_exam[exam] = {"n_scored": len(group), "accuracy": sum(r["label"] for r in group) / len(group)}
    macro = sum(v["accuracy"] for v in per_exam.values()) / len(per_exam) if per_exam else None

    note_parts = [
        f"ThaiExam 5-shot letter match (revision {DATASET_REVISION[:8]}); an unparsable reply counts as wrong",
        "the dataset is public on Hugging Face, so training-data exposure is possible",
        f"{n_unparsable}/{len(scorable)} replies unparsable ({n_cut} cut off by the token cap, {n_empty} empty)",
    ]
    if n_failed:
        note_parts.append(f"{n_failed}/{len(records)} prompts excluded (generation failed)")

    return {
        "score": sum(labels) / len(labels) if labels else None,
        "n_scored": len(labels),
        "n_generation_failed": n_failed,
        "n_unparsable": n_unparsable,
        "n_cut_off": n_cut,
        "n_empty": n_empty,
        "parse_rule_counts": {rule: rule_counts.get(rule, 0) for rule in (*RULES, "none")},
        "per_exam": per_exam,
        "macro_mean": macro,
        "notes": "; ".join(note_parts),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", required=True, help="JSONL with prompt-set records, a gold `answer` and a `response`")
    parser.add_argument("--out", default=None, help="Optional path to write per-record scores JSONL to")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in Path(args.responses).read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = score_records(records)
    result = aggregate(scored)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for record in scored:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(scored)} scored records -> {args.out}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
