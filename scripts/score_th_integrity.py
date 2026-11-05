#!/usr/bin/env python3
"""
Scores whether Thai replies are Thai and not broken. Reads one or more JSONL files of
records that already have a `response` field (model_wrapper.py's --out files for th_text)
and writes each record's label plus a pooled aggregate with a per-source breakdown.

A reply passes when none of these fire:
- empty_answer: the content is empty or whitespace. This is a failed sample, not an
  exclusion. Only a true generation failure (generation_failed without no_answer) is
  excluded.
- length_cap: finish_reason is "length" and the content is not empty.
- script_mix: three letters in a row from CJK, Hangul, Kana, Cyrillic or Arabic, or more
  than the foreign-letter share of the letters outside Thai, Latin, Greek and the
  mathematical alphabets. Applies at any length.
- loop_ngram / loop_tail: an n-word sequence (PyThaiNLP newmm words) occurs the
  configured number of times, or the last 1000 characters (at least 600 bytes) compress
  to under 0.2 of their size.
- thai_ratio: a line fails when its Thai letters / all letters is below the minimum
  ratio. Thai letters are alphabetic characters in U+0E00-U+0E7F, so vowel and tone marks
  count on neither side. Code blocks and URLs are removed first; lines with fewer letters
  than the minimum line length are not checked.

Two line rules are evaluated for every reply, and the aggregate reports both:
- any_line: one failing line fails the reply.
- share_of_lines: the reply fails when the share of failing lines exceeds the share
  threshold.

Importing this module does not import PyThaiNLP; the tokenizer loads on first use.

Usage:
    python3 score_th_integrity.py --responses th_text=responses_th_text.jsonl \
        --out scored_th_integrity.jsonl --review review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from integrity_checks import (
    REASON_EMPTY,
    REASON_LENGTH,
    REASON_SCRIPT,
    ScriptRule,
    is_excluded,
    load_records,
    loop_failure,
    pass_rate,
    script_context,
    script_failure,
    strip_code_and_urls,
)

SCORING_METHOD = "thai_char_ratio_script_degeneracy"

REASON_THAI_RATIO = "thai_ratio"

LINE_RULES = ("any_line", "share_of_lines")
DEFAULT_LINE_RULE = "any_line"
DEFAULT_SHARE_THRESHOLD = 0.25

DEFAULT_MIN_THAI_RATIO = 0.5
DEFAULT_MIN_LINE_LETTERS = 5
DEFAULT_FOREIGN_LETTER_SHARE = 0.01

THAI_BLOCK = (0x0E00, 0x0E7F)

MIN_SCRIPT_RUN = 3
RUN_SCRIPT_PREFIXES = ("CJK", "HANGUL", "HIRAGANA", "KATAKANA", "CYRILLIC", "ARABIC")
EXEMPT_SCRIPT_PREFIXES = ("THAI", "LATIN", "GREEK", "MATHEMATICAL", "DOUBLE-STRUCK")

NGRAM_WORDS = 10
NGRAM_REPEATS_TO_FAIL = 4
TAIL_CHARS = 1000
TAIL_MIN_BYTES = 600
TAIL_MAX_ZLIB_RATIO = 0.2

EXCERPT_CHARS = 200
EXCERPT_LINES = 2

Tokenize = Callable[[str], list[str]]


def tokenize_words(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def script_rule(foreign_letter_share: float) -> ScriptRule:
    return ScriptRule(EXEMPT_SCRIPT_PREFIXES, RUN_SCRIPT_PREFIXES, MIN_SCRIPT_RUN, foreign_letter_share)


def thai_letter_counts(line: str) -> tuple[int, int]:
    """(Thai letters, all letters). Both sides use str.isalpha, so the ratio never exceeds 1."""
    thai = letters = 0
    for ch in line:
        if ch.isalpha():
            letters += 1
            if THAI_BLOCK[0] <= ord(ch) <= THAI_BLOCK[1]:
                thai += 1
    return thai, letters


def checked_lines(text: str, min_line_letters: int) -> list[tuple[str, float]]:
    """(original line, Thai letter ratio) for every line with at least min_line_letters letters."""
    rows = []
    for line in text.splitlines():
        thai, letters = thai_letter_counts(line)
        if letters >= min_line_letters:
            rows.append((line.strip(), thai / letters))
    return rows


def evaluate(
    record: dict,
    share_threshold: float,
    min_thai_ratio: float,
    min_line_letters: int,
    foreign_letter_share: float,
    tokenize: Tokenize,
) -> dict:
    response = record["response"]
    if not response.strip():
        return {"common": [REASON_EMPTY], "lines_checked": 0, "non_th": [], "fails_any": False, "fails_share": False}

    common = []
    if record.get("finish_reason") == "length":
        common.append(REASON_LENGTH)
    text = strip_code_and_urls(response)
    if script_failure(text, script_rule(foreign_letter_share)):
        common.append(REASON_SCRIPT)
    loop = loop_failure(text, tokenize, NGRAM_WORDS, NGRAM_REPEATS_TO_FAIL, TAIL_CHARS, TAIL_MIN_BYTES, TAIL_MAX_ZLIB_RATIO)
    if loop:
        common.append(loop)

    lines = checked_lines(text, min_line_letters)
    non_th = [(ratio, line) for line, ratio in lines if ratio < min_thai_ratio]
    return {
        "common": common,
        "lines_checked": len(lines),
        "non_th": non_th,
        "fails_any": bool(non_th),
        "fails_share": bool(lines) and len(non_th) / len(lines) > share_threshold,
    }


def excerpt(record: dict, outcome: dict, first_reason: str, foreign_letter_share: float) -> str:
    response = record["response"]
    if first_reason == REASON_EMPTY:
        return ""
    if first_reason == REASON_SCRIPT:
        return script_context(strip_code_and_urls(response), EXEMPT_SCRIPT_PREFIXES, EXCERPT_CHARS)
    if first_reason == REASON_THAI_RATIO:
        return " | ".join(f"[{ratio:.2f}] {line[:150]}" for ratio, line in outcome["non_th"][:EXCERPT_LINES])
    return response[-EXCERPT_CHARS:]


def score_records(
    records: list[dict],
    line_rule: str = DEFAULT_LINE_RULE,
    share_threshold: float = DEFAULT_SHARE_THRESHOLD,
    min_thai_ratio: float = DEFAULT_MIN_THAI_RATIO,
    min_line_letters: int = DEFAULT_MIN_LINE_LETTERS,
    foreign_letter_share: float = DEFAULT_FOREIGN_LETTER_SHARE,
    tokenize: Tokenize | None = None,
) -> list[dict]:
    if line_rule not in LINE_RULES:
        raise ValueError(f"line_rule must be one of {LINE_RULES}, got {line_rule!r}")
    for record in records:
        if "response" not in record:
            raise ValueError(f"Record {record.get('id')!r} is missing a 'response' field")
    tokenize = tokenize or tokenize_words

    for record in records:
        record["scoring_method"] = SCORING_METHOD
        if is_excluded(record):
            record.update(
                excluded=True,
                exclusion_reason="generate_failed",
                label=None,
                failure_reason=None,
                reasons_any_line=[],
                reasons_share_of_lines=[],
                lines_checked=0,
                lines_non_th=0,
                review_excerpt="",
            )
            continue
        outcome = evaluate(record, share_threshold, min_thai_ratio, min_line_letters, foreign_letter_share, tokenize)
        reasons_any = outcome["common"] + ([REASON_THAI_RATIO] if outcome["fails_any"] else [])
        reasons_share = outcome["common"] + ([REASON_THAI_RATIO] if outcome["fails_share"] else [])
        selected = reasons_any if line_rule == "any_line" else reasons_share
        first = (reasons_any or reasons_share or [None])[0]
        record.update(
            excluded=False,
            exclusion_reason=None,
            label=int(not selected),
            failure_reason=selected[0] if selected else None,
            reasons_any_line=reasons_any,
            reasons_share_of_lines=reasons_share,
            lines_checked=outcome["lines_checked"],
            lines_non_th=len(outcome["non_th"]),
            review_excerpt=excerpt(record, outcome, first, foreign_letter_share) if first else "",
        )
    return records


def ratio_pass_rate(records: list[dict], key: str) -> float | None:
    """Share of records with no thai_ratio failure; the other checks are ignored."""
    if not records:
        return None
    return sum(1 for r in records if REASON_THAI_RATIO not in r[key]) / len(records)


def pythainlp_version() -> str:
    try:
        import pythainlp

        return pythainlp.__version__
    except ImportError:
        return "not installed"


def aggregate(records: list[dict]) -> dict:
    scorable = [r for r in records if not r["excluded"]]
    n_failed = len(records) - len(scorable)
    reasons = Counter(r["failure_reason"] for r in scorable if r["failure_reason"])
    sources = sorted({r.get("source", "") for r in scorable})
    per_source = {
        s: {"n_scored": len(group), "pass_rate": pass_rate(group)}
        for s in sources
        for group in [[r for r in scorable if r.get("source", "") == s]]
    }
    n_without_finish_reason = sum(1 for r in scorable if r.get("finish_reason") is None)
    n_no_checked_line = sum(1 for r in scorable if r["lines_checked"] == 0 and r["failure_reason"] != REASON_EMPTY)
    checked = [r for r in scorable if r["lines_checked"] > 0]

    note_parts = [
        "Thai letter ratio per line + script check + degeneracy checks",
        f"word tokenizer pythainlp {pythainlp_version()} newmm",
        "per source: " + ", ".join(f"{s or '(none)'} {v['n_scored']} scored" for s, v in per_source.items()),
    ]
    if n_failed:
        note_parts.append(f"{n_failed}/{len(records)} replies excluded (generation failed)")
    if n_without_finish_reason:
        note_parts.append(f"{n_without_finish_reason}/{len(scorable)} replies had no finish_reason, so the length check did not apply")

    return {
        "score": pass_rate(scorable),
        "n_scored": len(scorable),
        "n_generation_failed": n_failed,
        "n_without_finish_reason": n_without_finish_reason,
        "n_lines_checked": sum(r["lines_checked"] for r in scorable),
        "n_replies_with_no_checked_line": n_no_checked_line,
        "failures_by_reason": dict(reasons),
        "per_source": per_source,
        "pass_rate_by_line_rule": {
            "any_line": pass_rate(scorable, "reasons_any_line"),
            "share_of_lines": pass_rate(scorable, "reasons_share_of_lines"),
        },
        "n_replies_checked": len(checked),
        "ratio_pass_rate_checked": {
            "any_line": ratio_pass_rate(checked, "reasons_any_line"),
            "share_of_lines": ratio_pass_rate(checked, "reasons_share_of_lines"),
        },
        "mean_line_accuracy_checked": (
            sum(1 - r["lines_non_th"] / r["lines_checked"] for r in checked) / len(checked) if checked else None
        ),
        "notes": "; ".join(note_parts),
    }


REVIEW_COLUMNS = (
    "id",
    "source",
    "fails_any_line",
    "fails_share_of_lines",
    "reasons_any_line",
    "reasons_share_of_lines",
    "lines_checked",
    "lines_non_th",
    "excerpt",
    "verdict",
)


def review_rows(records: list[dict]) -> list[dict]:
    """One row per reply that fails under either line rule, with an empty verdict column to fill in."""
    rows = []
    for r in records:
        if r["excluded"] or not (r["reasons_any_line"] or r["reasons_share_of_lines"]):
            continue
        rows.append(
            {
                "id": r.get("id"),
                "source": r.get("source", ""),
                "fails_any_line": bool(r["reasons_any_line"]),
                "fails_share_of_lines": bool(r["reasons_share_of_lines"]),
                "reasons_any_line": "|".join(r["reasons_any_line"]),
                "reasons_share_of_lines": "|".join(r["reasons_share_of_lines"]),
                "lines_checked": r["lines_checked"],
                "lines_non_th": r["lines_non_th"],
                "excerpt": r["review_excerpt"],
                "verdict": "",
            }
        )
    return rows


def write_review(records: list[dict], path: Path) -> int:
    rows = review_rows(records)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None, tokenize: Tokenize | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", nargs="+", required=True, help="label=path (or path) of JSONL files with a `response` field")
    parser.add_argument("--out", default=None, help="Optional path to write per-record scores JSONL to")
    parser.add_argument("--review", default=None, help="Optional path to write the reviewable CSV of failing replies to")
    parser.add_argument("--line-rule", choices=LINE_RULES, default=DEFAULT_LINE_RULE, help="Rule that sets each record's label")
    parser.add_argument("--share-threshold", type=float, default=DEFAULT_SHARE_THRESHOLD, help="share_of_lines: fail above this share")
    parser.add_argument("--min-thai-ratio", type=float, default=DEFAULT_MIN_THAI_RATIO, help="A line fails below this Thai letter ratio")
    parser.add_argument("--min-line-letters", type=int, default=DEFAULT_MIN_LINE_LETTERS, help="Lines with fewer letters are not checked")
    parser.add_argument("--foreign-letter-share", type=float, default=DEFAULT_FOREIGN_LETTER_SHARE, help="script_mix: fail above this share of foreign letters")
    args = parser.parse_args(argv)

    records = load_records(args.responses)
    scored = score_records(
        records,
        line_rule=args.line_rule,
        share_threshold=args.share_threshold,
        min_thai_ratio=args.min_thai_ratio,
        min_line_letters=args.min_line_letters,
        foreign_letter_share=args.foreign_letter_share,
        tokenize=tokenize,
    )
    result = aggregate(scored)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for record in scored:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(scored)} scored records -> {args.out}")
    if args.review:
        n_rows = write_review(scored, Path(args.review))
        print(f"Wrote {n_rows} rows for review -> {args.review}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
