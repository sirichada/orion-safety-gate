#!/usr/bin/env python3
"""
Scores whether English replies are English and not broken. Reads one or more
JSONL files of prompt-set records that already have a `response` field (e.g.
model_wrapper.py's --out files for over_refusal and en_text) and writes each
record's label plus a pooled aggregate with a per-source breakdown.

A reply passes when none of these fire:
- empty_answer: the content is empty or whitespace. This is a failed sample,
  not an exclusion. Only a true generation failure (generation_failed without
  no_answer) is excluded.
- length_cap: finish_reason is "length" and the content is not empty.
- script_mix: more than 1% of the letters are outside the Latin, Greek and
  mathematical-alphabet scripts, or three letters in a row are CJK, Hangul, Kana, Thai, Cyrillic or
  Arabic. Applies at any length.
- loop_ngram / loop_tail: a 10-word sequence occurs four or more times, or the
  last 1000 characters (at least 600 bytes) compress to under 0.2 of their size.
- language_line: line-level language identification with fastText, following
  the Language Confusion Benchmark (Cohere-Labs-Community/language-confusion,
  compute_metrics.py, commit 0cfdbb4a64502648938c456b2fa9689968f22298). Code
  blocks and URLs are removed first. Each line is then normalized as the
  benchmark does: ASCII punctuation is deleted, "—" becomes a space and the
  Arabic comma is deleted. Words are counted and the language identified on the
  normalized line (so a markdown table row or a bullet has fewer words), while
  excerpts keep the original line. Lines of four words or fewer are not
  identified, so a short reply skips this check; every other line is identified
  by its top-1 label, and a label with probability of 0.3 or less counts as
  "unknown", which is not English.

Two line rules are evaluated for every reply, and the aggregate reports both:
- any_line: one line that is not English fails the reply (the benchmark's
  line-level pass rate, LPR).
- share_of_lines: the reply fails when the share of non-English lines exceeds
  the share threshold (the benchmark's per-reply line accuracy, `acc`, as a
  pass/fail cut).
The benchmark leaves replies with no identified line out of its line-level
denominator. The pooled score here counts them as passing the language check
(they can still fail the other checks); the aggregate also reports the
language-only rates over replies with at least one identified line, which are
the comparable ones.

The benchmark scored only the first 100 tokens of a reply; this scorer reads the
whole reply, so any_line gives a long reply more chances to fail.

Importing this module does not import fastText; the model loads on first use.

Usage:
    python3 score_en_integrity.py --responses over_refusal=responses_over_refusal.jsonl \
        en_text=responses_en_text.jsonl --out scored_en_integrity.jsonl --review review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import string
from collections import Counter
from pathlib import Path
from typing import Callable

from integrity_checks import (
    REASON_EMPTY,
    REASON_LENGTH,
    REASON_NGRAM,
    REASON_SCRIPT,
    REASON_TAIL,
    ScriptRule,
    is_excluded,
    load_records,
    parse_responses_arg,
    pass_rate,
    loop_failure as _loop_failure,
    script_context as _script_context,
    script_failure as _script_failure,
    strip_code_and_urls,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LID_MODEL_PATH = REPO_ROOT / "data" / "lid" / "lid.176.bin"

TARGET_LANGUAGE = "en"
UNKNOWN_LANGUAGE = "unknown"
# A top-1 label at or below this probability is not trusted.
LID_MIN_CONFIDENCE = 0.3
SCORING_METHOD = "fasttext_line_script_degeneracy"

LINE_RULES = ("any_line", "share_of_lines")
DEFAULT_LINE_RULE = "any_line"
DEFAULT_SHARE_THRESHOLD = 0.25

# Lines of this many words or fewer are not identified.
MAX_SKIPPED_LINE_WORDS = 4

MAX_NON_LATIN_SHARE = 0.01
MIN_SCRIPT_RUN = 3
RUN_SCRIPT_PREFIXES = ("CJK", "HANGUL", "HIRAGANA", "KATAKANA", "THAI", "CYRILLIC", "ARABIC")
EXEMPT_SCRIPT_PREFIXES = ("LATIN", "GREEK", "MATHEMATICAL", "DOUBLE-STRUCK")

NGRAM_WORDS = 10
NGRAM_REPEATS_TO_FAIL = 4
TAIL_CHARS = 1000
TAIL_MIN_BYTES = 600
TAIL_MAX_ZLIB_RATIO = 0.2

REASON_LANGUAGE = "language_line"

EXCERPT_CHARS = 200
EXCERPT_LINES = 2

SCRIPT_RULE = ScriptRule(EXEMPT_SCRIPT_PREFIXES, RUN_SCRIPT_PREFIXES, MIN_SCRIPT_RUN, MAX_NON_LATIN_SHARE)

Identify = Callable[[str], str]

_MODELS: dict[str, object] = {}


def load_lid_model(path: Path = LID_MODEL_PATH):
    """Loads the fastText model on first use; later calls return the same object."""
    key = str(path)
    if key not in _MODELS:
        if not Path(path).exists():
            raise FileNotFoundError(f"fastText language-ID model not found at {path}; run scripts/fetch_data.sh")
        import fasttext

        _MODELS[key] = fasttext.load_model(key)
    return _MODELS[key]


def predict_language(model, line: str) -> str:
    """Top-1 language code for one line (no newline inside), or "unknown" when the model is not confident."""
    try:
        labels, probs = model.predict(line, k=1)
    except ValueError as e:
        # fastText 0.9.3's wrapper builds its probability array with a call NumPy 2 rejects.
        if "copy" not in str(e):
            raise
        predictions = model.f.predict(line + "\n", 1, 0.0, "strict")
        labels = [label for _, label in predictions]
        probs = [prob for prob, _ in predictions]
    if not labels or not probs[0] > LID_MIN_CONFIDENCE:
        return UNKNOWN_LANGUAGE
    return labels[0].removeprefix("__label__")


def make_identifier(path: Path = LID_MODEL_PATH) -> Identify:
    return lambda line: predict_language(load_lid_model(path), line)


def script_failure(text: str) -> bool:
    return _script_failure(text, SCRIPT_RULE)


def script_context(text: str) -> str:
    return _script_context(text, EXEMPT_SCRIPT_PREFIXES, EXCERPT_CHARS)


def loop_failure(text: str) -> str | None:
    return _loop_failure(text, str.split, NGRAM_WORDS, NGRAM_REPEATS_TO_FAIL, TAIL_CHARS, TAIL_MIN_BYTES, TAIL_MAX_ZLIB_RATIO)


_PUNCTUATION = str.maketrans("", "", string.punctuation + "،")


def normalize_line(line: str) -> str:
    return line.replace("—", " ").translate(_PUNCTUATION).strip()


def identified_lines(text: str) -> list[tuple[str, str]]:
    """(original, normalized) for every line that has more than four words once normalized."""
    pairs = ((line.strip(), normalize_line(line)) for line in text.splitlines())
    return [(original, normalized) for original, normalized in pairs if len(normalized.split()) > MAX_SKIPPED_LINE_WORDS]


def evaluate(record: dict, identify: Identify, share_threshold: float) -> dict:
    response = record["response"]
    if not response.strip():
        return {"common": [REASON_EMPTY], "lines_checked": 0, "non_en": [], "fails_any": False, "fails_share": False}

    common = []
    if record.get("finish_reason") == "length":
        common.append(REASON_LENGTH)
    text = strip_code_and_urls(response)
    if script_failure(text):
        common.append(REASON_SCRIPT)
    loop = loop_failure(text)
    if loop:
        common.append(loop)

    lines = identified_lines(text)
    non_en = []
    for original, normalized in lines:
        label = identify(normalized)
        if label != TARGET_LANGUAGE:
            non_en.append((label, original))
    return {
        "common": common,
        "lines_checked": len(lines),
        "non_en": non_en,
        "fails_any": bool(non_en),
        "fails_share": bool(lines) and len(non_en) / len(lines) > share_threshold,
    }


def excerpt(record: dict, outcome: dict, first_reason: str) -> str:
    response = record["response"]
    if first_reason == REASON_EMPTY:
        return ""
    if first_reason == REASON_SCRIPT:
        return script_context(strip_code_and_urls(response))
    if first_reason == REASON_LANGUAGE:
        return " | ".join(f"[{label}] {line[:150]}" for label, line in outcome["non_en"][:EXCERPT_LINES])
    return response[-EXCERPT_CHARS:]


def score_records(
    records: list[dict],
    line_rule: str = DEFAULT_LINE_RULE,
    share_threshold: float = DEFAULT_SHARE_THRESHOLD,
    identify: Identify | None = None,
) -> list[dict]:
    if line_rule not in LINE_RULES:
        raise ValueError(f"line_rule must be one of {LINE_RULES}, got {line_rule!r}")
    for record in records:
        if "response" not in record:
            raise ValueError(f"Record {record.get('id')!r} is missing a 'response' field")
    identify = identify or make_identifier()

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
                lines_non_en=0,
                review_excerpt="",
            )
            continue
        outcome = evaluate(record, identify, share_threshold)
        reasons_any = outcome["common"] + ([REASON_LANGUAGE] if outcome["fails_any"] else [])
        reasons_share = outcome["common"] + ([REASON_LANGUAGE] if outcome["fails_share"] else [])
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
            lines_non_en=len(outcome["non_en"]),
            review_excerpt=excerpt(record, outcome, first) if first else "",
        )
    return records


def language_pass_rate(records: list[dict], key: str) -> float | None:
    """Share of records with no language_line failure; the other checks are ignored."""
    if not records:
        return None
    return sum(1 for r in records if REASON_LANGUAGE not in r[key]) / len(records)


def aggregate(records: list[dict]) -> dict:
    scorable = [r for r in records if not r["excluded"]]
    n_failed = len(records) - len(scorable)
    reasons = Counter(r["failure_reason"] for r in scorable if r["failure_reason"])
    sources = sorted({r.get("source", "") for r in scorable})
    per_source = {
        s: {
            "n_scored": len(group),
            "pass_rate": pass_rate(group),
        }
        for s in sources
        for group in [[r for r in scorable if r.get("source", "") == s]]
    }
    n_without_finish_reason = sum(1 for r in scorable if r.get("finish_reason") is None)
    n_no_long_line = sum(1 for r in scorable if r["lines_checked"] == 0 and r["failure_reason"] != REASON_EMPTY)
    identified = [r for r in scorable if r["lines_checked"] > 0]

    note_parts = [
        "fastText line-level language ID (lines over 4 words) + script check + degeneracy checks",
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
        "n_lines_identified": sum(r["lines_checked"] for r in scorable),
        "n_replies_with_no_long_line": n_no_long_line,
        "failures_by_reason": dict(reasons),
        "per_source": per_source,
        "pass_rate_by_line_rule": {
            "any_line": pass_rate(scorable, "reasons_any_line"),
            "share_of_lines": pass_rate(scorable, "reasons_share_of_lines"),
        },
        "n_replies_identified": len(identified),
        "language_pass_rate_identified": {
            "any_line": language_pass_rate(identified, "reasons_any_line"),
            "share_of_lines": language_pass_rate(identified, "reasons_share_of_lines"),
        },
        "mean_line_accuracy_identified": (
            sum(1 - r["lines_non_en"] / r["lines_checked"] for r in identified) / len(identified) if identified else None
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
    "lines_non_en",
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
                "lines_non_en": r["lines_non_en"],
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


def main(argv: list[str] | None = None, identify: Identify | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", nargs="+", required=True, help="label=path (or path) of JSONL files with a `response` field")
    parser.add_argument("--out", default=None, help="Optional path to write per-record scores JSONL to")
    parser.add_argument("--review", default=None, help="Optional path to write the reviewable CSV of failing replies to")
    parser.add_argument("--line-rule", choices=LINE_RULES, default=DEFAULT_LINE_RULE, help="Rule that sets each record's label")
    parser.add_argument("--share-threshold", type=float, default=DEFAULT_SHARE_THRESHOLD, help="share_of_lines: fail above this share")
    parser.add_argument("--lid-model", default=str(LID_MODEL_PATH), help="Path to the fastText language-ID model")
    args = parser.parse_args(argv)

    records = load_records(args.responses)
    scored = score_records(
        records,
        line_rule=args.line_rule,
        share_threshold=args.share_threshold,
        identify=identify or make_identifier(Path(args.lid_model)),
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
