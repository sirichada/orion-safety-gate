"""
Per-category counts and seeded sample replies for a case, read from its scored file.

Stdlib-only. A record is "flagged" when its label marks the outcome the case counts: harmful (en_text,
th_text), refused (over_refusal), failing the integrity check (en_integrity, th_integrity) or answered wrongly
(th_knowledge). The sample draw is seeded from the model id and case id, so the same scored file always gives the
same replies.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from seeding import derive_seed

FLAGGED_LABEL = {
    "en_text": 1,
    "th_text": 1,
    "over_refusal": 1,
    "en_integrity": 0,
    "th_integrity": 0,
    "th_knowledge": 0,
}
SAMPLE_SIZE = 5
REVIEW_SAMPLE_SIZE = 30
HEAD_CHARS = 300
REVIEW_REPLY_CHARS = 4000
EXTRA_FIELDS = {
    "en_integrity": ("failure_reason", "source"),
    "th_integrity": ("failure_reason", "source"),
    "th_knowledge": ("answer", "predicted", "parse_rule"),
}


def category_summary(case_id: str, records: list[dict]) -> list[dict]:
    """One row per native_category, most flagged first: n_scored, n_flagged, rate."""
    flagged_label = FLAGGED_LABEL[case_id]
    counts: dict[str, list[int]] = {}
    for record in records:
        if record.get("excluded") or record.get("label") is None:
            continue
        row = counts.setdefault(str(record.get("native_category")), [0, 0])
        row[0] += 1
        row[1] += int(record["label"] == flagged_label)
    rows = [{"category": name, "n_scored": n, "n_flagged": k, "rate": k / n} for name, (n, k) in counts.items()]
    return sorted(rows, key=lambda row: (-row["rate"], row["category"]))


def _sample_row(case_id: str, record: dict, flagged: bool, full_reply: bool) -> dict:
    reply = (record.get("response") or "").strip()
    limit = REVIEW_REPLY_CHARS if full_reply else HEAD_CHARS
    row = {
        "id": record.get("id"),
        "category": record.get("native_category"),
        "outcome": "flagged" if flagged else "not_flagged",
        "prompt": (record.get("prompt") or "")[:HEAD_CHARS],
        "reply": reply[:limit],
        "reply_chars": len(reply),
    }
    for key in EXTRA_FIELDS.get(case_id, ()):
        if key in record:
            row[key] = record[key]
    return row


def sample_generations(case_id: str, records: list[dict], model_id: str, *, review: bool = False) -> list[dict]:
    """Random flagged and not-flagged replies. With review, 30 flagged replies are kept in full for a reviewer."""
    flagged_label = FLAGGED_LABEL[case_id]
    scorable = sorted((r for r in records if not r.get("excluded") and r.get("label") is not None), key=lambda r: str(r.get("id")))
    flagged = [r for r in scorable if r["label"] == flagged_label]
    rest = [r for r in scorable if r["label"] != flagged_label]
    rng = random.Random(derive_seed(f"{model_id}|{case_id}"))
    n_flagged = REVIEW_SAMPLE_SIZE if review else SAMPLE_SIZE
    picked_flagged = rng.sample(flagged, min(n_flagged, len(flagged)))
    picked_rest = rng.sample(rest, min(SAMPLE_SIZE, len(rest)))
    return [_sample_row(case_id, r, True, review) for r in picked_flagged] + [_sample_row(case_id, r, False, False) for r in picked_rest]


def attach_summaries(case_results: list[dict], scored_files: dict[str, Path], model_id: str) -> None:
    """Fill category_summary and sample_generations of each scored case from its scored file, when one exists."""
    for result in case_results:
        path = scored_files.get(result["case_id"])
        if result["score"] is None or path is None or not Path(path).is_file():
            continue
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        result["category_summary"] = category_summary(result["case_id"], records)
        result["sample_generations"] = sample_generations(
            result["case_id"], records, model_id, review=result["status"] == "needs_review"
        )
