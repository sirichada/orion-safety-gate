#!/usr/bin/env python3
"""
Applies a reviewer's decision on a needs_review case to a recorded verdict.

Writes verdict.reviewed-<stamp>.json next to the verdict it reads and never replaces or edits any existing file.
It re-runs no model and no classifier: the reviewed replies are the flagged ones sampled into the verdict.

review.json fields: case_id, reviewer, date (YYYY-MM-DD), n_reviewed, n_judged_harmful, decision (accept|reject),
note (optional) and judgements, a list of {id, harmful}. The file must name a case that needs review, judge exactly
the flagged replies sampled into the verdict, and have counts that agree with its judgements.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from orchestrator import SuiteError
from verdict import NEEDS_REVIEW, apply_case_review

VERDICT_NAME = "verdict.json"
DECISIONS = ("accept", "reject")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def review_problems(verdict: dict, review) -> list[str]:
    if not isinstance(review, dict):
        return ["review.json must hold one JSON object"]
    problems = []
    for key in ("case_id", "reviewer", "date", "n_reviewed", "n_judged_harmful", "decision", "judgements"):
        if key not in review:
            problems.append(f"missing field {key!r}")
    if problems:
        return problems
    if review["decision"] not in DECISIONS:
        problems.append(f"decision must be one of {DECISIONS}, got {review['decision']!r}")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        problems.append("reviewer must be a name")
    try:
        date.fromisoformat(review["date"])
    except (TypeError, ValueError):
        problems.append(f"date must be YYYY-MM-DD, got {review['date']!r}")
    if "note" in review and not isinstance(review["note"], str):
        problems.append("note must be text")
    judgements = review["judgements"]
    well_formed = isinstance(judgements, list) and all(isinstance(j, dict) and "id" in j and isinstance(j.get("harmful"), bool) for j in judgements)
    if not well_formed:
        problems.append("judgements must be a list of {id, harmful} with harmful true or false")
    if not _is_int(review["n_reviewed"]) or not _is_int(review["n_judged_harmful"]):
        problems.append("n_reviewed and n_judged_harmful must be whole numbers")
    elif review["n_judged_harmful"] > review["n_reviewed"] or review["n_judged_harmful"] < 0:
        problems.append(f"n_judged_harmful ({review['n_judged_harmful']}) must be between 0 and n_reviewed ({review['n_reviewed']})")
    if problems:
        return problems
    if review["n_reviewed"] != len(judgements):
        problems.append(f"n_reviewed is {review['n_reviewed']} but there are {len(judgements)} judgements")
    if review["n_judged_harmful"] != sum(1 for j in judgements if j["harmful"]):
        problems.append(f"n_judged_harmful is {review['n_judged_harmful']} but the judgements mark {sum(1 for j in judgements if j['harmful'])} harmful")
    ids = [j["id"] for j in judgements]
    if len(set(ids)) != len(ids):
        problems.append("a reply is judged more than once")

    case = next((c for c in verdict["case_results"] if c["case_id"] == review["case_id"]), None)
    if case is None:
        problems.append(f"the verdict has no case {review['case_id']!r}")
    elif case["status"] != NEEDS_REVIEW:
        problems.append(f"{review['case_id']} is {case['status']!r} in this verdict, not needs_review, so there is nothing to review")
    else:
        sampled = [s["id"] for s in case.get("sample_generations") or [] if s.get("outcome") == "flagged"]
        if not sampled:
            problems.append(f"{review['case_id']} has no sampled flagged replies in this verdict; re-score it first")
        elif set(ids) != set(sampled):
            problems.append(f"the judgements must cover exactly the {len(sampled)} flagged replies sampled in this verdict (missing {len(set(sampled) - set(ids))}, unknown {len(set(ids) - set(sampled))})")
    return problems


def apply_review(run_dir: Path, review_path: Path, *, verdict_name: str = VERDICT_NAME, stamp: str | None = None) -> tuple[dict, Path]:
    run_dir = Path(run_dir).resolve()
    verdict_path = run_dir / verdict_name
    if not verdict_path.is_file():
        raise SuiteError(f"{verdict_path} not found")
    try:
        verdict_bytes = verdict_path.read_bytes()
        verdict = json.loads(verdict_bytes.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise SuiteError(f"{verdict_path} cannot be read as a verdict: {e}") from e
    try:
        review_bytes = Path(review_path).read_bytes()
        review = json.loads(review_bytes.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise SuiteError(f"{review_path} cannot be read as JSON: {e}") from e
    problems = review_problems(verdict, review)
    if problems:
        raise SuiteError("Review refused:\n  " + "\n  ".join(problems))

    source_sha = hashlib.sha256(verdict_bytes).hexdigest()
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    for earlier in sorted(run_dir.glob("verdict.reviewed-*.json")):
        try:
            applied = json.loads(earlier.read_text(encoding="utf-8")).get("review_of") or {}
        except (OSError, ValueError):
            continue
        if applied.get("source_verdict_sha256") == source_sha and applied.get("case_id") == review["case_id"]:
            raise SuiteError(f"Review refused: {earlier.name} already applies a review of {review['case_id']} to this verdict")

    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = run_dir / f"verdict.reviewed-{stamp}.json"
    if out.exists():
        raise SuiteError(f"{out} already exists; wait a second and run again")
    record = {
        "reviewer": review["reviewer"].strip(),
        "date": review["date"],
        "n_reviewed": review["n_reviewed"],
        "n_judged_harmful": review["n_judged_harmful"],
        "decision": review["decision"],
        "note": review.get("note", ""),
        "judgements": [{"id": j["id"], "harmful": j["harmful"]} for j in review["judgements"]],
        "review_sha256": review_sha,
    }
    reviewed = apply_case_review(verdict, review["case_id"], record)
    reviewed["review_of"] = {
        "source_verdict": verdict_name,
        "source_verdict_sha256": source_sha,
        "case_id": review["case_id"],
        "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(reviewed, indent=2, ensure_ascii=False), encoding="utf-8")
    return reviewed, out
