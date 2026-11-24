"""Builds fixture verdicts for the report and review tests, offline and with no model."""

from __future__ import annotations

import json
from pathlib import Path

from capabilities import parse_capabilities
from case_summary import attach_summaries
from cases import REGISTRY, select_cases
from verdict import apply_case_review, build_case_result, build_verdict, result_options

THRESHOLDS = json.loads((Path(__file__).resolve().parent.parent / "config" / "thresholds.json").read_text(encoding="utf-8"))
CASES = {case.case_id: case for case in REGISTRY}
PASSING = {"en_text": 0.150, "th_text": 0.134, "over_refusal": 0.016, "en_integrity": 0.998, "th_integrity": 0.960, "th_knowledge": 0.673}
CATEGORIES = {
    "en_text": ("misinformation", "cybercrime"),
    "th_text": ("Misinformation Harms", "Malicious Uses"),
    "over_refusal": ("homonyms", "safe_targets"),
    "en_integrity": ("misinformation", "homonyms"),
    "th_integrity": ("Information Hazards", "Malicious Uses"),
    "th_knowledge": ("onet", "tgat"),
}
N = 200
MODEL = "Fixture Model 1"
RUN_ARGS = {
    "model": MODEL,
    "backend": "api",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    "api": {"spent_thb": 12.5},
    "prompt_sample": None,
    "provenance": {"suite": {"commit": "0123456789abcdef0123456789abcdef01234567"}},
}


def records_for(case_id: str, score: float, n: int = N, replies: dict | None = None) -> list[dict]:
    """n scored records whose flagged share matches the score (flagged = harmful, refused, failing or wrong)."""
    k_pos = round(score * n)
    records = []
    for i in range(n):
        positive = i < k_pos
        label = int(positive)
        records.append(
            {
                "id": f"{case_id}_{i:03d}",
                "native_category": CATEGORIES[case_id][i % 2],
                "prompt": f"prompt {i}" if not case_id.startswith("th") else f"คำถามที่ {i}",
                "response": (replies or {}).get(i, f"reply {i} " + "text " * 80),
                "label": label,
                "excluded": False,
                "model_input": "Question: q\nA. one\nB. two\nC. three\nD. four\nE. five\nAnswer:",
                "answer": "B",
                "predicted": "B" if label else "D",
            }
        )
    return records


def write_scored(directory: Path, scores: dict, replies: dict | None = None) -> dict:
    """Scored files for each case in `scores`; returns case_id -> path."""
    paths = {}
    for case_id, score in scores.items():
        if score is None:
            continue
        n = N
        recs = records_for(case_id, score, n, (replies or {}).get(case_id))
        if case_id in ("en_integrity", "th_integrity", "th_knowledge"):
            # Integrity and knowledge: label 1 is the good outcome, so the score is the share labelled 1.
            for i, rec in enumerate(recs):
                rec["label"] = int(i < round(score * n))
                rec["predicted"] = "B" if rec["label"] else "D"
        path = directory / f"scored_{case_id}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        paths[case_id] = path
    return paths


def make_verdict(tmp_path: Path, scores: dict | None = None, *, only=None, run_args=None, excluded: dict | None = None, empty: dict | None = None, replies=None, deviations=None, with_samples=True) -> dict:
    scores = {**PASSING, **(scores or {})}
    directory = Path(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    selection = select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["en", "th"]}), only=only)
    paths = write_scored(directory, {c.case_id: scores[c.case_id] for c in selection.cases}, replies)
    results = []
    for case in selection.cases:
        entry = THRESHOLDS["cases"][case.case_id]
        score = scores[case.case_id]
        n_excluded = (excluded or {}).get(case.case_id, 0)
        n_prompts = N + n_excluded
        derived = case.derived
        result = build_case_result(
            case,
            score=None if score is None else score,
            n_prompts=n_prompts,
            n_scored=0 if score is None else N,
            threshold=entry["threshold"],
            min_scored=entry["min_scored"],
            scoring_method="fixture",
            notes="",
            n_excluded=None if derived else n_excluded,
            n_no_answer=None if derived else (empty or {}).get(case.case_id, 0),
            max_excluded_fraction=entry.get("max_excluded_fraction"),
            max_new_tokens=None if derived else 4096,
            cost_thb=None if derived else 1.25,
            **result_options(entry),
        )
        results.append(result)
    verdict = build_verdict(
        MODEL,
        selection,
        results,
        thresholds_provisional=True,
        run_args={**RUN_ARGS, **(run_args or {})},
        run_timestamp="2026-11-20T09:00:00+00:00",
        deviations=deviations,
        scored_files=paths if with_samples else None,
    )
    return verdict


def reviewed(verdict: dict, decision: str = "accept", harmful: int = 3) -> dict:
    case = next(c for c in verdict["case_results"] if c["status"] == "needs_review")
    flagged = [s["id"] for s in case["sample_generations"] if s["outcome"] == "flagged"]
    review = {
        "reviewer": "Fixture Reviewer",
        "date": "2026-11-25",
        "n_reviewed": len(flagged),
        "n_judged_harmful": harmful,
        "decision": decision,
        "note": "most flags are refusals",
        "judgements": [{"id": fid, "harmful": i < harmful} for i, fid in enumerate(flagged)],
    }
    return apply_case_review(verdict, case["case_id"], review)
