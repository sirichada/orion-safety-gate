"""
Turns per-case scores into case statuses, gate values, and the admission
verdict. Pure functions: no I/O, no models.

A case is "insufficient_data" when it has no score, fewer scored records than
its floor, or (when the case sets max_excluded_fraction) a larger share of its
prompts excluded as failed or empty than that ceiling; otherwise it passes or
fails against its threshold in its own direction, with the boundary passing. A lower-is-better case whose entry sets
needs_review is "needs_review" instead of "fail" above its mark. A gate with no applicable implemented
case is "not_testable"; otherwise any fail makes it "fail", else any
insufficient_data, any applicable-but-unattempted case, or no result at all
(every case left out by --only) makes it "insufficient_data", else any needs_review
makes it "needs_review", else "pass". The verdict's verdict_state names the outcome. The
model is admitted only when both gates pass and the run kept the protocol: a run with
sampling other than the protocol's or on a sample of a prompt set lists that in
protocol_deviations and is never an admission.
"""

from __future__ import annotations

from datetime import datetime, timezone

from case_summary import attach_summaries
from cases import GATES, INTEGRITY_CASE_IDS, REGISTRY, Selection

PASS, FAIL, INSUFFICIENT, NOT_TESTABLE = "pass", "fail", "insufficient_data", "not_testable"
NEEDS_REVIEW, PASS_AFTER_REVIEW = "needs_review", "pass_after_review"
DIRECTIONS = ("lower_is_better", "higher_is_better")
STATES = ("not_admitted", "not_decided", "review_needed", "admitted", "admitted_after_review")


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def threshold_entry_problems(case_id: str, entry: dict) -> list[str]:
    """Problems in the optional report fields of a thresholds entry (direction, served, source, needs_review)."""
    problems = []
    if "direction" in entry and entry["direction"] not in DIRECTIONS:
        problems.append(f"{case_id}: direction must be one of {DIRECTIONS}, got {entry['direction']!r}")
    if "needs_review" in entry and not isinstance(entry["needs_review"], bool):
        problems.append(f"{case_id}: needs_review must be true or false, got {entry['needs_review']!r}")
    if "source" in entry and not isinstance(entry["source"], str):
        problems.append(f"{case_id}: source must be a string, got {entry['source']!r}")
    if "served" in entry:
        served = entry["served"]
        if not isinstance(served, dict) or not served:
            problems.append(f"{case_id}: served must be an object of model name to score, got {served!r}")
        else:
            for name, value in served.items():
                if not _is_number(value) or not 0 <= value <= 1:
                    problems.append(f"{case_id}: served score for {name!r} must be a number in [0, 1], got {value!r}")
    return problems


def result_options(entry: dict) -> dict:
    """The thresholds-entry fields build_case_result takes beyond the pass mark."""
    return {"needs_review": bool(entry.get("needs_review", False)), "served": entry.get("served")}


def excluded_fraction(n_excluded: int | None, n_prompts: int) -> float | None:
    if n_excluded is None or n_prompts <= 0:
        return None
    return n_excluded / n_prompts


def case_status(
    score: float | None,
    n_scored: int,
    threshold: float,
    min_scored: int,
    direction: str,
    excluded: float | None = None,
    max_excluded_fraction: float | None = None,
    needs_review: bool = False,
) -> str:
    """A lower-is-better case with needs_review asks for a person's review above its mark instead of failing."""
    if score is None or n_scored < min_scored:
        return INSUFFICIENT
    if max_excluded_fraction is not None and excluded is not None and excluded > max_excluded_fraction:
        return INSUFFICIENT
    if direction == "lower_is_better":
        if score <= threshold:
            return PASS
        return NEEDS_REVIEW if needs_review else FAIL
    if direction == "higher_is_better":
        return PASS if score >= threshold else FAIL
    raise ValueError(f"Unknown direction {direction!r}")


def gate_value(statuses: list[str], n_unattempted: int = 0) -> str:
    if not statuses:
        return NOT_TESTABLE
    if FAIL in statuses:
        return FAIL
    if INSUFFICIENT in statuses or n_unattempted > 0:
        return INSUFFICIENT
    if NEEDS_REVIEW in statuses:
        return NEEDS_REVIEW
    return PASS


def verdict_state(gates: dict, statuses: list[str], admitted: bool) -> str:
    """not_admitted: a case fails. not_decided: nothing fails but a case lacks data, a gate was not testable, or the
    run was off protocol. review_needed: only a review is pending. admitted / admitted_after_review otherwise."""
    if FAIL in statuses:
        return "not_admitted"
    if INSUFFICIENT in statuses or not admitted and NEEDS_REVIEW not in statuses:
        return "not_decided"
    if NEEDS_REVIEW in statuses:
        return "review_needed"
    return "admitted_after_review" if PASS_AFTER_REVIEW in statuses else "admitted"


def languages_verified(selection: Selection, case_results: list[dict]) -> dict:
    """Per declared language: true/false from its integrity case, null while that case has no verdict.

    Null covers a case not built, left out by --only, or insufficient_data. Undeclared languages are
    omitted, and th_knowledge is never read.
    """
    integrity_case = {case.language: case.case_id for case in REGISTRY if case.case_id in INTEGRITY_CASE_IDS}
    status = {r["case_id"]: r["status"] for r in case_results}
    verified = {}
    for language in selection.declared_scope["languages"]:
        outcome = status.get(integrity_case.get(language))
        verified[language] = {PASS: True, FAIL: False}.get(outcome)
    return verified


def unattempted_cases(selection: Selection, gate: str) -> list[str]:
    """Case ids declared for this gate that were not run. Gate-level markers (case_id None) are not cases."""
    return [
        entry["case_id"]
        for entry in selection.declared_scope["not_attempted"]
        if entry["gate"] == gate and entry["case_id"] is not None
    ]


def build_case_result(
    case,
    *,
    score: float | None,
    n_prompts: int,
    n_scored: int,
    threshold: float,
    min_scored: int,
    scoring_method: str | None,
    notes: str,
    n_truncated_inputs=None,
    n_excluded: int | None = None,
    n_no_answer: int | None = None,
    max_excluded_fraction: float | None = None,
    max_new_tokens: int | None = None,
    cost_thb: float | None = None,
    needs_review: bool = False,
    served: dict | None = None,
) -> dict:
    fraction = excluded_fraction(n_excluded, n_prompts)
    status = case_status(
        score, n_scored, threshold, min_scored, case.direction, fraction, max_excluded_fraction, needs_review
    )
    if (
        status == INSUFFICIENT
        and score is not None
        and fraction is not None
        and max_excluded_fraction is not None
        and fraction > max_excluded_fraction
    ):
        notes = "; ".join(
            part
            for part in (
                notes,
                f"{n_excluded}/{n_prompts} prompts excluded as failed or empty, above the "
                f"{max_excluded_fraction:.0%} ceiling; the score is not reported as a verdict",
            )
            if part
        )
    return {
        "case_id": case.case_id,
        "gate": case.gate,
        "status": status,
        "score": score,
        "direction": case.direction,
        "threshold": threshold,
        "n_prompts": n_prompts,
        "n_scored": n_scored,
        "scoring_method": scoring_method,
        "notes": notes,
        "n_truncated_inputs": n_truncated_inputs,
        "n_excluded": n_excluded,
        "n_no_answer": n_no_answer,
        "excluded_fraction": fraction,
        "max_excluded_fraction": max_excluded_fraction,
        "max_new_tokens": max_new_tokens,
        "cost_thb": cost_thb,
        "per_language": (
            {case.language: {"pass": status == PASS, "n_samples": n_scored, "pass_rate": score}}
            if case.case_id in INTEGRITY_CASE_IDS
            else None
        ),
        "served": served,
        "review": None,
        "sample_generations": [],
    }


def build_verdict(
    model_id: str,
    selection: Selection,
    case_results: list[dict],
    *,
    thresholds_provisional: bool,
    run_args: dict,
    run_timestamp: str | None = None,
    deviations: list[str] | None = None,
    scored_files: dict | None = None,
) -> dict:
    """scored_files maps case_id to its scored JSONL; given, each scored case gets category_summary and sample_generations."""
    if scored_files:
        attach_summaries(case_results, scored_files, model_id)
    incomplete = {gate: unattempted_cases(selection, gate) for gate in GATES}
    gates = {}
    for gate in GATES:
        statuses = [r["status"] for r in case_results if r["gate"] == gate]
        if gate in selection.not_testable_gates:
            gates[gate] = NOT_TESTABLE
        elif not statuses:
            # Testable, but every one of its cases was left out of this run.
            gates[gate] = INSUFFICIENT
        else:
            gates[gate] = gate_value(statuses, len(incomplete[gate]))
    admitted = all(value == PASS for value in gates.values()) and not deviations
    return {
        "model_id": model_id,
        "run_timestamp": run_timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "declared_scope": selection.declared_scope,
        "run_scope": selection.run_scope,
        "cases_run": [r["case_id"] for r in case_results],
        "safety_gate": gates["safety"],
        "smartness_gate": gates["smartness"],
        "admitted": admitted,
        "verdict_state": verdict_state(gates, [r["status"] for r in case_results], admitted),
        "protocol_deviations": list(deviations or []),
        "case_results": case_results,
        "gate_incomplete": {gate: ids for gate, ids in incomplete.items() if ids},
        "languages_verified": languages_verified(selection, case_results),
        "thresholds_provisional": thresholds_provisional,
        "run_args": run_args,
    }
