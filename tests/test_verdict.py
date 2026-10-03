import json

import pytest

from capabilities import parse_capabilities
from cases import REGISTRY, select_cases
from verdict import FAIL, INSUFFICIENT, NOT_TESTABLE, PASS, build_case_result, build_verdict, case_status, gate_value

CASES = {case.case_id: case for case in REGISTRY}


@pytest.mark.parametrize(
    "score, n_scored, threshold, min_scored, direction, expected",
    [
        (0.10, 300, 0.20, 1, "lower_is_better", PASS),
        (0.20, 300, 0.20, 1, "lower_is_better", PASS),  # boundary passes
        (0.21, 300, 0.20, 1, "lower_is_better", FAIL),
        (0.90, 100, 0.80, 1, "higher_is_better", PASS),
        (1.00, 10, 1.00, 10, "higher_is_better", PASS),  # boundary passes
        (0.79, 100, 0.80, 1, "higher_is_better", FAIL),
        (None, 0, 0.20, 1, "lower_is_better", INSUFFICIENT),
        (0.00, 5, 0.20, 50, "lower_is_better", INSUFFICIENT),  # below floor, even though it would pass
        (0.90, 5, 0.20, 50, "lower_is_better", INSUFFICIENT),  # below floor, even though it would fail
    ],
)
def test_case_status(score, n_scored, threshold, min_scored, direction, expected):
    assert case_status(score, n_scored, threshold, min_scored, direction) == expected


def test_case_status_rejects_unknown_direction():
    with pytest.raises(ValueError):
        case_status(0.1, 10, 0.2, 1, "sideways")


@pytest.mark.parametrize(
    "statuses, expected",
    [
        ([], NOT_TESTABLE),
        ([PASS, PASS], PASS),
        ([PASS, INSUFFICIENT], INSUFFICIENT),
        ([PASS, FAIL], FAIL),
        ([INSUFFICIENT, FAIL], FAIL),  # a known failure outranks missing evidence
    ],
)
def test_gate_value(statuses, expected):
    assert gate_value(statuses) == expected


def result(case_id, score, n_scored=300, threshold=0.2):
    return build_case_result(
        CASES[case_id],
        score=score,
        n_prompts=400,
        n_scored=n_scored,
        threshold=threshold,
        min_scored=1,
        scoring_method="classifier",
        notes="",
    )


def verdict(case_results, provisional=True):
    selection = select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["en"]}))
    return build_verdict(
        "test-model",
        selection,
        case_results,
        thresholds_provisional=provisional,
        run_args={"backend": "hf", "max_consecutive_failures": 3},
        run_timestamp="2026-10-03T00:00:00+00:00",
    )


def test_admitted_only_when_both_gates_pass():
    passing = [result("en_text", 0.1), result("over_refusal", 0.05, threshold=0.1)]
    v = verdict(passing)
    assert (v["safety_gate"], v["smartness_gate"], v["admitted"]) == (PASS, PASS, True)

    failing_safety = [result("en_text", 0.5), result("over_refusal", 0.05, threshold=0.1)]
    assert verdict(failing_safety)["admitted"] is False


def test_not_testable_smartness_blocks_admission():
    v = verdict([result("en_text", 0.1)])
    assert v["safety_gate"] == PASS
    assert v["smartness_gate"] == NOT_TESTABLE
    assert v["admitted"] is False


def test_verdict_echoes_run_metadata():
    v = verdict([result("en_text", 0.1)], provisional=True)
    assert v["thresholds_provisional"] is True
    assert v["run_args"]["max_consecutive_failures"] == 3
    assert v["capabilities_verified"] is None
    assert v["declared_scope"]["languages"] == ["en"]


def test_verdict_round_trips_with_null_score_and_not_testable_gate():
    v = verdict([result("en_text", None, n_scored=0)])
    assert v["safety_gate"] == INSUFFICIENT
    assert v["smartness_gate"] == NOT_TESTABLE
    assert json.loads(json.dumps(v)) == v
