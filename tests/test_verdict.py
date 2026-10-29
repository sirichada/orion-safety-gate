import json

import pytest

from capabilities import parse_capabilities
from cases import REGISTRY, Selection, select_cases
from verdict import (
    FAIL,
    INSUFFICIENT,
    NOT_TESTABLE,
    PASS,
    build_case_result,
    build_verdict,
    case_status,
    gate_value,
    unattempted_cases,
)

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


@pytest.mark.parametrize(
    "statuses, n_unattempted, expected",
    [
        ([PASS], 1, INSUFFICIENT),  # a declared check that never ran keeps the gate from passing
        ([PASS, PASS], 2, INSUFFICIENT),
        ([FAIL], 1, FAIL),  # a known failure is never masked
        ([INSUFFICIENT], 1, INSUFFICIENT),
        ([PASS], 0, PASS),
        ([], 0, NOT_TESTABLE),
        ([], 2, NOT_TESTABLE),  # nothing ran at all: reported as not testable, not as incomplete
    ],
)
def test_gate_value_with_unattempted_cases(statuses, n_unattempted, expected):
    assert gate_value(statuses, n_unattempted) == expected


def selection_with_not_attempted(entries):
    return Selection(
        cases=(),
        not_testable_gates=frozenset(),
        declared_scope={"modalities": ["text"], "languages": ["en"], "not_attempted": entries},
    )


def test_unattempted_cases_ignores_gate_level_markers_and_other_gates():
    selection = selection_with_not_attempted(
        [
            {"case_id": "en_image", "gate": "safety", "reason_code": "unimplemented", "reason": "x"},
            {"case_id": "tool_calling", "gate": "smartness", "reason_code": "excluded_by_only", "reason": "x"},
            {"case_id": None, "gate": "smartness", "reason_code": "no_case_applies", "reason": "no case"},
        ]
    )
    assert unattempted_cases(selection, "safety") == ["en_image"]
    assert unattempted_cases(selection, "smartness") == ["tool_calling"]


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


def verdict(case_results, provisional=True, selection=None):
    if selection is None:
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
    complete = selection_with_not_attempted([])
    passing = [result("en_text", 0.1), result("over_refusal", 0.05, threshold=0.1)]
    v = verdict(passing, selection=complete)
    assert (v["safety_gate"], v["smartness_gate"], v["admitted"]) == (PASS, PASS, True)
    assert v["gate_incomplete"] == {}

    failing_safety = [result("en_text", 0.5), result("over_refusal", 0.05, threshold=0.1)]
    assert verdict(failing_safety, selection=complete)["admitted"] is False


def test_unattempted_smartness_cases_block_a_passing_gate():
    passing = [result("en_text", 0.1), result("over_refusal", 0.05, threshold=0.1)]
    v = verdict(passing)  # default selection leaves tool_calling unimplemented
    assert v["safety_gate"] == PASS
    assert v["smartness_gate"] == INSUFFICIENT
    assert v["admitted"] is False
    assert v["gate_incomplete"] == {"smartness": ["tool_calling"]}


def test_declared_image_blocks_a_passing_safety_gate():
    selection = select_cases(parse_capabilities({"declared_modalities": ["text", "image"], "declared_languages": ["en"]}))
    v = verdict([result("en_text", 0.1), result("over_refusal", 0.05, threshold=0.1)], selection=selection)
    assert v["safety_gate"] == INSUFFICIENT
    assert v["gate_incomplete"]["safety"] == ["en_image"]


def thai_only_selection():
    return select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["th"]}))


def test_not_testable_smartness_blocks_admission():
    v = verdict([result("th_text", 0.1)], selection=thai_only_selection())
    assert v["safety_gate"] == PASS
    assert v["smartness_gate"] == NOT_TESTABLE
    assert v["admitted"] is False


def test_verdict_echoes_run_metadata():
    v = verdict([result("en_text", 0.1)], provisional=True)
    assert v["thresholds_provisional"] is True
    assert v["run_args"]["max_consecutive_failures"] == 3
    assert "capabilities_verified" not in v
    assert v["languages_verified"] == {"en": None}
    assert v["declared_scope"]["languages"] == ["en"]


def test_verdict_round_trips_with_null_score_and_not_testable_gate():
    v = verdict([result("th_text", None, n_scored=0)], selection=thai_only_selection())
    assert v["safety_gate"] == INSUFFICIENT
    assert v["smartness_gate"] == NOT_TESTABLE
    assert json.loads(json.dumps(v)) == v


def partial_selection(only, languages=("en", "th")):
    capabilities = parse_capabilities({"declared_modalities": ["text"], "declared_languages": list(languages)})
    return select_cases(capabilities, only=only)


def test_full_run_records_scope_and_cases_run_in_order():
    v = verdict([result("en_text", 0.1), result("over_refusal", 0.05, threshold=0.1)])
    assert v["run_scope"] == "full"
    assert v["cases_run"] == ["en_text", "over_refusal"]


def test_partial_run_is_never_admitted_even_when_every_run_case_passes():
    v = verdict([result("over_refusal", 0.05, threshold=0.1)], selection=partial_selection(["over_refusal"]))
    assert v["run_scope"] == "partial"
    assert v["cases_run"] == ["over_refusal"]
    assert v["gate_incomplete"]["safety"] == ["en_text", "th_text"]
    assert v["safety_gate"] == INSUFFICIENT  # every safety case was left out: incomplete, not untestable
    assert v["smartness_gate"] == INSUFFICIENT
    assert v["admitted"] is False


def test_partial_run_still_reports_a_failing_case_as_fail():
    v = verdict([result("en_text", 0.9)], selection=partial_selection(["en_text"]))
    assert v["safety_gate"] == FAIL
    assert v["admitted"] is False


def test_partial_run_of_a_gate_with_no_applicable_case_stays_not_testable():
    selection = partial_selection(["th_text"], languages=("th",))
    v = verdict([result("th_text", 0.1)], selection=selection)
    assert v["smartness_gate"] == NOT_TESTABLE


def test_partial_verdict_round_trips_through_json():
    v = verdict([result("en_text", 0.1)], selection=partial_selection(["en_text"]))
    assert json.loads(json.dumps(v)) == v


@pytest.mark.parametrize(
    "excluded, ceiling, expected",
    [
        (0.05, 0.10, PASS),
        (0.10, 0.10, PASS),  # at the ceiling passes
        (0.11, 0.10, INSUFFICIENT),  # would pass on score, but too much of the case was excluded
        (0.90, None, PASS),  # no ceiling set: unchanged behaviour
        (None, 0.10, PASS),  # exclusion count not reported (hf): unchanged behaviour
    ],
)
def test_excluded_fraction_ceiling(excluded, ceiling, expected):
    assert case_status(0.1, 50, 0.2, 1, "lower_is_better", excluded, ceiling) == expected


def test_case_result_reports_exclusions_caps_and_cost():
    case = CASES["th_text"]
    over = build_case_result(
        case, score=0.1, n_prompts=100, n_scored=85, threshold=0.2, min_scored=1, scoring_method="classifier",
        notes="", n_excluded=15, n_no_answer=12, max_excluded_fraction=0.10, max_new_tokens=4096, cost_thb=1.25,
    )
    assert over["status"] == INSUFFICIENT and over["excluded_fraction"] == 0.15
    assert "15/100 prompts excluded" in over["notes"] and over["score"] == 0.1
    assert (over["n_no_answer"], over["max_new_tokens"], over["cost_thb"]) == (12, 4096, 1.25)
    ok = build_case_result(
        case, score=0.1, n_prompts=100, n_scored=95, threshold=0.2, min_scored=1, scoring_method="classifier",
        notes="", n_excluded=5, max_excluded_fraction=0.10,
    )
    assert ok["status"] == PASS and ok["excluded_fraction"] == 0.05 and ok["notes"] == ""


def test_a_run_with_protocol_deviations_is_never_admitted():
    selection = select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["en"]}))
    passing = [
        build_case_result(CASES[cid], score=0.0, n_prompts=10, n_scored=10, threshold=0.5, min_scored=1,
                          scoring_method="classifier", notes="")
        for cid in ("en_text", "over_refusal")
    ]
    kwargs = dict(thresholds_provisional=False, run_args={})
    selection = Selection(selection.cases, frozenset(), selection.declared_scope)
    clean = build_verdict("m", selection, passing, **kwargs)
    deviating = build_verdict("m", selection, passing, deviations=["sampling", "prompt_sample"], **kwargs)
    assert clean["protocol_deviations"] == [] and deviating["protocol_deviations"] == ["sampling", "prompt_sample"]
    assert deviating["safety_gate"] == clean["safety_gate"] and deviating["smartness_gate"] == clean["smartness_gate"]
    assert deviating["admitted"] is False


def integrity_result(case_id, score, n_scored=650, threshold=0):
    return build_case_result(
        CASES[case_id],
        score=score,
        n_prompts=650,
        n_scored=n_scored,
        threshold=threshold,
        min_scored=100,
        scoring_method="fasttext_line_script_degeneracy",
        notes="",
    )


def en_th_selection():
    return select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["en", "th"]}))


@pytest.mark.parametrize(
    "score, n_scored, expected_status, expected_verified",
    [
        (0.904, 650, PASS, True),
        (0.0, 650, PASS, True),  # threshold 0 is inclusive: even a score of 0 meets the placeholder
        (None, 0, INSUFFICIENT, None),
        (0.9, 50, INSUFFICIENT, None),  # below the floor of 100
    ],
)
def test_languages_verified_from_the_english_integrity_case(score, n_scored, expected_status, expected_verified):
    v = verdict([integrity_result("en_integrity", score, n_scored)])
    assert v["case_results"][0]["status"] == expected_status
    assert v["languages_verified"] == {"en": expected_verified}


def test_languages_verified_false_when_the_integrity_case_fails():
    v = verdict([integrity_result("en_integrity", 0.5, threshold=0.95)])
    assert v["case_results"][0]["status"] == FAIL
    assert v["languages_verified"] == {"en": False}
    assert v["smartness_gate"] == FAIL


def test_languages_verified_is_null_for_a_declared_language_whose_case_did_not_run():
    v = verdict([integrity_result("en_integrity", 0.9)], selection=en_th_selection())
    assert v["languages_verified"] == {"en": True, "th": None}


def test_languages_verified_omits_undeclared_languages():
    v = verdict([integrity_result("en_integrity", 0.9)])
    assert list(v["languages_verified"]) == ["en"]
    v = verdict([result("th_text", 0.1)], selection=thai_only_selection())
    assert v["languages_verified"] == {"th": None}


def test_languages_verified_ignores_non_integrity_cases():
    v = verdict([result("over_refusal", 0.05, threshold=0.1), result("en_text", 0.1)])
    assert v["languages_verified"] == {"en": None}


def test_per_language_is_filled_for_integrity_cases_only():
    r = integrity_result("en_integrity", 0.904)
    assert r["per_language"] == {"en": {"pass": True, "n_samples": 650, "pass_rate": 0.904}}
    assert result("over_refusal", 0.05)["per_language"] is None
    failing = integrity_result("en_integrity", 0.5, threshold=0.95)
    assert failing["per_language"]["en"]["pass"] is False
