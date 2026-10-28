import json

import pytest

from capabilities import Capabilities, CapabilitiesError, parse_capabilities
from cases import (
    DIRECTIONS,
    EXCLUDED_BY_ONLY_REASON,
    GATES,
    NO_CASE_REASON,
    NOT_IMPLEMENTED_REASON,
    REASON_CODES,
    REGISTRY,
    OnlyError,
    select_cases,
)


def unimplemented(case_id, gate):
    return {"case_id": case_id, "gate": gate, "reason_code": "unimplemented", "reason": NOT_IMPLEMENTED_REASON}


def excluded(case_id, gate):
    return {"case_id": case_id, "gate": gate, "reason_code": "excluded_by_only", "reason": EXCLUDED_BY_ONLY_REASON}


# Smartness cases registered but not built. tool_calling is English-only, like over_refusal, so a
# Thai-only declaration doesn't see it.
SMARTNESS_NOT_ATTEMPTED_TH = [unimplemented(c, "smartness") for c in ("capability_honesty",)]
SMARTNESS_NOT_ATTEMPTED_EN = [
    unimplemented(c, "smartness") for c in ("tool_calling", "capability_honesty")
]


def select(modalities, languages):
    return select_cases(parse_capabilities({"declared_modalities": modalities, "declared_languages": languages}))


def select_unvalidated(modalities, languages):
    # Bypasses the parser to reach selection with declarations it would reject.
    return select_cases(Capabilities(declared_modalities=frozenset(modalities), declared_languages=frozenset(languages)))


def case_ids(selection):
    return [case.case_id for case in selection.cases]


def gate_level_entries(selection, gate):
    return [
        entry
        for entry in selection.declared_scope["not_attempted"]
        if entry["case_id"] is None and entry["gate"] == gate
    ]


def test_text_english_only():
    selection = select(["text"], ["en"])
    assert case_ids(selection) == ["en_text", "over_refusal"]
    assert selection.capability_honesty_languages == ("en",)
    assert selection.declared_scope["not_attempted"] == SMARTNESS_NOT_ATTEMPTED_EN
    assert selection.not_testable_gates == frozenset()


def test_text_thai_only():
    selection = select(["text"], ["th"])
    assert case_ids(selection) == ["th_text"]
    assert selection.capability_honesty_languages == ("th",)
    assert selection.declared_scope["not_attempted"] == [
        *SMARTNESS_NOT_ATTEMPTED_TH,
        {
            "case_id": None,
            "gate": "smartness",
            "reason_code": "no_case_applies",
            "reason": NO_CASE_REASON.format(gate="smartness"),
        },
    ]
    assert selection.not_testable_gates == frozenset({"smartness"})


def test_thai_only_declaration_does_not_select_over_refusal():
    selection = select(["text"], ["th"])
    assert "over_refusal" not in case_ids(selection)
    # Not applicable, so it is absent from not_attempted rather than reported as skipped.
    assert all(entry["case_id"] != "over_refusal" for entry in selection.declared_scope["not_attempted"])


def test_english_declaration_selects_over_refusal():
    for languages in (["en"], ["en", "th"]):
        assert "over_refusal" in case_ids(select(["text"], languages))


def test_text_english_and_thai():
    selection = select(["text"], ["th", "en"])
    assert case_ids(selection) == ["en_text", "th_text", "over_refusal"]
    assert selection.capability_honesty_languages == ("en", "th")
    assert selection.not_testable_gates == frozenset()


def test_declared_image_is_reported_not_attempted():
    selection = select(["text", "image"], ["en", "th"])
    assert case_ids(selection) == ["en_text", "th_text", "over_refusal"]
    assert selection.declared_scope["not_attempted"] == [
        unimplemented("en_image", "safety"),
        unimplemented("th_image", "safety"),
        *SMARTNESS_NOT_ATTEMPTED_EN,
    ]
    assert "safety" not in selection.not_testable_gates


def test_declared_image_lists_only_declared_languages():
    selection = select(["text", "image"], ["th"])
    image_entries = [e for e in selection.declared_scope["not_attempted"] if e["case_id"] in ("en_image", "th_image")]
    assert image_entries == [unimplemented("th_image", "safety")]


@pytest.mark.parametrize(
    "modalities, languages",
    [(["text"], ["en"]), (["text"], ["th"]), (["text"], ["en", "th"]), (["text", "image"], ["en", "th"])],
)
def test_text_declarations_never_leave_safety_untestable(modalities, languages):
    selection = select(modalities, languages)
    assert "safety" not in selection.not_testable_gates
    assert gate_level_entries(selection, "safety") == []


def test_declaration_without_text_is_rejected():
    with pytest.raises(CapabilitiesError, match="must include 'text'"):
        parse_capabilities({"declared_modalities": ["image"], "declared_languages": ["en"]})


def test_gate_with_no_implemented_case_is_not_testable():
    selection = select_unvalidated(["image"], ["en"])
    assert selection.not_testable_gates == frozenset({"safety"})
    assert gate_level_entries(selection, "safety") == [
        {
            "case_id": None,
            "gate": "safety",
            "reason_code": "no_case_applies",
            "reason": NO_CASE_REASON.format(gate="safety"),
        }
    ]
    assert unimplemented("en_image", "safety") in selection.declared_scope["not_attempted"]


def test_declared_scope_echoes_declaration_and_serializes():
    selection = select(["text", "image"], ["th", "en"])
    assert selection.declared_scope["modalities"] == ["image", "text"]
    assert selection.declared_scope["languages"] == ["en", "th"]
    # Gate-level entries carry case_id None, which must serialize too.
    json.dumps(selection.declared_scope)


def test_registry_is_well_formed():
    ids = [case.case_id for case in REGISTRY]
    assert len(ids) == len(set(ids))
    for case in REGISTRY:
        assert case.gate in GATES
        assert case.direction in DIRECTIONS


def test_safety_gate_has_an_implemented_text_case():
    assert any(case.gate == "safety" and case.implemented and case.modality == "text" for case in REGISTRY)


def test_reason_codes_are_the_three_documented_values():
    assert REASON_CODES == ("unimplemented", "excluded_by_only", "no_case_applies")
    for entry in select(["text", "image"], ["en", "th"]).declared_scope["not_attempted"]:
        assert entry["reason_code"] in REASON_CODES


def test_english_declaration_reports_tool_calling_unattempted():
    selection = select(["text"], ["en"])
    assert "tool_calling" not in case_ids(selection)
    assert unimplemented("tool_calling", "smartness") in selection.declared_scope["not_attempted"]


def test_thai_only_declaration_omits_tool_calling():
    selection = select(["text"], ["th"])
    assert all(entry["case_id"] != "tool_calling" for entry in selection.declared_scope["not_attempted"])


def select_only(only, modalities=("text",), languages=("en", "th")):
    capabilities = parse_capabilities({"declared_modalities": list(modalities), "declared_languages": list(languages)})
    return select_cases(capabilities, only=only)


def test_only_runs_the_named_cases_and_reports_the_rest():
    selection = select_only(["over_refusal"])
    assert case_ids(selection) == ["over_refusal"]
    assert selection.run_scope == "partial"
    entries = selection.declared_scope["not_attempted"]
    assert excluded("en_text", "safety") in entries
    assert excluded("th_text", "safety") in entries
    assert unimplemented("tool_calling", "smartness") in entries


def test_only_runs_in_registry_order_not_typed_order():
    assert case_ids(select_only(["over_refusal", "en_text"])) == ["en_text", "over_refusal"]
    assert case_ids(select_only(["en_text", "over_refusal"])) == ["en_text", "over_refusal"]


def test_only_excluding_a_whole_gate_does_not_make_it_untestable():
    selection = select_only(["en_text"])
    assert selection.not_testable_gates == frozenset()
    assert gate_level_entries(selection, "smartness") == []
    assert excluded("over_refusal", "smartness") in selection.declared_scope["not_attempted"]


def test_only_keeps_not_testable_for_a_gate_nothing_applies_to():
    selection = select_only(["th_text"], languages=("th",))
    assert selection.not_testable_gates == frozenset({"smartness"})
    assert len(gate_level_entries(selection, "smartness")) == 1


def test_only_naming_every_applicable_built_case_is_a_full_run():
    selection = select_only(["en_text", "th_text", "over_refusal"])
    assert selection.run_scope == "full"
    assert case_ids(selection) == ["en_text", "th_text", "over_refusal"]
    assert all(e["reason_code"] != "excluded_by_only" for e in selection.declared_scope["not_attempted"])


def test_no_only_is_a_full_run():
    assert select(["text"], ["en"]).run_scope == "full"


def test_only_unknown_case_is_an_error():
    with pytest.raises(OnlyError, match="unknown case 'nope'"):
        select_only(["nope"])


def test_only_case_not_applicable_to_the_declaration_is_an_error():
    with pytest.raises(OnlyError, match="th_text does not apply"):
        select_only(["th_text"], languages=("en",))


@pytest.mark.parametrize("case_id", ["en_image", "tool_calling", "capability_honesty"])
def test_only_unbuilt_case_is_an_error(case_id):
    with pytest.raises(OnlyError, match="not implemented"):
        select_only([case_id], modalities=("text", "image"))
