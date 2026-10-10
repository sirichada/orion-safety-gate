import json

import pytest

from capabilities import Capabilities, CapabilitiesError, parse_capabilities
from cases import DIRECTIONS, GATES, NO_CASE_REASON, NOT_IMPLEMENTED_REASON, REGISTRY, select_cases

# Smartness cases that are registered but not implemented yet, so every declaration reports them.
SMARTNESS_UNIMPLEMENTED = ["coding_capability", "capability_honesty"]
SMARTNESS_NOT_ATTEMPTED = [
    {"case_id": case_id, "gate": "smartness", "reason": NOT_IMPLEMENTED_REASON} for case_id in SMARTNESS_UNIMPLEMENTED
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
    assert selection.declared_scope["not_attempted"] == SMARTNESS_NOT_ATTEMPTED
    assert selection.not_testable_gates == frozenset()


def test_text_thai_only():
    selection = select(["text"], ["th"])
    assert case_ids(selection) == ["th_text"]
    assert selection.capability_honesty_languages == ("th",)
    assert selection.declared_scope["not_attempted"] == [
        *SMARTNESS_NOT_ATTEMPTED,
        {"case_id": None, "gate": "smartness", "reason": NO_CASE_REASON.format(gate="smartness")},
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
        {"case_id": "en_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON},
        {"case_id": "th_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON},
        *SMARTNESS_NOT_ATTEMPTED,
    ]
    assert "safety" not in selection.not_testable_gates


def test_declared_image_lists_only_declared_languages():
    selection = select(["text", "image"], ["th"])
    image_entries = [e for e in selection.declared_scope["not_attempted"] if e["case_id"] in ("en_image", "th_image")]
    assert image_entries == [{"case_id": "th_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON}]


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
        {"case_id": None, "gate": "safety", "reason": NO_CASE_REASON.format(gate="safety")}
    ]
    assert {"case_id": "en_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON} in selection.declared_scope[
        "not_attempted"
    ]


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
