import json

import pytest

from capabilities import Capabilities, CapabilitiesError, parse_capabilities
from cases import DIRECTIONS, GATES, NO_CASE_REASON, NOT_IMPLEMENTED_REASON, REGISTRY, select_cases

SMARTNESS = ["coding_capability", "over_refusal", "capability_honesty"]


def select(modalities, languages):
    return select_cases(parse_capabilities({"declared_modalities": modalities, "declared_languages": languages}))


def select_unvalidated(modalities, languages):
    # Bypasses the parser to reach selection with declarations it would reject.
    return select_cases(Capabilities(declared_modalities=frozenset(modalities), declared_languages=frozenset(languages)))


def case_ids(selection):
    return [case.case_id for case in selection.cases]


def gate_level_entries(selection):
    return [entry for entry in selection.declared_scope["not_attempted"] if entry["case_id"] is None]


def test_text_english_only():
    selection = select(["text"], ["en"])
    assert case_ids(selection) == ["en_text", *SMARTNESS]
    assert selection.capability_honesty_languages == ("en",)
    assert selection.declared_scope["not_attempted"] == []
    assert selection.not_testable_gates == frozenset()


def test_text_thai_only():
    selection = select(["text"], ["th"])
    assert case_ids(selection) == ["th_text", *SMARTNESS]
    assert selection.capability_honesty_languages == ("th",)
    assert selection.declared_scope["not_attempted"] == []


def test_text_english_and_thai():
    selection = select(["text"], ["th", "en"])
    assert case_ids(selection) == ["en_text", "th_text", *SMARTNESS]
    assert selection.capability_honesty_languages == ("en", "th")
    assert selection.not_testable_gates == frozenset()


def test_declared_image_is_reported_not_attempted():
    selection = select(["text", "image"], ["en", "th"])
    assert case_ids(selection) == ["en_text", "th_text", *SMARTNESS]
    assert selection.declared_scope["not_attempted"] == [
        {"case_id": "en_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON},
        {"case_id": "th_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON},
    ]
    assert selection.not_testable_gates == frozenset()


def test_declared_image_lists_only_declared_languages():
    selection = select(["text", "image"], ["th"])
    assert selection.declared_scope["not_attempted"] == [
        {"case_id": "th_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON},
    ]


@pytest.mark.parametrize(
    "modalities, languages",
    [(["text"], ["en"]), (["text"], ["th"]), (["text"], ["en", "th"]), (["text", "image"], ["en", "th"])],
)
def test_text_declarations_never_leave_a_gate_untestable(modalities, languages):
    selection = select(modalities, languages)
    assert selection.not_testable_gates == frozenset()
    assert gate_level_entries(selection) == []


def test_declaration_without_text_is_rejected():
    with pytest.raises(CapabilitiesError, match="must include 'text'"):
        parse_capabilities({"declared_modalities": ["image"], "declared_languages": ["en"]})


def test_gate_with_no_implemented_case_is_not_testable():
    selection = select_unvalidated(["image"], ["en"])
    assert selection.not_testable_gates == frozenset({"safety"})
    assert gate_level_entries(selection) == [
        {"case_id": None, "gate": "safety", "reason": NO_CASE_REASON.format(gate="safety")}
    ]
    assert {"case_id": "en_image", "gate": "safety", "reason": NOT_IMPLEMENTED_REASON} in selection.declared_scope[
        "not_attempted"
    ]


def test_declared_scope_echoes_declaration_and_serializes():
    selection = select(["text", "image"], ["th", "en"])
    assert selection.declared_scope["modalities"] == ["image", "text"]
    assert selection.declared_scope["languages"] == ["en", "th"]
    json.dumps(selection.declared_scope)
    # Gate-level entries carry case_id None, which must serialize too.
    json.dumps(select_unvalidated(["image"], ["en"]).declared_scope)


def test_registry_is_well_formed():
    ids = [case.case_id for case in REGISTRY]
    assert len(ids) == len(set(ids))
    for case in REGISTRY:
        assert case.gate in GATES
        assert case.direction in DIRECTIONS


def test_every_gate_has_an_implemented_text_case():
    for gate in GATES:
        assert any(case.gate == gate and case.implemented and case.modality in (None, "text") for case in REGISTRY)
