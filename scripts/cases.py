#!/usr/bin/env python3
"""
Case registry and selection: turns a candidate model's declared capabilities
into the list of cases the suite runs against it, plus the declared_scope
block recorded in the verdict.

A case whose capability was declared but which isn't implemented yet is
reported in declared_scope["not_attempted"], never silently dropped. A gate
with no selected case is reported as not testable, never as passing. Declared
capabilities always include text, so the safety gate always has a case; the
smartness gate stays not testable until its cases are implemented.

Usage:
    python3 cases.py capabilities.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from capabilities import Capabilities, load_capabilities

GATES = ("safety", "smartness")
DIRECTIONS = ("lower_is_better", "higher_is_better")

NOT_IMPLEMENTED_REASON = "not implemented in this version"
NO_CASE_REASON = "no implemented {gate} case applies to the declared capabilities"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    gate: str
    direction: str
    # None means the case runs regardless of what was declared.
    modality: str | None
    language: str | None
    implemented: bool


REGISTRY: tuple[CaseSpec, ...] = (
    CaseSpec("en_text", "safety", "lower_is_better", "text", "en", True),
    CaseSpec("th_text", "safety", "lower_is_better", "text", "th", True),
    CaseSpec("en_image", "safety", "lower_is_better", "image", "en", False),
    CaseSpec("th_image", "safety", "lower_is_better", "image", "th", False),
    CaseSpec("coding_capability", "smartness", "higher_is_better", None, None, False),
    CaseSpec("over_refusal", "smartness", "lower_is_better", None, None, False),
    # Samples only the declared languages; see Selection.capability_honesty_languages.
    CaseSpec("capability_honesty", "smartness", "higher_is_better", None, None, False),
)


@dataclass(frozen=True)
class Selection:
    cases: tuple[CaseSpec, ...]
    capability_honesty_languages: tuple[str, ...]
    not_testable_gates: frozenset[str]
    declared_scope: dict


def _applies(case: CaseSpec, capabilities: Capabilities) -> bool:
    if case.modality is not None and not capabilities.supports_modality(case.modality):
        return False
    if case.language is not None and not capabilities.supports_language(case.language):
        return False
    return True


def select_cases(capabilities: Capabilities) -> Selection:
    applicable = [case for case in REGISTRY if _applies(case, capabilities)]
    cases = tuple(case for case in applicable if case.implemented)
    not_attempted = [
        {"case_id": case.case_id, "gate": case.gate, "reason": NOT_IMPLEMENTED_REASON}
        for case in applicable
        if not case.implemented
    ]

    selected_gates = {case.gate for case in cases}
    not_testable_gates = frozenset(gate for gate in GATES if gate not in selected_gates)
    for gate in sorted(not_testable_gates):
        not_attempted.append({"case_id": None, "gate": gate, "reason": NO_CASE_REASON.format(gate=gate)})

    return Selection(
        cases=cases,
        capability_honesty_languages=tuple(sorted(capabilities.declared_languages)),
        not_testable_gates=not_testable_gates,
        declared_scope={
            "modalities": sorted(capabilities.declared_modalities),
            "languages": sorted(capabilities.declared_languages),
            "not_attempted": not_attempted,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a capabilities.json file")
    args = parser.parse_args()

    selection = select_cases(load_capabilities(args.path))
    print(f"cases: {[case.case_id for case in selection.cases]}")
    print(f"capability_honesty languages: {list(selection.capability_honesty_languages)}")
    print(f"not testable gates: {sorted(selection.not_testable_gates)}")
    print(json.dumps({"declared_scope": selection.declared_scope}, indent=2))


if __name__ == "__main__":
    main()
