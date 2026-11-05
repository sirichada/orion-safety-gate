#!/usr/bin/env python3
"""
Case registry and selection: turns a candidate model's declared capabilities
into the list of cases the suite runs against it, plus the declared_scope
block recorded in the verdict.

A derived case generates nothing: it scores the saved replies of its source
cases from the same run, so it runs after them and `only` must name all of them.

A case whose capability was declared but which isn't implemented yet is
reported in declared_scope["not_attempted"], never silently dropped. A gate
with no applicable implemented case is reported as not testable, never as
passing. Declared capabilities always include text, so the safety gate always
has a case; the smartness gate has an implemented case only when English is
declared (over_refusal's prompts and matcher are English-only), and is
otherwise not testable. A gate with any declared-but-unimplemented case, or any
case left out by `only`, cannot pass (see verdict.py).

Usage:
    python3 cases.py capabilities.json [--only case_id[,case_id...]]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Sequence

from capabilities import Capabilities, load_capabilities

GATES = ("safety", "smartness")
DIRECTIONS = ("lower_is_better", "higher_is_better")

REASON_UNIMPLEMENTED = "unimplemented"
REASON_EXCLUDED_BY_ONLY = "excluded_by_only"
REASON_NO_CASE_APPLIES = "no_case_applies"
REASON_CODES = (REASON_UNIMPLEMENTED, REASON_EXCLUDED_BY_ONLY, REASON_NO_CASE_APPLIES)

NOT_IMPLEMENTED_REASON = "not implemented in this version"
EXCLUDED_BY_ONLY_REASON = "left out by --only"
NO_CASE_REASON = "no implemented {gate} case applies to the declared capabilities"

RUN_SCOPES = ("full", "partial")

# The cases that check a language declaration is true; verdict.languages_verified reads only these.
INTEGRITY_CASE_IDS = ("en_integrity", "th_integrity")


class OnlyError(ValueError):
    """`only` names a case the suite cannot run for this candidate."""


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    gate: str
    direction: str
    # None means the case runs regardless of what was declared.
    modality: str | None
    language: str | None
    implemented: bool
    # Cases whose saved replies this case scores; empty for a case that generates its own.
    sources: tuple[str, ...] = ()

    @property
    def derived(self) -> bool:
        return bool(self.sources)


REGISTRY: tuple[CaseSpec, ...] = (
    CaseSpec("en_text", "safety", "lower_is_better", "text", "en", True),
    CaseSpec("th_text", "safety", "lower_is_better", "text", "th", True),
    CaseSpec("en_image", "safety", "lower_is_better", "image", "en", False),
    CaseSpec("th_image", "safety", "lower_is_better", "image", "th", False),
    CaseSpec("over_refusal", "smartness", "lower_is_better", None, "en", True),
    CaseSpec("tool_calling", "smartness", "higher_is_better", None, "en", False),
    CaseSpec("en_integrity", "smartness", "higher_is_better", None, "en", True, ("over_refusal", "en_text")),
    CaseSpec("th_integrity", "smartness", "higher_is_better", None, "th", True, ("th_text",)),
    CaseSpec("th_knowledge", "smartness", "higher_is_better", None, "th", True),
)


@dataclass(frozen=True)
class Selection:
    cases: tuple[CaseSpec, ...]
    not_testable_gates: frozenset[str]
    declared_scope: dict
    run_scope: str = "full"

    @property
    def generating_cases(self) -> tuple[CaseSpec, ...]:
        """The cases that send prompts to the candidate; derived cases only score what these saved."""
        return tuple(case for case in self.cases if not case.derived)


def _applies(case: CaseSpec, capabilities: Capabilities) -> bool:
    if case.modality is not None and not capabilities.supports_modality(case.modality):
        return False
    if case.language is not None and not capabilities.supports_language(case.language):
        return False
    return True


def _check_only(only: Sequence[str], capabilities: Capabilities) -> None:
    by_id = {case.case_id: case for case in REGISTRY}
    for case_id in only:
        case = by_id.get(case_id)
        if case is None:
            raise OnlyError(f"--only: unknown case {case_id!r}; known cases: {', '.join(by_id)}")
        if not _applies(case, capabilities):
            raise OnlyError(
                f"--only: {case_id} does not apply to the declared capabilities "
                f"(modalities {sorted(capabilities.declared_modalities)}, "
                f"languages {sorted(capabilities.declared_languages)})"
            )
        if not case.implemented:
            raise OnlyError(f"--only: {case_id} is registered but {NOT_IMPLEMENTED_REASON}")
        missing = [source for source in case.sources if source not in only]
        if missing:
            raise OnlyError(
                f"--only: {case_id} scores the saved replies of {', '.join(case.sources)}; "
                f"also name {', '.join(missing)}"
            )


def select_cases(capabilities: Capabilities, only: Sequence[str] | None = None) -> Selection:
    if only is not None:
        _check_only(only, capabilities)
    wanted = None if only is None else frozenset(only)

    applicable = [case for case in REGISTRY if _applies(case, capabilities)]
    cases: list[CaseSpec] = []
    not_attempted = []
    for case in applicable:
        if not case.implemented:
            code, reason = REASON_UNIMPLEMENTED, NOT_IMPLEMENTED_REASON
        elif wanted is not None and case.case_id not in wanted:
            code, reason = REASON_EXCLUDED_BY_ONLY, EXCLUDED_BY_ONLY_REASON
        else:
            cases.append(case)
            continue
        not_attempted.append({"case_id": case.case_id, "gate": case.gate, "reason_code": code, "reason": reason})

    # Judged on the applicable implemented cases, before `only`, so leaving a gate's cases out
    # of a partial run does not make the gate untestable.
    testable_gates = {case.gate for case in applicable if case.implemented}
    not_testable_gates = frozenset(gate for gate in GATES if gate not in testable_gates)
    for gate in sorted(not_testable_gates):
        not_attempted.append(
            {
                "case_id": None,
                "gate": gate,
                "reason_code": REASON_NO_CASE_APPLIES,
                "reason": NO_CASE_REASON.format(gate=gate),
            }
        )

    partial = any(entry["reason_code"] == REASON_EXCLUDED_BY_ONLY for entry in not_attempted)
    return Selection(
        cases=tuple(cases),
        not_testable_gates=not_testable_gates,
        declared_scope={
            "modalities": sorted(capabilities.declared_modalities),
            "languages": sorted(capabilities.declared_languages),
            "not_attempted": not_attempted,
        },
        run_scope="partial" if partial else "full",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a capabilities.json file")
    parser.add_argument("--only", default=None, help="Comma-separated case ids to run")
    args = parser.parse_args()

    only = [part.strip() for part in args.only.split(",")] if args.only else None
    selection = select_cases(load_capabilities(args.path), only=only)
    print(f"cases: {[case.case_id for case in selection.cases]}")
    print(f"run scope: {selection.run_scope}")
    print(f"not testable gates: {sorted(selection.not_testable_gates)}")
    print(json.dumps({"declared_scope": selection.declared_scope}, indent=2))


if __name__ == "__main__":
    main()
