#!/usr/bin/env python3
"""
Parses and validates a candidate model's declared capabilities -- which
modalities (text/image) and languages (en/th) it claims to support. This is
a self-reported claim, not a verified fact: it decides which grid cells and
checks to run, but doesn't confirm the claim is true.

Expected JSON shape:
    {
      "declared_modalities": ["text"],
      "declared_languages": ["en", "th"]
    }

Usage:
    python3 capabilities.py capabilities.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ALLOWED_MODALITIES = {"text", "image"}
ALLOWED_LANGUAGES = {"en", "th"}


class CapabilitiesError(ValueError):
    """Raised when a capabilities file is missing fields or has invalid values."""


@dataclass(frozen=True)
class Capabilities:
    declared_modalities: frozenset[str]
    declared_languages: frozenset[str]

    def supports_modality(self, modality: str) -> bool:
        return modality in self.declared_modalities

    def supports_language(self, language: str) -> bool:
        return language in self.declared_languages


def _validate_field(data: dict, field: str, allowed: set[str]) -> frozenset[str]:
    if field not in data:
        raise CapabilitiesError(f"Missing required field: {field!r}")

    value = data[field]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CapabilitiesError(f"{field!r} must be a list of strings, got {value!r}")

    if not value:
        raise CapabilitiesError(f"{field!r} must not be empty")

    unknown = set(value) - allowed
    if unknown:
        raise CapabilitiesError(f"{field!r} contains unrecognized value(s) {sorted(unknown)}; allowed: {sorted(allowed)}")

    return frozenset(value)


def parse_capabilities(data: dict) -> Capabilities:
    """Validates a raw dict (already JSON-decoded) against the capabilities schema."""
    if not isinstance(data, dict):
        raise CapabilitiesError(f"Expected a JSON object, got {type(data).__name__}")

    modalities = _validate_field(data, "declared_modalities", ALLOWED_MODALITIES)
    languages = _validate_field(data, "declared_languages", ALLOWED_LANGUAGES)

    # Every check the suite runs sends text prompts, so a model must accept text to be tested at all.
    if "text" not in modalities:
        raise CapabilitiesError(f"'declared_modalities' must include 'text', got {sorted(modalities)}")

    return Capabilities(declared_modalities=modalities, declared_languages=languages)


def load_capabilities(path: str) -> Capabilities:
    """Reads and validates a capabilities JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_capabilities(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a capabilities.json file")
    args = parser.parse_args()

    capabilities = load_capabilities(args.path)
    print(f"declared_modalities: {sorted(capabilities.declared_modalities)}")
    print(f"declared_languages: {sorted(capabilities.declared_languages)}")


if __name__ == "__main__":
    main()
