"""
Deterministic seed derivation from a declared model identifier, so re-runs
against the same model sample the same prompts.
"""

import hashlib


def derive_seed(model_id: str) -> int:
    # sha256, not hash(): Python's str hash is salted per process.
    return int.from_bytes(hashlib.sha256(model_id.encode("utf-8")).digest()[:8], "big")
