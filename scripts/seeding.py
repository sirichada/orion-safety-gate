"""
Deterministic seed derivation from a string. case_summary.py seeds it with
the model id and case id to pick the sample replies shown in the report, so
a re-score shows the same replies. It plays no part in which prompts a run
sends.
"""

import hashlib


def derive_seed(model_id: str) -> int:
    # sha256, not hash(): Python's str hash is salted per process.
    return int.from_bytes(hashlib.sha256(model_id.encode("utf-8")).digest()[:8], "big")
