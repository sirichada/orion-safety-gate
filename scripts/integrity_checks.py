"""
Language-independent checks shared by the integrity scorers. Every setting is an
argument; each scorer owns its constants and passes them in.

Importing this module imports no language-ID or tokenizer package.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Callable

REASON_EMPTY = "empty_answer"
REASON_LENGTH = "length_cap"
REASON_SCRIPT = "script_mix"
REASON_NGRAM = "loop_ngram"
REASON_TAIL = "loop_tail"

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class ScriptRule:
    """Letters whose Unicode name starts with an exempt prefix are allowed. Any other letter counts as foreign;
    min_run foreign letters in a row from run_prefixes fail the text, as does a foreign share above max_share
    (None turns the share rule off)."""

    exempt_prefixes: tuple[str, ...]
    run_prefixes: tuple[str, ...]
    min_run: int
    max_share: float | None


def strip_code_and_urls(text: str) -> str:
    text = _FENCE.sub("\n", text)
    text = _INLINE_CODE.sub(" ", text)
    return _URL.sub(" ", text)


def script_failure(text: str, rule: ScriptRule) -> bool:
    letters = foreign = run = 0
    for ch in text:
        if not ch.isalpha():
            run = 0
            continue
        letters += 1
        name = unicodedata.name(ch, "")
        if name.startswith(rule.exempt_prefixes):
            run = 0
            continue
        foreign += 1
        run = run + 1 if name.startswith(rule.run_prefixes) else 0
        if run >= rule.min_run:
            return True
    return rule.max_share is not None and letters > 0 and foreign / letters > rule.max_share


def script_context(text: str, exempt_prefixes: tuple[str, ...], fallback_chars: int) -> str:
    """The first foreign letter with 60 characters either side, or the start of the text when there is none."""
    for i, ch in enumerate(text):
        if ch.isalpha() and not unicodedata.name(ch, "").startswith(exempt_prefixes):
            return text[max(0, i - 60) : i + 60]
    return text[:fallback_chars]


def ngram_failure(tokens: list[str], n: int, repeats: int) -> bool:
    if len(tokens) < n:
        return False
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return counts.most_common(1)[0][1] >= repeats


def tail_failure(text: str, chars: int, min_bytes: int, max_ratio: float) -> bool:
    tail = text[-chars:].encode("utf-8")
    return len(tail) >= min_bytes and len(zlib.compress(tail)) / len(tail) < max_ratio


def loop_failure(
    text: str,
    tokenize: Callable[[str], list[str]],
    n: int,
    repeats: int,
    tail_chars: int,
    tail_min_bytes: int,
    tail_max_ratio: float,
) -> str | None:
    if ngram_failure(tokenize(text), n, repeats):
        return REASON_NGRAM
    if tail_failure(text, tail_chars, tail_min_bytes, tail_max_ratio):
        return REASON_TAIL
    return None
