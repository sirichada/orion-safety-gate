"""Pins the English helper outputs on fixed texts so moving them cannot change behaviour."""

import hashlib

import pytest

import score_en_integrity as s


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:12]

PROSE = " ".join(f"word{i}" for i in range(60))
LOOP_SENTENCE = "the quick brown fox jumps over the lazy dog again and again"

TEXTS = {
    "plain": "This is a plain English answer with enough words to be a normal reply.",
    "empty": "",
    "digits": "12345 67890 !!! ???",
    "thai": "สวัสดีครับ วันนี้อากาศดีมาก",
    "cjk_run": "The answer is 如何应对 in this case.",
    "cjk_single_short": "Use 的 here",
    "cjk_single_long": PROSE + " 的 " + PROSE,
    "hangul_run": "Hello 안녕하세요 world",
    "kana_run": "Hello こんにちは world",
    "cyrillic_run": "Hello привет world",
    "arabic_run": "Hello مرحبا world",
    "greek": "alpha α β γ δ is a sequence",
    "math_alphabet": "Let 𝑥 and 𝑦 be reals, 𝔽 a field",
    "double_struck": "The set ℝ of reals and ℕ of naturals",
    "accented_latin": "Café résumé naïve Zoë über",
    "share_just_over": "a" * 99 + "的" * 2,
    "share_just_under": "a" * 300 + "的",
    "fenced": "Intro\n```python\nprint('如何应对')\n```\nOutro",
    "unterminated_fence": "Intro\n```\n如何应对 never closed",
    "inline_code": "Run `如何应对` now",
    "url": "See https://example.com/如何应对 for details",
    "repeat_three": " ".join([LOOP_SENTENCE] * 3),
    "repeat_four": " ".join([LOOP_SENTENCE] * 4),
    "repeat_eight": " ".join([LOOP_SENTENCE] * 8),
    "short_repeat": "ab " * 50,
    "tail_loop": PROSE + "\n" + "z" * 2000,
    "tail_just_short": "z" * 500,
    "tail_ordinary": " ".join(f"token{i}x{i * 7 % 13}" for i in range(400)),
    "mixed_blank_lines": "line one\n\n\nline two has several more words here\n",
    "emoji_punct": "Great! 😀 Done — see you.",
}

# (script_failure, digest of script_context, loop_failure, digest of strip_code_and_urls) per text.
GOLDEN = {'accented_latin': (False, 'a3b843562d4e', None, 'a3b843562d4e'),
 'arabic_run': (True, '4eb48c4c9222', None, '4eb48c4c9222'),
 'cjk_run': (True, '01a6d5b6868d', None, '01a6d5b6868d'),
 'cjk_single_long': (False, 'c66c826cdfae', 'loop_tail', 'ec8165c3f746'),
 'cjk_single_short': (True, 'd97e1ca89ee8', None, 'd97e1ca89ee8'),
 'cyrillic_run': (True, 'dd37353c8cfb', None, 'dd37353c8cfb'),
 'digits': (False, 'a02e9be41136', None, 'a02e9be41136'),
 'double_struck': (False, '5a5e63b2608a', None, '5a5e63b2608a'),
 'emoji_punct': (False, 'd0c403068ce0', None, 'd0c403068ce0'),
 'empty': (False, 'e3b0c44298fc', None, 'e3b0c44298fc'),
 'fenced': (True, '27e902ef56e5', None, '86977e156278'),
 'greek': (False, '7925812fe3fa', None, '7925812fe3fa'),
 'hangul_run': (True, '34ec2402b870', None, '34ec2402b870'),
 'inline_code': (True, '9e9e0b88b351', None, 'edbbefe53db0'),
 'kana_run': (True, 'fa5ec1131092', None, 'fa5ec1131092'),
 'math_alphabet': (False, 'f5527112de81', None, 'f5527112de81'),
 'mixed_blank_lines': (False, '330d6f8b0fc2', None, '330d6f8b0fc2'),
 'plain': (False, 'e8c1af12a155', None, 'e8c1af12a155'),
 'repeat_eight': (False, 'c9abcf1e0a1d', 'loop_ngram', '7c730ce78517'),
 'repeat_four': (False, 'c9abcf1e0a1d', 'loop_ngram', '6fa680037080'),
 'repeat_three': (False, '1558a17c4502', None, '1558a17c4502'),
 'share_just_over': (True, 'dbbe1d7ccdb3', None, '2e7b0d677e7b'),
 'share_just_under': (False, 'a2380df43fe5', None, '1a716a055921'),
 'short_repeat': (False, 'bbef2ef7ebb9', 'loop_ngram', 'bbef2ef7ebb9'),
 'tail_just_short': (False, '983a71da8178', None, '36559492ba4d'),
 'tail_loop': (False, 'c54a02472785', 'loop_tail', 'a21c3e82c865'),
 'tail_ordinary': (False, 'b39d79e9eeb5', None, '7831436ab370'),
 'thai': (True, '6d5651083d1e', None, '6d5651083d1e'),
 'unterminated_fence': (True, '25a80ed1a053', None, 'b221e10fff81'),
 'url': (True, '84ef7eec18fb', None, '18d6ac764b6f')}


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_helper_outputs_are_unchanged(name):
    text = TEXTS[name]
    stripped = s.strip_code_and_urls(text)
    got = (s.script_failure(text), digest(s.script_context(text)), s.loop_failure(text), digest(stripped))
    assert got == GOLDEN[name]
