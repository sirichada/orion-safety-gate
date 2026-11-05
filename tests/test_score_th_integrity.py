import csv
import json
import random

import pytest

import score_th_integrity as s
from integrity_checks import REASON_NGRAM, REASON_TAIL
from score_th_integrity import aggregate, score_records

THAI = "ผู้ใช้งานที่ได้รับอนุญาตสามารถเข้าถึงระบบได้"
FOREIGN_LINE = "This line is written entirely in English words"


def rec(response, rid="r1", source="th_text", **extra):
    return {"id": rid, "prompt": "p", "response": response, "generation_failed": False, "source": source, **extra}


def score(records, **kwargs):
    kwargs.setdefault("tokenize", str.split)
    return score_records(records, **kwargs)


def one(response, **kwargs):
    extra = kwargs.pop("extra", {})
    return score([rec(response, **extra)], **kwargs)[0]


CONSONANTS = [chr(c) for c in range(0x0E01, 0x0E2F)]
LATIN_WORDS = "river stone market quiet engine garden window paper story light bridge winter coffee signal".split()


def thai_line(seed):
    rng = random.Random(seed)
    return "".join(rng.choice(CONSONANTS) for _ in range(40))


def foreign_line(seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(LATIN_WORDS) for _ in range(8))


def thai_lines(n_thai, n_foreign=0):
    """Distinct lines, so the loop checks never fire."""
    return "\n".join([thai_line(i) for i in range(n_thai)] + [foreign_line(1000 + i) for i in range(n_foreign)])


# ---- Thai letter ratio ----------------------------------------------------------------------


def test_pure_thai_reply_passes():
    r = one(THAI)
    assert r["label"] == 1 and r["failure_reason"] is None and r["lines_checked"] == 1


def test_english_only_line_fails_the_ratio():
    r = one(FOREIGN_LINE)
    assert r["label"] == 0 and r["failure_reason"] == s.REASON_THAI_RATIO and r["lines_non_th"] == 1


def test_marks_leave_both_sides_so_the_ratio_never_exceeds_one():
    thai, letters = s.thai_letter_counts("ผู้ใช้งานที่ได้รับอนุญาต")
    assert thai == letters
    assert s.checked_lines("ผู้ใช้งานที่ได้รับอนุญาต", 5)[0][1] == 1.0


def test_marks_do_not_count_as_thai_letters_against_latin_letters():
    line = "ผู้ใช้งาน" + "abcdefgh"
    thai, letters = s.thai_letter_counts(line)
    assert (thai, letters) == (6, 14)


def test_ratio_of_exactly_one_half_passes():
    r = one("กขคงจ" + "abcde")
    assert r["lines_checked"] == 1 and r["label"] == 1


def test_just_below_one_half_fails():
    r = one("กขคงจ" + "abcdef")
    assert r["label"] == 0 and r["failure_reason"] == s.REASON_THAI_RATIO


def test_line_with_too_few_letters_is_not_checked():
    r = one("ok\n1234 5678\n- - -\n" + THAI)
    assert r["lines_checked"] == 1 and r["label"] == 1


def test_digits_and_punctuation_are_not_letters():
    assert s.thai_letter_counts("๑๒๓ 123 !!! ---") == (0, 0)


def test_min_line_letters_is_a_setting():
    assert one("abcd", min_line_letters=5)["lines_checked"] == 0
    assert one("abcd", min_line_letters=4)["lines_checked"] == 1


def test_min_thai_ratio_is_a_setting():
    line = "กขคงจ" + "abcdef"
    assert one(line, min_thai_ratio=0.5)["label"] == 0
    assert one(line, min_thai_ratio=0.4)["label"] == 1


# ---- line rules -----------------------------------------------------------------------------


def test_any_line_fails_on_one_foreign_line():
    r = one(thai_lines(9, 1), line_rule="any_line")
    assert r["label"] == 0 and r["reasons_any_line"] == [s.REASON_THAI_RATIO] and r["reasons_share_of_lines"] == []


def test_share_of_lines_tolerates_a_minority_of_foreign_lines():
    r = one(thai_lines(9, 1), line_rule="share_of_lines", share_threshold=0.25)
    assert r["label"] == 1 and r["reasons_any_line"] == [s.REASON_THAI_RATIO]


def test_share_of_lines_boundary_is_not_inclusive_of_the_threshold():
    assert one(thai_lines(3, 1), line_rule="share_of_lines", share_threshold=0.25)["label"] == 1
    assert one(thai_lines(2, 1), line_rule="share_of_lines", share_threshold=0.25)["label"] == 0


def test_unknown_line_rule_is_rejected():
    with pytest.raises(ValueError, match="line_rule"):
        score([rec(THAI)], line_rule="nope")


def test_missing_response_field_is_rejected():
    with pytest.raises(ValueError, match="response"):
        score([{"id": "x"}])


# ---- script mix -----------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["วางแผน如何应对เจ้านาย", "**升级为ความรุนแรงขั้นสูง**"])
def test_cjk_inside_thai_fails_script_mix(text):
    r = one(THAI + "\n" + text)
    assert s.REASON_SCRIPT in r["reasons_any_line"] and r["label"] == 0


def test_lone_cjk_character_between_thai_words_is_under_the_share():
    text = thai_lines(10)
    assert one(text.replace(text[5], "的", 1))["failure_reason"] is None


def test_lone_cjk_character_over_the_share_fails():
    r = one("กขค的งจฉ", foreign_letter_share=0.05)
    assert s.REASON_SCRIPT in r["reasons_any_line"]


def test_foreign_share_is_a_setting():
    text = " ".join([THAI] * 3) + " 的"
    assert one(text, foreign_letter_share=0.5)["failure_reason"] is None
    assert one(text, foreign_letter_share=0.001)["failure_reason"] == s.REASON_SCRIPT


@pytest.mark.parametrize("run", ["привет", "مرحبا", "안녕하세요", "こんにちは"])
def test_other_script_runs_fail(run):
    assert s.REASON_SCRIPT in one(THAI + " " + run)["reasons_any_line"]


def test_latin_greek_and_math_alphabet_in_a_thai_reply_pass_the_script_check():
    text = THAI + " Python API α β 𝑥 ℝ"
    assert s.REASON_SCRIPT not in one(text)["reasons_any_line"]


def test_code_and_urls_are_stripped_before_every_check():
    text = THAI + "\n```python\nprint('如何应对')\nenglish only line here\n```\nดูรายละเอียดเพิ่มเติมที่ https://example.com/如何应对"
    r = one(text)
    assert r["label"] == 1 and r["lines_checked"] == 2


# ---- empty, length, exclusion ---------------------------------------------------------------


@pytest.mark.parametrize("response", ["", "  \n\t"])
def test_empty_reply_is_a_failure_not_an_exclusion(response):
    r = one(response)
    assert r["excluded"] is False and r["label"] == 0 and r["failure_reason"] == s.REASON_EMPTY


def test_no_answer_reply_flagged_generation_failed_is_a_failure():
    r = one("", extra={"generation_failed": True, "no_answer": True})
    assert r["excluded"] is False and r["failure_reason"] == s.REASON_EMPTY


def test_true_generation_failure_is_excluded():
    r = one("", extra={"generation_failed": True})
    assert r["excluded"] is True and r["label"] is None and r["lines_non_th"] == 0


def test_length_finish_with_content_fails():
    r = one(THAI, extra={"finish_reason": "length"})
    assert r["failure_reason"] == s.REASON_LENGTH


# ---- loops ----------------------------------------------------------------------------------


def test_ten_gram_repeated_four_times_fails_three_times_passes():
    sentence = " ".join(f"คำ{chr(0x0E01 + i)}" for i in range(10))
    assert REASON_NGRAM in one("\n".join([sentence] * 4))["reasons_any_line"]
    assert REASON_NGRAM not in one("\n".join([sentence] * 3))["reasons_any_line"]


def test_tail_compression_catches_a_loop_without_repeated_words():
    r = one(THAI + "\n" + "ก" * 2000)
    assert REASON_TAIL in r["reasons_any_line"]


def test_ordinary_thai_reply_passes_both_loop_checks():
    text = thai_lines(30)
    assert one(text)["failure_reason"] is None


def test_newmm_segments_a_repeated_spaceless_thai_sentence():
    pytest.importorskip("pythainlp")
    sentence = "ฉันชอบกินข้าวผัดกุ้งที่ร้านอาหารริมแม่น้ำ"
    text = sentence * 10
    r = score_records([rec(text)])[0]
    assert REASON_NGRAM in r["reasons_any_line"] or REASON_TAIL in r["reasons_any_line"]


def test_first_failing_reason_is_the_recorded_one():
    r = one(FOREIGN_LINE + "\n" + THAI + " 如何应对", extra={"finish_reason": "length"})
    assert r["reasons_any_line"][0] == s.REASON_LENGTH and r["failure_reason"] == s.REASON_LENGTH


# ---- aggregate and review -------------------------------------------------------------------


def test_aggregate_pools_sources_and_reports_both_line_rules():
    records = score(
        [rec(THAI, "a"), rec(thai_lines(9, 1), "b"), rec(FOREIGN_LINE, "c", source="other"), rec("", "d")],
        line_rule="any_line",
    )
    result = aggregate(records)
    assert result["n_scored"] == 4 and result["score"] == 0.25
    assert result["pass_rate_by_line_rule"] == {"any_line": 0.25, "share_of_lines": 0.5}
    assert result["failures_by_reason"] == {s.REASON_THAI_RATIO: 2, s.REASON_EMPTY: 1}
    assert set(result["per_source"]) == {"th_text", "other"}
    assert result["n_replies_with_no_checked_line"] == 0
    assert "pythainlp" in result["notes"]


def test_ratio_pass_rate_ignores_replies_with_no_checked_line():
    records = score([rec(THAI, "a"), rec("ok", "b")])
    result = aggregate(records)
    assert result["n_replies_checked"] == 1 and result["ratio_pass_rate_checked"]["any_line"] == 1.0
    assert result["n_replies_with_no_checked_line"] == 1


def test_aggregate_of_nothing_scored_has_no_score():
    result = aggregate(score([rec("", generation_failed=True)]))
    assert result["score"] is None and result["n_generation_failed"] == 1


def test_review_rows_list_only_failing_replies_and_use_the_thai_column():
    records = score([rec(THAI, "ok"), rec(FOREIGN_LINE, "bad")])
    rows = s.review_rows(records)
    assert [row["id"] for row in rows] == ["bad"]
    assert list(rows[0]) == list(s.REVIEW_COLUMNS) and "lines_non_th" in rows[0] and rows[0]["verdict"] == ""
    assert rows[0]["excerpt"].startswith("[0.00] ")


def test_script_excerpt_shows_the_foreign_text():
    r = one(THAI + " ก" * 20 + " 如何应对 " + "ก" * 5)
    assert "如何应对" in r["review_excerpt"]


def test_importing_the_module_does_not_import_pythainlp():
    import subprocess
    import sys
    from pathlib import Path

    scripts = str(Path(s.__file__).parent)
    code = "import sys; sys.path.insert(0, %r); import score_th_integrity; print('pythainlp' in sys.modules)" % scripts
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# ---- CLI ------------------------------------------------------------------------------------


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def test_cli_scores_and_writes_scores_and_review(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"id": "a1", "response": THAI, "generation_failed": False}, {"id": "a2", "response": FOREIGN_LINE, "generation_failed": False}])
    out, review = tmp_path / "scored.jsonl", tmp_path / "review.csv"

    s.main(["--responses", f"th_text={path}", "--out", str(out), "--review", str(review)], tokenize=str.split)

    scored = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["source"], r["label"]) for r in scored] == [("a1", "th_text", 1), ("a2", "th_text", 0)]
    with open(review, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["id"] for row in rows] == ["a2"]
    assert '"pass_rate_by_line_rule"' in capsys.readouterr().out


def test_cli_settings_reach_the_scorer(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    write_jsonl(path, [{"id": "x", "response": "กขคงจ" + "abcdef", "generation_failed": False}])
    s.main(["--responses", str(path), "--min-thai-ratio", "0.4", "--out", str(tmp_path / "o.jsonl")], tokenize=str.split)
    assert json.loads((tmp_path / "o.jsonl").read_text(encoding="utf-8"))["label"] == 1
