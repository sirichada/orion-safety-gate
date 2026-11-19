import json

import pytest

import score_th_knowledge as s
from score_th_knowledge import aggregate, extract_letter, score_records


def rec(response, answer="C", exam="onet", rid="r1", **extra):
    return {"id": rid, "native_category": exam, "answer": answer, "response": response, "generation_failed": False, **extra}


def one(response, answer="C", **extra):
    return score_records([rec(response, answer, **extra)])[0]


@pytest.mark.parametrize(
    "reply, letter, rule",
    [
        ("C", "C", "bare"),
        ("c", "C", "bare"),
        ("**C**", "C", "bare"),
        ("(C)", "C", "bare"),
        ("C.", "C", "bare"),
        ("  \n C \n ", "C", "bare"),
        ("Answer: C", "C", "bare"),
        ("คำตอบ: C", "C", "bare"),
        ("The answer is c", "C", "marker"),
        ("Reasoning about the options.\n\nAnswer: **B**\n", "B", "marker"),
        ("Final answer: (E) the last option", "E", "marker"),
        ("Therefore the correct answer is **c**", "C", "marker"),
        ("I first thought A. but actually the answer is D.", "D", "marker"),
        ("answer is B\nthen on reflection\nAnswer: E", "E", "marker"),
        ("Some working\n**D**", "D", "last_line"),
        ("Some working\n(c)", "C", "last_line"),
        ("Some working\nC)", "C", "last_line"),
        ("Some working\nC. 12.5", "C", "last_line"),
    ],
)
def test_letter_is_extracted_by_the_first_rule_that_fires(reply, letter, rule):
    assert extract_letter(reply) == (letter, rule)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "A, B",
        "AB",
        "The answer is a result of the process",
        "I will answer a question about it",
        "คำตอบคือ ค",
        "คำตอบ: ก",
        "A. one\nB. two\nC. three\nE. four",
        "Some working with the letter C in the middle and no conclusion",
        "Maybe A or maybe B, hard to say",
    ],
)
def test_replies_no_rule_fires_on_are_unparsable(reply):
    assert extract_letter(reply) == (None, "none")


def test_marker_window_is_the_last_300_characters():
    assert extract_letter("Answer: B " + "x " * 200)[0] is None


def test_an_echoed_option_is_taken_only_when_it_is_the_only_option_line():
    assert extract_letter("work\nC. 12.5") == ("C", "last_line")
    assert extract_letter("A. 1\nB. 2\nC. 12.5") == (None, "none")


def test_correct_letter_is_label_one_and_wrong_or_unparsable_is_zero():
    assert one("C")["label"] == 1
    assert one("B")["label"] == 0
    r = one("no idea")
    assert r["label"] == 0 and r["predicted"] is None and r["parse_rule"] == "none" and r["excluded"] is False


def test_comparison_is_case_insensitive_on_the_reply_and_uppercase_on_the_gold():
    assert one("c", "C")["label"] == 1


def test_empty_reply_and_no_answer_failure_are_wrong_not_excluded():
    assert one("")["label"] == 0 and one("")["excluded"] is False
    r = one("", generation_failed=True, no_answer=True)
    assert r["excluded"] is False and r["label"] == 0


def test_true_generation_failure_is_excluded():
    r = one("", generation_failed=True)
    assert r["excluded"] is True and r["label"] is None and r["exclusion_reason"] == "generate_failed"


def test_a_cut_off_reply_with_a_marker_still_scores_the_marker():
    r = one("work... Answer: C", finish_reason="length")
    assert r["label"] == 1


def test_missing_response_or_answer_is_rejected():
    with pytest.raises(ValueError, match="response"):
        score_records([{"id": "x", "answer": "A"}])
    with pytest.raises(ValueError, match="answer"):
        score_records([{"id": "x", "response": "A"}])


def test_aggregate_pools_questions_and_reports_exams_and_their_macro_mean():
    records = score_records(
        [
            rec("C", "C", "onet", "1"), rec("A", "C", "onet", "2"), rec("C", "C", "onet", "3"), rec("C", "C", "onet", "4"),
            rec("B", "B", "ic", "5"),
            rec("x", "A", "tgat", "6"),
            rec("", "A", "tgat", "7", generation_failed=True),
        ]
    )
    result = aggregate(records)
    assert result["n_scored"] == 6 and result["n_generation_failed"] == 1
    assert result["score"] == pytest.approx(4 / 6)
    assert result["per_exam"] == {
        "onet": {"n_scored": 4, "accuracy": 0.75},
        "ic": {"n_scored": 1, "accuracy": 1.0},
        "tgat": {"n_scored": 1, "accuracy": 0.0},
    }
    assert result["macro_mean"] == pytest.approx((0.75 + 1.0 + 0.0) / 3)
    assert result["score"] != pytest.approx(result["macro_mean"])


def test_aggregate_counts_unparsable_cut_off_and_rules_separately():
    records = score_records(
        [rec("C", rid="1"), rec("Answer: C", rid="2"), rec("w\n**C**", rid="3"), rec("nothing", rid="4"),
         rec("", rid="5", finish_reason="length")]
    )
    result = aggregate(records)
    assert result["n_unparsable"] == 2 and result["n_cut_off"] == 1
    assert result["parse_rule_counts"] == {"bare": 2, "marker": 0, "last_line": 1, "none": 2}
    assert "2/5 replies unparsable" in result["notes"] and "public on Hugging Face" in result["notes"]


def test_aggregate_of_nothing_scored_has_no_score():
    result = aggregate(score_records([rec("", generation_failed=True)]))
    assert result["score"] is None and result["macro_mean"] is None and result["per_exam"] == {}


def test_cli_writes_scores_and_prints_the_aggregate(tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [rec("C", rid="a"), rec("A", rid="b")]) + "\n", encoding="utf-8")
    out = tmp_path / "o.jsonl"
    s.main(["--responses", str(path), "--out", str(out)])
    scored = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["label"] for r in scored] == [1, 0]
    assert '"macro_mean"' in capsys.readouterr().out


def test_aggregate_counts_empty_replies():
    recs = [
        {"id": "a", "native_category": "onet", "answer": "A", "response": "A", "finish_reason": "stop"},
        {"id": "b", "native_category": "onet", "answer": "B", "response": "  ", "finish_reason": "stop"},
        {"id": "c", "native_category": "onet", "answer": "C", "response": "", "finish_reason": "length"},
    ]
    result = aggregate(score_records(recs))
    assert result["n_empty"] == 2 and result["n_cut_off"] == 1 and result["n_unparsable"] == 2
    assert "2 empty" in result["notes"]


@pytest.mark.parametrize(
    "reply, letter",
    [
        ("ประโยชน์ต่อสุขภาพทั่วไป ดังนั้น ข้อ B จึงเป็นข้อที่ไม่สามารถอนุมานไดจากข้อความ", "B"),
        ("ถือเป็นการใช้ภาษาที่ไม่ถูกต้องตามหลักไวยากรณ์ในทางที่กว้างๆ\n\nดังนั้น คำตอบที่ถูกต้องที่สุดคือ ข้อ B", "B"),
        ("\n\n\\boxed{A}", "A"),
        ("\n\n\\boxed{B}", "B"),
        ("Therefore, it is the factor not used in the valuation process.\n\n<|answer|>D</|answer|>", "D"),
        ("\n\nข้อ D", "D"),
        ("ผลรวมส่งกลับมาได้ร้อยละ 32 ซึ่งใกล้เคียงกับ 30% มากที่สุดในตัวเลือก จึงตอบ E. 30", "E"),
        ("คำตอบสุดท้ายคือ ข้อ C", "C"),
        ("ตอบ A", "A"),
    ],
)
def test_boxed_tagged_and_thai_markers_are_read(reply, letter):
    assert extract_letter(reply) == (letter, "marker")


@pytest.mark.parametrize(
    "reply, letter",
    [
        ("ข้อ A ไม่ถูกต้อง เพราะเหตุผลหนึ่ง\n\\boxed{C}", "C"),
        ("\\boxed{C} แต่เมื่อพิจารณาอีกครั้ง ข้อ A", "A"),
        ("I first said A. then <|answer|>B</|answer|>", "B"),
        ("The answer is A.\nข้อ D", "D"),
    ],
)
def test_the_last_explicit_marker_wins_not_the_first_letter(reply, letter):
    assert extract_letter(reply)[0] == letter


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "\\boxed{8}",
        "**คำตอบ:** \\(\\boxed{8}\\) (ไม่ตรงกับตัวเลือกที่ให้มา)",
        "\\boxed{a}",
        "ข้อ a",
        "ตอบ ก",
        "ข้อความนี้ตอบคำถามเกี่ยวกับ A และ B",
        "จงหาเดิมนครชมีเงินน้อยกว่าคิมหันท์กี่บาท\nA. 295\nB. 285\nC. 250\nD. 210\nE. 195\nAnswer:",
    ],
)
def test_boxed_tagged_and_thai_forms_do_not_fire_on_ordinary_text_or_a_non_letter(reply):
    assert extract_letter(reply) == (None, "none")
