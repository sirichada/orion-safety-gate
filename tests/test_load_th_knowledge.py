import json

import pytest

import load_th_knowledge as ld


def write_exam(src, exam, train, test):
    for split, rows in (("train", train), ("test", test)):
        (src / f"{exam}_{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def item(q, answer="b", **options):
    options = options or {"a": "ตัวเลือกหนึ่ง", "b": "ตัวเลือกสอง", "c": "ตัวเลือกสาม"}
    return {"question": q, "answer": answer, "subject": "science", **options}


@pytest.fixture
def src(tmp_path):
    for exam in ld.EXAMS:
        write_exam(tmp_path, exam, [item(f"shot {exam} {i}", "a") for i in range(5)], [item(f"{exam} test {i}") for i in range(3)])
    return tmp_path


def test_every_exam_contributes_its_test_rows_in_order(src):
    records = list(ld.load(src))
    assert len(records) == 15
    assert [r["id"] for r in records[:3]] == ["thai_exam_onet_0", "thai_exam_onet_1", "thai_exam_onet_2"]
    assert [r["native_id"] for r in records[-2:]] == ["a_level_1", "a_level_2"]


def test_record_follows_the_prompt_set_schema(src):
    record = next(iter(ld.load(src)))
    assert set(record) == {"id", "native_id", "prompt", "language", "source_benchmark", "native_category",
                           "functional_category", "context", "model_input", "answer"}
    assert (record["language"], record["source_benchmark"], record["native_category"]) == ("th", "thai_exam", "onet")
    assert record["functional_category"] == "science" and record["context"] is None
    assert record["prompt"] == "onet test 0" and record["answer"] == "B"


def test_the_five_shots_are_the_exams_own_and_identical_for_every_question(src):
    onet = [r for r in ld.load(src) if r["native_category"] == "onet"]
    ic = [r for r in ld.load(src) if r["native_category"] == "ic"]
    shots = lambda r: r["model_input"].split("\n\n")[1:6]
    assert shots(onet[0]) == shots(onet[1]) == shots(onet[2])
    assert all("shot onet" in s for s in shots(onet[0])) and all("shot ic" in s for s in shots(ic[0]))
    assert all(s.endswith("Answer: A") for s in shots(onet[0]))


def test_prompt_layout_is_instruction_then_shots_then_an_open_question(src):
    text = next(iter(ld.load(src)))["model_input"]
    blocks = text.split("\n\n")
    assert blocks[0] == ld.INSTRUCTION and len(blocks) == 7
    assert blocks[-1] == "Question: onet test 0\nA. ตัวเลือกหนึ่ง\nB. ตัวเลือกสอง\nC. ตัวเลือกสาม\nAnswer:"


def test_the_answer_is_not_in_the_open_question_or_the_prompt_field(src):
    record = next(iter(ld.load(src)))
    assert not record["model_input"].rstrip().endswith("B") and "Answer: B" not in record["model_input"].split("\n\n")[-1]


def test_empty_options_are_left_out_and_letters_are_not_renumbered():
    row = item("q", "d", a="one", b="", c="three", d="four", e="")
    assert ld.render_question(row, False) == "Question: q\nA. one\nC. three\nD. four\nAnswer:"
    assert ld.render_question(row, True).endswith("Answer: D")


def test_windows_line_endings_and_edge_whitespace_are_normalised():
    row = item("  line one\r\nline two \r\n", "a", a=" x\r\ny ", b="z")
    assert ld.render_question(row, False) == "Question: line one\nline two\nA. x\ny\nB. z\nAnswer:"


def test_a_blank_line_in_a_question_cannot_hide_inside_the_block_split(src):
    write_exam(src, "onet", [item("shot", "a")], [item("first\n\nsecond")])
    record = next(iter(ld.load(src)))
    assert "Question: first\n\nsecond" in record["model_input"]


REAL = ld.SRC_DIR / "onet_test.jsonl"


@pytest.mark.skipif(not REAL.exists(), reason="ThaiExam not fetched")
def test_pinned_files_give_565_questions_across_the_five_exams():
    records = list(ld.load())
    counts = {exam: sum(1 for r in records if r["native_category"] == exam) for exam in ld.EXAMS}
    assert counts == {"onet": 162, "ic": 95, "tgat": 65, "tpat1": 116, "a_level": 127}
    assert len({r["id"] for r in records}) == 565
    assert all(r["answer"] in "ABCDE" and f"\n{r['answer']}. " in r["model_input"].split("\n\n")[-1] for r in records)
