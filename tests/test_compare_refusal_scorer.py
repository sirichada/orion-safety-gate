import csv
import json

import pytest

import compare_refusal_scorer as crs


def write_run(folder, name, replies):
    folder.mkdir(parents=True)
    (folder / "verdict.json").write_text(json.dumps({"model_id": name}), encoding="utf-8")
    lines = []
    for i, (reply, label) in enumerate(replies):
        lines.append(json.dumps({"id": f"xstest_{i}", "prompt": f"prompt {i}", "response": reply, "label": label, "excluded": False, "native_category": "homonyms"}))
    (folder / "scored_over_refusal.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_wildguard_reads_the_three_answers_and_tolerates_junk():
    text = "Harmful request: no\nResponse refusal: yes\nHarmful response: n/a"
    assert crs.parse_wildguard(text) == {"harmful_request": 0, "refusal": 1, "harmful_response": None}
    assert crs.parse_wildguard("") == {"harmful_request": None, "refusal": None, "harmful_response": None}
    assert crs.parse_wildguard("Response refusal: maybe")["refusal"] is None


def test_partial_candidate_finds_refusal_wording_anywhere_including_curly_apostrophes():
    assert crs.partial_candidate("Here is how. However I can’t help with the last part.")
    assert not crs.partial_candidate("Here is how to kill a Python process.")


def test_compare_rows_counts_each_cell_and_ignores_unparsed():
    rows = [
        {"prefix": 1, "judge": 1}, {"prefix": 1, "judge": 0}, {"prefix": 0, "judge": 1}, {"prefix": 0, "judge": 0},
        {"prefix": 0, "judge": 0, "partial_candidate": False}, {"prefix": 0, "judge": None},
    ]
    stats = crs.compare_rows(rows)
    assert (stats["both_refused"], stats["prefix_only"], stats["judge_only"], stats["neither"]) == (1, 1, 1, 2)
    assert stats["n"] == 6 and stats["n_unparsed"] == 1 and stats["agreement"] == pytest.approx(0.6)


def test_run_writes_summary_csv_and_judged_rows_with_one_judge_call(tmp_path):
    write_run(tmp_path / "a", "Model A", [("I cannot help.", 1), ("Sure, here you go. I can't do the rest.", 0), ("Fine.", 0)])
    write_run(tmp_path / "b", "Model B", [("Sorry, no.", 1), ("Yes.", 0)])
    calls = []

    def judge(items):
        calls.append(len(items))
        answers = [1, 0, 0, 0, 1]
        return [{"refusal": a, "raw": f"Response refusal: {'yes' if a else 'no'}"} for a in answers[: len(items)]]

    human_rows = [
        {"response": "I cannot help", "response_refusal_label": "refusal"},
        {"response": "Sure thing", "response_refusal_label": "compliance"},
        {"response": "Here you go", "response_refusal_label": "refusal"},
        {"response": "x", "response_refusal_label": None},
    ]
    out = tmp_path / "out"
    summary = crs.run([tmp_path / "a", tmp_path / "b"], judge, human_rows, out)
    assert calls == [5]
    assert summary["models"]["Model A"]["partial_candidates"] == 1 and summary["models"]["Model B"]["judge_only"] == 1
    h = summary["wildguardtest_prefix_vs_human"]
    assert h["n"] == 3 and h["human_only"] == 1 and h["both_refused"] == 1 and h["neither"] == 1
    rows = list(csv.DictReader(open(out / "disagreements.csv", encoding="utf-8")))
    assert {r["kind"] for r in rows} == {"partial_candidate", "judge_refused_prefix_missed", "prefix_refused_judge_complied"}
    assert all(r["hand_label"] == "" for r in rows)
    assert len((out / "judged_over_refusal.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert "no partial class" in (out / "summary.txt").read_text(encoding="utf-8")


def test_a_missing_scored_file_is_a_clear_error(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit, match="scored_over_refusal.jsonl"):
        crs.read_scored(tmp_path / "x")
