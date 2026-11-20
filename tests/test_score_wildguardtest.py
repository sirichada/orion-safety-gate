import json

import pytest

import score_wildguardtest as wg


def row(i, harm, prompt_harm="harmful", adversarial=False, sub="cyberattack", reply="some reply"):
    return {
        "prompt": f"prompt {i}",
        "response": reply,
        "adversarial": adversarial,
        "prompt_harm_label": prompt_harm,
        "response_refusal_label": "compliance",
        "response_harm_label": harm,
        "subcategory": sub,
    }


def test_rows_without_a_harm_label_or_reply_are_left_out_and_fields_are_mapped():
    rows = [row(0, "harmful"), row(1, None), row(2, "unharmful", reply=""), row(3, "unharmful", adversarial=True)]
    records = wg.to_records(rows)
    assert [r["id"] for r in records] == ["wildguardtest_0", "wildguardtest_3"]
    first = records[0]
    assert first["functional_category"] == "standard" and first["context"] == "" and first["prompt"] == "prompt 0"
    assert records[1]["adversarial"] is True


def test_agreement_counts_kappa_and_the_perfect_and_empty_cases():
    perfect = wg.agreement([(1, 1), (1, 1), (0, 0), (0, 0)])
    assert perfect["agreement"] == 1.0 and perfect["kappa"] == pytest.approx(1.0) and perfect["f1"] == 1.0
    mixed = wg.agreement([(1, 1), (1, 0), (0, 1), (0, 0)])
    assert (mixed["both_harmful"], mixed["reference_harmful_only"], mixed["classifier_yes_only"], mixed["both_not"]) == (1, 1, 1, 1)
    assert mixed["agreement"] == 0.5 and mixed["kappa"] == pytest.approx(0.0)
    empty = wg.agreement([])
    assert empty["n"] == 0 and empty["agreement"] is None and empty["kappa"] is None
    assert wg.agreement([(0, 0), (0, 0)])["precision"] is None


def scored(records, labels):
    out = []
    for r, label in zip(records, labels):
        r = dict(r, label=label, excluded=label == -1)
        out.append(r)
    return out


def test_summary_splits_by_prompt_harm_adversarial_and_subcategory_and_skips_ambiguous():
    rows = [
        row(0, "harmful", "harmful", False, "cyberattack"),
        row(1, "harmful", "harmful", True, "cyberattack"),
        row(2, "unharmful", "unharmful", False, "benign"),
        row(3, "unharmful", "unharmful", True, "benign"),
        row(4, "harmful", "harmful", False, "sexual_content"),
    ]
    summary = wg.summarize(scored(wg.to_records(rows), [1, 0, 0, 1, -1]))
    assert summary["n_rows"] == 5 and summary["n_unscored"] == 1 and summary["overall"]["n"] == 4
    assert summary["overall"]["agreement"] == 0.5
    assert summary["by_prompt_harm"]["harmful"]["n"] == 2 and summary["by_prompt_harm"]["unharmful"]["agreement"] == 0.5
    assert summary["by_adversarial"]["True"]["n"] == 2
    assert list(summary["by_subcategory"])[0] == "cyberattack" and "sexual_content" not in summary["by_subcategory"]
    assert "Definitions differ" in summary["note"]


def test_run_writes_the_scored_rows_and_the_summary_with_the_definition_note(tmp_path):
    records = wg.to_records([row(0, "harmful"), row(1, "unharmful", "unharmful", sub="benign")])
    out = tmp_path / "out"
    summary = wg.run(records, lambda recs: scored(recs, [1, 0]), out)
    assert summary["overall"]["agreement"] == 1.0
    lines = (out / "scored_wildguardtest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["label"] for l in lines] == [1, 0]
    assert "Definitions differ" in (out / "summary.txt").read_text(encoding="utf-8")
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["n_rows"] == 2


def test_wilson_interval_brackets_the_proportion():
    lo, hi = wg.wilson(50, 100)
    assert lo < 0.5 < hi and wg.wilson(0, 0) is None
