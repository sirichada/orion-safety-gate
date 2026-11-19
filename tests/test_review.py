import json

import pytest

from orchestrator import SuiteError
from report_helpers import make_verdict
from review import apply_review, review_problems


def run_folder(tmp_path, scores=None, **kwargs):
    folder = tmp_path / "run"
    verdict = make_verdict(folder, {"th_text": 0.201, **(scores or {})}, **kwargs)
    (folder / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False), encoding="utf-8")
    return folder, verdict


def review_for(verdict, **changes):
    case = next(c for c in verdict["case_results"] if c["case_id"] == "th_text")
    flagged = [s["id"] for s in case["sample_generations"] if s["outcome"] == "flagged"]
    review = {
        "case_id": "th_text",
        "reviewer": "A Reviewer",
        "date": "2026-11-25",
        "n_reviewed": len(flagged),
        "n_judged_harmful": 3,
        "decision": "accept",
        "note": "mostly refusals",
        "judgements": [{"id": fid, "harmful": i < 3} for i, fid in enumerate(flagged)],
    }
    review.update(changes)
    return review


def write_review(tmp_path, review, name="review.json"):
    path = tmp_path / name
    path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    return path


def test_an_accepted_review_writes_a_new_verdict_and_leaves_the_original_alone(tmp_path):
    folder, verdict = run_folder(tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    reviewed, out = apply_review(folder, write_review(tmp_path, review_for(verdict)), stamp="20261125-090000")
    assert out == folder.resolve() / "verdict.reviewed-20261125-090000.json" and json.loads(out.read_text(encoding="utf-8")) == reviewed
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.name in before} == before
    assert reviewed["admitted"] is True and reviewed["verdict_state"] == "admitted_after_review"
    assert reviewed["safety_gate"] == "pass" and reviewed["smartness_gate"] == "pass"
    case = next(c for c in reviewed["case_results"] if c["case_id"] == "th_text")
    assert case["status"] == "pass_after_review" and case["score"] == pytest.approx(0.201) and case["review"]["decision"] == "accept"
    assert case["review"]["reviewer"] == "A Reviewer" and case["review"]["n_judged_harmful"] == 3
    assert reviewed["review_of"]["source_verdict"] == "verdict.json"


def test_a_rejected_review_fails_the_gate_and_the_verdict(tmp_path):
    folder, verdict = run_folder(tmp_path)
    reviewed, _ = apply_review(folder, write_review(tmp_path, review_for(verdict, decision="reject")), stamp="s")
    assert reviewed["safety_gate"] == "fail" and reviewed["admitted"] is False and reviewed["verdict_state"] == "not_admitted"
    assert next(c for c in reviewed["case_results"] if c["case_id"] == "th_text")["status"] == "fail"


def test_a_review_does_not_hide_another_failing_case(tmp_path):
    folder, verdict = run_folder(tmp_path, {"th_integrity": 0.5})
    reviewed, _ = apply_review(folder, write_review(tmp_path, review_for(verdict)), stamp="s")
    assert reviewed["admitted"] is False and reviewed["smartness_gate"] == "fail" and reviewed["verdict_state"] == "not_admitted"


def test_a_review_leaves_a_run_with_missing_data_undecided(tmp_path):
    folder, verdict = run_folder(tmp_path, excluded={"th_knowledge": 60})
    reviewed, _ = apply_review(folder, write_review(tmp_path, review_for(verdict)), stamp="s")
    assert reviewed["admitted"] is False and reviewed["verdict_state"] == "not_decided" and reviewed["safety_gate"] == "pass"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"decision": "maybe"}, "decision must be one of"),
        ({"n_judged_harmful": 31, "n_reviewed": 30}, "n_judged_harmful .31. must be between 0 and n_reviewed .30."),
        ({"n_judged_harmful": 5}, "n_judged_harmful is 5 but the judgements mark 3 harmful"),
        ({"n_reviewed": 29}, "n_reviewed is 29 but there are 30 judgements"),
        ({"reviewer": " "}, "reviewer must be a name"),
        ({"date": "25 Nov"}, "date must be YYYY-MM-DD"),
        ({"case_id": "en_text"}, "is 'pass' in this verdict, not needs_review"),
        ({"case_id": "nonsense"}, "has no case 'nonsense'"),
        ({"judgements": "x"}, "judgements must be a list"),
    ],
)
def test_an_invalid_review_is_refused_and_writes_nothing(tmp_path, changes, message):
    folder, verdict = run_folder(tmp_path)
    before = sorted(p.name for p in folder.iterdir())
    with pytest.raises(SuiteError, match=message):
        apply_review(folder, write_review(tmp_path, review_for(verdict, **changes)), stamp="s")
    assert sorted(p.name for p in folder.iterdir()) == before


def test_a_review_for_another_run_is_refused(tmp_path):
    folder, verdict = run_folder(tmp_path)
    other = review_for(verdict)
    other["judgements"] = [{"id": f"elsewhere_{i}", "harmful": False} for i in range(30)]
    other["n_judged_harmful"] = 0
    with pytest.raises(SuiteError, match="must cover exactly the 30 flagged replies"):
        apply_review(folder, write_review(tmp_path, other), stamp="s")


def test_a_review_that_skips_a_reply_or_repeats_one_is_refused(tmp_path):
    folder, verdict = run_folder(tmp_path)
    review = review_for(verdict)
    short = {**review, "judgements": review["judgements"][:-1], "n_reviewed": 29}
    with pytest.raises(SuiteError, match="must cover exactly"):
        apply_review(folder, write_review(tmp_path, short), stamp="s")
    twice = {**review, "judgements": review["judgements"] + review["judgements"][:1], "n_reviewed": 31}
    with pytest.raises(SuiteError, match="judged more than once"):
        apply_review(folder, write_review(tmp_path, twice), stamp="s")


def test_a_case_that_does_not_need_review_is_refused(tmp_path):
    folder, _ = run_folder(tmp_path, {"th_text": 0.134})
    review = review_for(make_verdict(tmp_path / "other", {"th_text": 0.201}))
    with pytest.raises(SuiteError, match="is 'pass' in this verdict, not needs_review"):
        apply_review(folder, write_review(tmp_path, review), stamp="s")


def test_the_same_review_applied_twice_is_refused(tmp_path):
    folder, verdict = run_folder(tmp_path)
    path = write_review(tmp_path, review_for(verdict))
    apply_review(folder, path, stamp="20261125-090000")
    with pytest.raises(SuiteError, match="already applies a review of th_text"):
        apply_review(folder, path, stamp="20261125-090001")
    assert len(list(folder.glob("verdict.reviewed-*.json"))) == 1


def test_a_reviewed_verdict_cannot_be_reviewed_again(tmp_path):
    folder, verdict = run_folder(tmp_path)
    _, out = apply_review(folder, write_review(tmp_path, review_for(verdict)), stamp="20261125-090000")
    with pytest.raises(SuiteError, match="not needs_review"):
        apply_review(folder, write_review(tmp_path, review_for(verdict, reviewer="Another")), verdict_name=out.name, stamp="20261125-090001")


def test_an_existing_output_file_is_never_replaced(tmp_path):
    folder, verdict = run_folder(tmp_path)
    existing = folder / "verdict.reviewed-s.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(SuiteError, match="already exists"):
        apply_review(folder, write_review(tmp_path, review_for(verdict)), stamp="s")
    assert existing.read_text(encoding="utf-8") == "keep"


def test_a_verdict_without_sampled_replies_asks_for_a_re_score(tmp_path):
    folder, _ = run_folder(tmp_path, with_samples=False)
    review = review_for(make_verdict(tmp_path / "other", {"th_text": 0.201}))
    with pytest.raises(SuiteError, match="re-score it first"):
        apply_review(folder, write_review(tmp_path, review), stamp="s")


def test_unreadable_inputs_give_clear_errors(tmp_path):
    folder, verdict = run_folder(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(SuiteError, match="cannot be read as JSON"):
        apply_review(folder, bad, stamp="s")
    with pytest.raises(SuiteError, match="not found"):
        apply_review(tmp_path / "missing", write_review(tmp_path, {}), stamp="s")
    assert review_problems(verdict, []) == ["review.json must hold one JSON object"]
    assert "missing field 'decision'" in review_problems(verdict, {"case_id": "th_text"})


def test_the_cli_writes_the_reviewed_verdict_and_its_report(tmp_path):
    import safety_suite

    folder, verdict = run_folder(tmp_path)
    path = write_review(tmp_path, review_for(verdict))
    args = safety_suite.build_parser().parse_args(["review", str(folder), "--review", str(path)])
    result = safety_suite.review_command(args)
    assert result["verdict_state"] == "admitted_after_review"
    assert len(list(folder.glob("verdict.reviewed-*.json"))) == 1 and len(list(folder.glob("report.reviewed-*.html"))) == 1
    assert "Admitted after review" in next(folder.glob("report.reviewed-*.html")).read_text(encoding="utf-8")


def test_the_cli_report_command_writes_and_refuses_to_overwrite(tmp_path):
    import safety_suite

    folder, _ = run_folder(tmp_path)
    args = safety_suite.build_parser().parse_args(["report", str(folder)])
    assert safety_suite.report_command(args) == folder / "report.html"
    with pytest.raises(SystemExit, match="never overwritten"):
        safety_suite.report_command(args)


def test_main_dispatches_report_and_review(tmp_path):
    import safety_suite

    folder, verdict = run_folder(tmp_path)
    assert safety_suite.main(["report", str(folder)]) == folder / "report.html"
    result = safety_suite.main(["review", str(folder), "--review", str(write_review(tmp_path, review_for(verdict)))])
    assert result["verdict_state"] == "admitted_after_review"
