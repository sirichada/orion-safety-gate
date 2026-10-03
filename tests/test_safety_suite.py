import json
from pathlib import Path

import pytest

import orchestrator
import safety_suite
from orchestrator import CaseRunner


def parse(*argv):
    return safety_suite.parse_args(["run", *argv])


BASE = ["--model", "m", "--model-id", "Mango 1.5-9B", "--capabilities", "c.json", "--thresholds", "t.json"]


def test_wrapper_args_forward_given_flags_and_always_n():
    args = parse(*BASE, "--backend", "api", "--batch-size", "4", "--max-consecutive-failures", "5")
    assert safety_suite.wrapper_args(args) == [
        "--model", "m", "--backend", "api", "--batch-size", "4", "--max-consecutive-failures", "5"
    ]
    assert safety_suite.wrapper_args(parse(*BASE)) == ["--model", "m", "--backend", "hf", "--max-consecutive-failures", "3"]


def test_local_paths_are_resolved_but_ids_pass_through(tmp_path, monkeypatch):
    (tmp_path / "adapter").mkdir()
    monkeypatch.chdir(tmp_path)
    args = parse(*BASE, "--adapter", "adapter")
    forwarded = safety_suite.wrapper_args(args)
    assert forwarded[forwarded.index("--adapter") + 1] == str((tmp_path / "adapter").resolve())
    assert forwarded[forwarded.index("--model") + 1] == "m"  # not a path: HF/API id unchanged


def test_required_arguments():
    with pytest.raises(SystemExit):
        parse("--model", "m")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Points every output location at tmp_path and gives th_text a 4-prompt file."""
    monkeypatch.setattr(safety_suite, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(safety_suite, "INDEX", tmp_path / "_index.md")
    monkeypatch.setattr(safety_suite, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "LOGS", tmp_path / "logs")
    prompts = tmp_path / "th_text.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"thaisafetybench_{i}"}) + "\n" for i in range(4)), encoding="utf-8")
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_text", CaseRunner(prompts, "score_th_text", "classifier"))
    caps = tmp_path / "c.json"
    caps.write_text(json.dumps({"declared_modalities": ["text"], "declared_languages": ["th"]}), encoding="utf-8")
    return tmp_path


def fake_stages(cmd, log_path):
    out = Path(cmd[cmd.index("--out") + 1])
    if cmd[1] == "model_wrapper.py":
        out.write_text("{}\n", encoding="utf-8")
    else:
        record = {"label": 0, "excluded": False, "exclusion_reason": None, "scoring_method": "classifier", "truncated": False}
        out.write_text((json.dumps(record) + "\n") * 4, encoding="utf-8")
    return 0


def test_end_to_end_run_writes_verdict_and_index_row(sandbox):
    thresholds = sandbox / "t.json"
    thresholds.write_text(json.dumps({"provisional": True, "cases": {"th_text": {"threshold": 0.2, "min_scored": 1}}}))
    argv = ["run", "--model", "m", "--model-id", "test model", "--capabilities", str(sandbox / "c.json"),
            "--thresholds", str(thresholds)]
    verdict = safety_suite.main(argv, run_stage=fake_stages)

    written = json.loads(next((sandbox / "results").glob("test_model_*/verdict.json")).read_text(encoding="utf-8"))
    assert written == verdict
    assert (verdict["safety_gate"], verdict["smartness_gate"], verdict["admitted"]) == ("pass", "not_testable", False)
    assert verdict["thresholds_provisional"] is True
    assert verdict["run_args"]["max_consecutive_failures"] == 3
    assert len(verdict["run_args"]["thresholds_sha256"]) == 64
    index = (sandbox / "_index.md").read_text(encoding="utf-8")
    assert "admitted=False" in index and "(provisional thresholds)" in index


def test_suite_error_still_gets_an_index_row_and_no_run(sandbox):
    thresholds = sandbox / "t.json"
    thresholds.write_text(json.dumps({"provisional": True, "cases": {}}))
    argv = ["run", "--model", "m", "--model-id", "x", "--capabilities", str(sandbox / "c.json"),
            "--thresholds", str(thresholds)]
    with pytest.raises(SystemExit, match="Suite error"):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run after a preflight failure"))
    index = (sandbox / "_index.md").read_text(encoding="utf-8")
    assert "suite error: Preflight failed" in index and "no verdict" in index
    assert not (sandbox / "results").exists()


def test_max_consecutive_failures_must_be_positive():
    with pytest.raises(SystemExit):
        parse(*BASE, "--max-consecutive-failures", "0")


def test_bad_input_file_still_gets_an_index_row(sandbox):
    argv = ["run", "--model", "m", "--model-id", "a|b", "--capabilities", str(sandbox / "missing.json"),
            "--thresholds", str(sandbox / "missing.json")]
    with pytest.raises(FileNotFoundError):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"))
    row = (sandbox / "_index.md").read_text(encoding="utf-8").strip()
    assert "FileNotFoundError" in row and "no verdict" in row
    assert row.count("|") == 7  # the "|" in the model id is escaped, so the table keeps 6 cells
