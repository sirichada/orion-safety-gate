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


def test_model_is_the_only_parser_required_argument():
    with pytest.raises(SystemExit):
        parse("--model-id", "x")
    assert parse("--model", "m").model_id is None  # enforced later by resolve_inputs


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
    verdict = safety_suite.main(argv, run_stage=fake_stages, interactive=False)

    written = json.loads(next((sandbox / "results").glob("test_model_*/verdict.json")).read_text(encoding="utf-8"))
    assert written == verdict
    assert (verdict["safety_gate"], verdict["smartness_gate"], verdict["admitted"]) == ("pass", "not_testable", False)
    assert verdict["gate_incomplete"] == {"smartness": ["coding_capability", "capability_honesty"]}
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
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run after a preflight failure"), interactive=False)
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
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)
    row = (sandbox / "_index.md").read_text(encoding="utf-8").strip()
    assert "FileNotFoundError" in row and "no verdict" in row
    assert row.count("|") == 7  # the "|" in the model id is escaped, so the table keeps 6 cells


def write_thresholds(sandbox):
    path = sandbox / "t.json"
    path.write_text(json.dumps({"provisional": True, "cases": {"th_text": {"threshold": 0.2, "min_scored": 1}}}))
    return path


class FakeAsk:
    def __init__(self, confirm_answer=True, confirm_raises=None):
        self.confirm_answer = confirm_answer
        self.confirm_raises = confirm_raises
        self.confirm_calls = 0

    def confirm(self, message, default=True):
        self.confirm_calls += 1
        if self.confirm_raises:
            raise self.confirm_raises
        return self.confirm_answer


def run_main(sandbox, *extra, ask=None, interactive=False):
    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", "th",
            "--thresholds", str(write_thresholds(sandbox)), *extra]
    return safety_suite.main(argv, run_stage=fake_stages, interactive=interactive, ask_factory=lambda: ask)


def test_flags_only_end_to_end(sandbox):
    verdict = run_main(sandbox)
    assert verdict["safety_gate"] == "pass"
    assert verdict["run_args"]["capabilities_source"] == "flags"


def test_summary_lists_tests_untested_items_and_provisional_note(sandbox, capsys):
    run_main(sandbox)
    out = capsys.readouterr().out
    assert "th_text (4 prompts)" in out and "total        4 prompts" in out
    assert "coding_capability" in out and "not implemented in this version" in out
    assert "provisional" in out


@pytest.mark.parametrize("answer", [False, None])  # None is questionary's result on Ctrl-C/Esc
def test_declined_confirmation_runs_nothing(sandbox, answer):
    ask = FakeAsk(confirm_answer=answer)
    assert run_main(sandbox, ask=ask, interactive=True) is None
    assert ask.confirm_calls == 1
    assert not (sandbox / "results").exists()
    row = (sandbox / "_index.md").read_text(encoding="utf-8")
    assert "cancelled before running" in row and "runs/logs" not in row


def test_yes_skips_confirmation_but_still_prints_summary(sandbox, capsys):
    ask = FakeAsk()
    run_main(sandbox, "--yes", ask=ask, interactive=True)
    assert ask.confirm_calls == 0
    assert "About to run:" in capsys.readouterr().out


def test_missing_model_id_row_says_not_given(sandbox):
    argv = ["run", "--model", "m", "--modalities", "text", "--languages", "th"]
    with pytest.raises(SystemExit, match="no interactive terminal"):
        safety_suite.main(argv, run_stage=fake_stages, interactive=False)
    assert "model-id=(not given)" in (sandbox / "_index.md").read_text(encoding="utf-8")


def test_confirm_that_cannot_render_says_pass_yes(sandbox):
    ask = FakeAsk(confirm_raises=OSError("NoConsoleScreenBufferError"))
    with pytest.raises(SystemExit, match="pass --yes .*OSError"):
        run_main(sandbox, ask=ask, interactive=True)
    assert not (sandbox / "results").exists()
    assert "suite error" in (sandbox / "_index.md").read_text(encoding="utf-8")


def test_prompt_backend_unavailable_at_confirm_says_pass_yes(sandbox):
    def broken():
        raise ImportError("No module named 'questionary'")

    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", "th",
            "--thresholds", str(write_thresholds(sandbox))]
    with pytest.raises(SystemExit, match="pass --yes .*ImportError"):
        safety_suite.main(argv, run_stage=fake_stages, interactive=True, ask_factory=broken)
    assert not (sandbox / "results").exists()


def test_no_input_overrides_a_real_terminal(sandbox, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", "th",
            "--thresholds", str(write_thresholds(sandbox)), "--no-input"]
    verdict = safety_suite.main(argv, run_stage=fake_stages, ask_factory=lambda: pytest.fail("must not prompt"))
    assert verdict["safety_gate"] == "pass"


def test_real_terminal_without_no_input_prompts(sandbox, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    ask = FakeAsk(confirm_answer=False)
    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", "th",
            "--thresholds", str(write_thresholds(sandbox))]
    assert safety_suite.main(argv, run_stage=fake_stages, ask_factory=lambda: ask) is None
    assert ask.confirm_calls == 1
