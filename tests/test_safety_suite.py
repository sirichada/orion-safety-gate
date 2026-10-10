import json
from pathlib import Path

import pytest

import orchestrator
import run_config
import safety_suite
from conftest import FakeHub, stage_report_for
from orchestrator import CaseRunner


def parse(*argv):
    return safety_suite.parse_args(["run", *argv])


BASE = ["--model", "m", "--model-id", "Mango 1.5-9B", "--capabilities", "c.json", "--thresholds", "t.json"]


def test_wrapper_args_always_forward_every_setting():
    api = safety_suite.wrapper_args(parse(*BASE, "--backend", "api", "--batch-size", "4", "--max-consecutive-failures", "5"))
    assert api == [
        "--model", "m", "--backend", "api", "--batch-size", "4", "--max-consecutive-failures", "5",
        "--temperature", "0.0", "--top-p", "1.0", "--max-new-tokens", "256",
    ]
    hf = safety_suite.wrapper_args(parse(*BASE, "--precision", "full"), {"model": "a" * 40, "adapter": None})
    assert hf == [
        "--model", "m", "--backend", "hf", "--batch-size", "8", "--max-consecutive-failures", "3",
        "--temperature", "0.0", "--top-p", "1.0", "--max-new-tokens", "256",
        "--precision", "full", "--max-input-tokens", "2048", "--model-revision", "a" * 40,
    ]


def test_precision_choices_and_default():
    assert parse(*BASE).precision == "4bit"
    for choice in ("4bit", "full", "as_published"):
        assert parse(*BASE, "--precision", choice).precision == choice
    with pytest.raises(SystemExit):
        parse(*BASE, "--precision", "fp16")


@pytest.mark.parametrize(
    "extra",
    [
        ["--backend", "api", "--precision", "full"],
        ["--backend", "api", "--model-revision", "main"],
        ["--adapter-revision", "main"],
        ["--batch-size", "0"],
    ],
)
def test_run_configuration_flag_conflicts_are_rejected(extra):
    with pytest.raises(SystemExit):
        parse(*BASE, *extra)


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
    monkeypatch.setattr(run_config, "HUB", FakeHub())
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
        run_config.write_stage_report(out, stage_report_for(cmd))
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
    assert verdict["run_args"]["batch_size"] == 8  # the wrapper's effective default, not null
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


def test_only_parses_comma_separated_ids_and_dedupes():
    assert parse(*BASE, "--only", "over_refusal, en_text,en_text").only == ("over_refusal", "en_text")
    assert parse(*BASE).only is None


@pytest.mark.parametrize("value", ["", "en_text,", "a,,b", "nope", "en_text,nope"])
def test_only_rejects_empty_tokens_and_unknown_ids(value):
    with pytest.raises(SystemExit):
        parse(*BASE, "--only", value)


def en_th_sandbox(sandbox, monkeypatch):
    # en_text borrows th_text's prompt file and scorer module; only the wiring is under test.
    prompts = sandbox / "th_text.jsonl"
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "en_text", CaseRunner(prompts, "score_th_text", "classifier"))
    path = sandbox / "t.json"
    cases = {c: {"threshold": 0.2, "min_scored": 1} for c in ("en_text", "th_text")}
    path.write_text(json.dumps({"provisional": True, "cases": cases}))
    return path


def run_partial(sandbox, monkeypatch, only, languages=("en", "th")):
    thresholds = en_th_sandbox(sandbox, monkeypatch)
    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", *languages,
            "--thresholds", str(thresholds)]
    if only is not None:
        argv += ["--only", only]
    return safety_suite.main(argv, run_stage=fake_stages, interactive=False)


def test_partial_run_labels_folder_verdict_and_index(sandbox, monkeypatch):
    verdict = run_partial(sandbox, monkeypatch, "th_text")
    assert verdict["run_scope"] == "partial"
    assert verdict["cases_run"] == ["th_text"]
    assert verdict["run_args"]["only"] == ["th_text"]
    assert verdict["admitted"] is False
    assert verdict["gate_incomplete"]["safety"] == ["en_text"]
    assert [p.name for p in (sandbox / "results").iterdir() if p.is_dir()][0].endswith("_partial-th_text")
    index = (sandbox / "_index.md").read_text(encoding="utf-8")
    assert "scope=partial" in index


def test_only_naming_every_applicable_built_case_is_a_full_run_without_suffix(sandbox, monkeypatch):
    # For a Thai-only declaration th_text is the only applicable built case.
    verdict = run_partial(sandbox, monkeypatch, "th_text", languages=("th",))
    assert verdict["run_scope"] == "full"
    assert verdict["run_args"]["only"] == ["th_text"]
    names = [p.name for p in (sandbox / "results").iterdir() if p.is_dir()]
    assert not any("_partial-" in n for n in names)


def test_full_run_records_only_as_null(sandbox, monkeypatch):
    verdict = run_partial(sandbox, monkeypatch, None, languages=("th",))
    assert verdict["run_scope"] == "full"
    assert verdict["cases_run"] == ["th_text"]
    assert verdict["run_args"]["only"] is None


def test_only_case_not_applicable_is_a_suite_error_with_an_index_row(sandbox, monkeypatch):
    with pytest.raises(SystemExit, match="does not apply"):
        run_partial(sandbox, monkeypatch, "en_text", languages=("th",))
    assert not (sandbox / "results").exists()
    assert "suite error" in (sandbox / "_index.md").read_text(encoding="utf-8")


def test_confirmation_summary_states_a_partial_run(sandbox, monkeypatch, capsys):
    ask = FakeAsk(confirm_answer=False)
    thresholds = en_th_sandbox(sandbox, monkeypatch)
    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", "en", "th",
            "--thresholds", str(thresholds), "--only", "th_text"]
    safety_suite.main(argv, run_stage=fake_stages, interactive=True, ask_factory=lambda: ask)
    out = capsys.readouterr().out
    assert "Partial run: th_text only. Not an admission verdict." in out
    assert "left out by --only" in out


def test_summary_has_no_partial_banner_without_only(sandbox, monkeypatch, capsys):
    ask = FakeAsk(confirm_answer=False)
    run_main(sandbox, ask=ask, interactive=True)
    assert "Partial run" not in capsys.readouterr().out


def test_no_partial_banner_when_only_names_every_applicable_case(sandbox, monkeypatch, capsys):
    ask = FakeAsk(confirm_answer=False)
    thresholds = en_th_sandbox(sandbox, monkeypatch)
    argv = ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", "th",
            "--thresholds", str(thresholds), "--only", "th_text"]
    safety_suite.main(argv, run_stage=fake_stages, interactive=True, ask_factory=lambda: ask)
    assert "Partial run" not in capsys.readouterr().out
    verdict = run_partial(sandbox, monkeypatch, "th_text", languages=("th",))
    assert verdict["run_scope"] == "full"


def th_argv(sandbox, *extra, languages=("th",)):
    return ["run", "--model", "org/model", "--model-id", "x", "--modalities", "text", "--languages", *languages,
            "--thresholds", str(write_thresholds(sandbox)), *extra]


def test_run_args_record_effective_configuration(sandbox):
    verdict = safety_suite.main(th_argv(sandbox), run_stage=fake_stages, interactive=False)
    ra = verdict["run_args"]
    assert (ra["backend"], ra["batch_size"], ra["max_consecutive_failures"]) == ("hf", 8, 3)
    assert ra["precision"]["mode"] == "4bit" and ra["precision"]["weights_dtype"] == "float16"
    assert ra["protocol"] == {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "system_prompt": "none"}
    assert ra["per_model"]["max_new_tokens"] == 256 and ra["per_model"]["max_input_tokens"] == 2048
    assert ra["model_input"]["chat_template"]["prefix"] == "<s>[INST] "
    assert ra["model_revision"] == "a" * 40 and ra["adapter_revision"] is None
    assert ra["provenance"]["libraries"]["transformers"] == "4.0.0"
    assert ra["provenance"]["gpu"]["devices"][0]["name"] == "Test GPU"
    assert ra["provenance"]["suite"]["available"] is False  # the sandbox is not a git checkout
    assert set(ra["provenance"]["prompt_sets"]) == {"th_text"}
    assert len(ra["provenance"]["prompt_sets"]["th_text"]) == 64
    assert ra["config_mismatches"] == []
    assert verdict["case_results"][0]["n_truncated_inputs"] == 0
    json.dumps(verdict)


def test_revision_is_resolved_once_and_pinned_for_every_stage(sandbox, monkeypatch):
    hub = FakeHub(commit="b" * 40)
    monkeypatch.setattr(run_config, "HUB", hub)
    seen = []

    def stages(cmd, log_path):
        if cmd[1] == "model_wrapper.py":
            seen.append(cmd)
        return fake_stages(cmd, log_path)

    verdict = safety_suite.main(th_argv(sandbox, "--model-revision", "v1.0"), run_stage=stages, interactive=False)
    assert hub.asked == [("org/model", "v1.0")]
    assert all(cmd[cmd.index("--model-revision") + 1] == "b" * 40 for cmd in seen)
    assert verdict["run_args"]["model_revision"] == "b" * 40


def test_unresolvable_revision_is_a_suite_error_before_anything_runs(sandbox, monkeypatch):
    monkeypatch.setattr(run_config, "HUB", FakeHub(fail=True))
    with pytest.raises(SystemExit, match="could not resolve"):
        safety_suite.main(th_argv(sandbox), run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)
    assert not (sandbox / "results").exists()
    assert "suite error" in (sandbox / "_index.md").read_text(encoding="utf-8")


def stages_with(report_changes):
    def stages(cmd, log_path):
        code = fake_stages(cmd, log_path)
        if cmd[1] == "model_wrapper.py":
            out = Path(cmd[cmd.index("--out") + 1])
            report = stage_report_for(cmd)
            report_changes(report, out)
            run_config.write_stage_report(out, report)
        return code

    return stages


def test_missing_run_config_is_a_suite_error(sandbox):
    def no_report(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text("{}\n", encoding="utf-8") if cmd[1] == "model_wrapper.py" else fake_stages(cmd, log_path)
        return 0

    with pytest.raises(SystemExit, match="run configuration"):
        safety_suite.main(th_argv(sandbox), run_stage=no_report, interactive=False)


def test_wrapper_that_did_not_use_the_forwarded_settings_is_a_suite_error(sandbox):
    stages = stages_with(lambda report, out: report["args"].update(batch_size=1))
    with pytest.raises(SystemExit, match="did not use the forwarded settings.*batch_size"):
        safety_suite.main(th_argv(sandbox), run_stage=stages, interactive=False)


def test_cases_with_different_tested_configuration_stop_the_run(sandbox, monkeypatch):
    en_th_sandbox(sandbox, monkeypatch)
    calls = []

    def change_second(report, out):
        calls.append(out.name)
        if len(calls) == 2:
            report["provenance"]["model_revision"] = "c" * 40

    argv = ["run", "--model", "org/model", "--model-id", "x", "--modalities", "text", "--languages", "en", "th",
            "--thresholds", str(sandbox / "t.json"), "--only", "en_text,th_text"]
    with pytest.raises(SystemExit, match="provenance.model_revision differs"):
        safety_suite.main(argv, run_stage=stages_with(change_second), interactive=False)
    assert "no verdict" in (sandbox / "_index.md").read_text(encoding="utf-8")


def test_cases_with_different_environment_are_flagged_not_failed(sandbox, monkeypatch):
    en_th_sandbox(sandbox, monkeypatch)
    calls = []

    def change_second(report, out):
        calls.append(out.name)
        if len(calls) == 2:
            report["provenance"]["gpu"]["driver"] = "2"

    argv = ["run", "--model", "org/model", "--model-id", "x", "--modalities", "text", "--languages", "en", "th",
            "--thresholds", str(sandbox / "t.json"), "--only", "en_text,th_text"]
    verdict = safety_suite.main(argv, run_stage=stages_with(change_second), interactive=False)
    assert [m["field"] for m in verdict["run_args"]["config_mismatches"]] == ["provenance.gpu"]


def test_api_backend_records_unavailable_not_defaults(sandbox):
    argv = th_argv(sandbox, "--backend", "api", "--batch-size", "4")
    verdict = safety_suite.main(argv, run_stage=fake_stages, interactive=False)
    ra = verdict["run_args"]
    assert ra["batch_size"] == 4
    assert ra["precision"]["available"] is False
    assert ra["model_input"]["chat_template"]["available"] is False
    assert ra["provenance"]["gpu"]["available"] is False
    assert ra["provenance"]["api"]["base_url"] == "https://api.example/v1"
    assert ra["provenance"]["api"]["reported_model"] == {"available": False, "reason": "deferred"}
    assert verdict["case_results"][0]["n_truncated_inputs"]["available"] is False


def test_summary_shows_the_effective_settings(sandbox, capsys):
    safety_suite.main(th_argv(sandbox, "--precision", "as_published"), run_stage=fake_stages, interactive=True,
                      ask_factory=lambda: FakeAsk(confirm_answer=False))
    out = capsys.readouterr().out
    assert "precision as_published" in out and "batch 8" in out and "max_new_tokens 256" in out
    assert "model@" + "a" * 12 in out
