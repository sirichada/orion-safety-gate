import dataclasses
import json
from pathlib import Path

import pytest

import cases
import orchestrator
import run_config
import safety_suite
from conftest import FakeHub, stage_report_for
from orchestrator import CaseRunner


def parse(*argv):
    return safety_suite.parse_args(["run", *argv])


BASE = ["--model", "m", "--model-id", "Mango 1.5-9B", "--capabilities", "c.json", "--thresholds", "t.json"]


def test_wrapper_args_always_forward_every_setting():
    api_args = parse(*BASE, "--backend", "api", "--batch-size", "4", "--max-consecutive-failures", "5", "--max-thb", "12")
    safety_suite.apply_protocol_sampling(api_args, run_config.GREEDY_SAMPLING)
    api = safety_suite.wrapper_args(api_args)
    assert api == [
        "--model", "m", "--backend", "api", "--batch-size", "4", "--max-consecutive-failures", "5",
        "--temperature", "0.0", "--top-p", "1.0", "--max-new-tokens", "256",
        "--api-concurrency", "3", "--no-answer-retries", "0", "--max-thb", "12.0",
    ]
    hf = safety_suite.wrapper_args(parse(*BASE, "--precision", "full"), {"model": "a" * 40, "adapter": None})
    assert hf == [
        "--model", "m", "--backend", "hf", "--batch-size", "8", "--max-consecutive-failures", "3",
        "--temperature", "0.0", "--top-p", "1.0", "--max-new-tokens", "256",
        "--precision", "full", "--max-input-tokens", "2048", "--model-revision", "a" * 40,
    ]


def test_per_case_caps_and_ledger_are_forwarded():
    args = parse(*BASE, "--backend", "api", "--max-thb", "3", "--api-concurrency", "2", "--resume", "some/run")
    forwarded = safety_suite.wrapper_args(args, None, {"max_new_tokens": 4096}, Path("run/spend_ledger.json"))
    assert forwarded[forwarded.index("--max-new-tokens") + 1] == "4096"
    assert forwarded[forwarded.index("--api-concurrency") + 1] == "2"
    assert forwarded[forwarded.index("--spend-ledger") + 1] == str(Path("run/spend_ledger.json"))
    assert "--resume" in forwarded
    assert safety_suite.forwarded_values(args, None, {"max_new_tokens": 4096})["max_new_tokens"] == 4096


def test_api_defaults_batch_to_a_checkpoint_size_and_concurrency_to_three():
    args = parse(*BASE, "--backend", "api", "--max-thb", "3")
    assert (args.batch_size, args.api_concurrency) == (12, 3)
    assert parse(*BASE).batch_size == 8


@pytest.mark.parametrize(
    "extra",
    [
        ["--backend", "api"],  # no --max-thb
        ["--backend", "api", "--max-thb", "0"],
        ["--backend", "api", "--max-thb", "5", "--api-concurrency", "4"],
        ["--max-thb", "5"],  # hf
    ],
)
def test_api_runs_need_a_spend_limit_and_at_most_three_in_flight(extra):
    with pytest.raises(SystemExit):
        parse(*BASE, *extra)


def test_dry_run_may_omit_the_spend_limit():
    assert parse(*BASE, "--backend", "api", "--dry-run").max_thb is None


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
    # These tests exercise th_text alone, so the other Thai cases are taken out of selection.
    monkeypatch.setattr(
        cases, "REGISTRY", tuple(dataclasses.replace(c, implemented=False) if c.case_id in ("th_integrity", "th_knowledge") else c
            for c in cases.REGISTRY)
    )
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
    assert verdict["gate_incomplete"] == {"smartness": ["th_integrity", "th_knowledge"]}
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
    assert "th_integrity, th_knowledge" in out and "not implemented in this version" in out
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
    assert ra["protocol"] == {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "caller_system_prompt": "none"}
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


def test_wrapper_config_error_message_reaches_the_suite_error_and_leaves_no_run_folder(sandbox):
    message = "the checkpoint is pre-quantized (awq); --precision 4bit would re-quantize or upcast it."

    def refuses(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        Path(str(out) + ".error.json").write_text(json.dumps({"kind": "PrecisionError", "message": message}))
        return 4

    with pytest.raises(SystemExit, match="pre-quantized"):
        safety_suite.main(th_argv(sandbox), run_stage=refuses, interactive=False)
    assert "pre-quantized" in (sandbox / "_index.md").read_text(encoding="utf-8")
    assert list((sandbox / "results").iterdir()) == []


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


GREEDY = {"temperature": 0.0, "top_p": 1.0}
VENDOR = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def write_generation(sandbox, protocols=None, **models):
    """Caps per model; every model gets the greedy protocol unless protocols says otherwise."""
    path = sandbox / "generation.json"
    if protocols is None:
        protocols = {"x": GREEDY}
    path.write_text(json.dumps({"cases": {}, "models": models, "protocols": protocols}), encoding="utf-8")
    return str(path)


API = ["--backend", "api", "--model", "Mango 1.5-9B"]


def test_api_backend_records_unavailable_not_defaults(sandbox):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--batch-size", "4", "--max-thb", "40", "--generation-config", generation)
    verdict = safety_suite.main(argv, run_stage=fake_stages, interactive=False)
    ra = verdict["run_args"]
    assert ra["batch_size"] == 4
    assert ra["precision"]["available"] is False
    assert ra["model_input"]["chat_template"]["available"] is False
    assert ra["provenance"]["gpu"]["available"] is False
    assert ra["provenance"]["api"]["base_url"] == "https://api.example/v1"
    assert ra["provenance"]["api"]["reported_model"] == {"available": False, "reason": "deferred"}
    assert verdict["case_results"][0]["n_truncated_inputs"]["available"] is False
    assert ra["api"] == {"concurrency": 3, "no_answer_retries": 0, "max_thb": 40.0, "spent_thb": None}
    assert ra["generation"]["per_case"] == {"th_text": {"max_new_tokens": 4096, "max_input_tokens": None}}
    assert len(ra["generation"]["sha256"]) == 64 and ra["generation"]["source"] == generation
    assert verdict["case_results"][0]["max_new_tokens"] == 4096


def test_api_run_without_a_cap_for_the_model_is_refused_before_any_stage(sandbox):
    generation = write_generation(sandbox, other={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation)
    with pytest.raises(SystemExit, match="no max_new_tokens for case 'th_text' and model 'x'"):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)


def test_api_run_for_an_unpriced_model_is_refused(sandbox):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, "--backend", "api", "--model", "Unpriced 1B", "--max-thb", "40", "--generation-config", generation)
    with pytest.raises(SystemExit, match="no price for model 'Unpriced 1B'"):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)


def test_each_case_is_forwarded_its_own_cap(sandbox, monkeypatch):
    other = sandbox / "over_refusal.jsonl"
    other.write_text("".join(json.dumps({"id": f"xstest_{i}"}) + "\n" for i in range(4)), encoding="utf-8")
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "over_refusal", CaseRunner(other, "score_over_refusal", "string_match"))
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}, "over_refusal": {"max_new_tokens": 1024}})
    seen = {}

    def stages(cmd, log_path):
        if cmd[1] == "model_wrapper.py":
            seen[Path(cmd[cmd.index("--out") + 1]).name] = cmd[cmd.index("--max-new-tokens") + 1]
        return fake_stages(cmd, log_path)

    both = sandbox / "both.json"
    both.write_text(json.dumps({"provisional": True, "cases": {
        "th_text": {"threshold": 0.2, "min_scored": 1}, "over_refusal": {"threshold": 0.2, "min_scored": 1}}}))
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--only", "th_text,over_refusal",
                   "--thresholds", str(both), languages=("en", "th"))
    safety_suite.main(argv, run_stage=stages, interactive=False)
    assert seen == {"responses_th_text.jsonl": "4096", "responses_over_refusal.jsonl": "1024"}


def test_dry_run_prints_spend_bounds_and_writes_nothing(sandbox, capsys):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--dry-run")
    result = safety_suite.main(argv, run_stage=lambda *a: pytest.fail("a dry run sends nothing"), interactive=False)
    out = capsys.readouterr().out
    assert result is None
    assert "spend bounds (Mango 1.5-9B" in out and "th_text" in out and "4,096" in out
    assert "expected is n/a" in out and "--max-thb 40" in out and "Dry run: nothing was sent" in out
    assert not (sandbox / "results").exists() and not (sandbox / "_index.md").exists()


def test_dry_run_shows_the_expected_spend_from_a_calibration_file(sandbox, capsys):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    calibration = sandbox / "calibration.json"
    calibration.write_text(json.dumps({
        "model": "Mango 1.5-9B",
        "cases": {"th_text": {"mean_prompt_tokens": 100, "mean_completion_tokens": 1000}},
    }), encoding="utf-8")
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--dry-run",
                   "--calibration-file", str(calibration))
    safety_suite.main(argv, run_stage=lambda *a: pytest.fail("a dry run sends nothing"), interactive=False)
    out = capsys.readouterr().out
    # 4 prompts x (100 x input price + 1000 x output price) / 1e6 THB
    assert "0.03" in out and "expected is n/a" not in out


def test_dry_run_rejects_a_calibration_file_for_another_model(sandbox):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    calibration = sandbox / "calibration.json"
    calibration.write_text(json.dumps({"model": "Qwen 3.8-27B", "cases": {}}), encoding="utf-8")
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--dry-run",
                   "--calibration-file", str(calibration))
    with pytest.raises(SystemExit, match="is for model 'Qwen 3.8-27B'"):
        safety_suite.main(argv, run_stage=lambda *a: None, interactive=False)


def test_resume_reuses_the_run_folder_and_forwards_resume(sandbox):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    run_dir = sandbox / "results" / "earlier"
    run_dir.mkdir(parents=True)
    commands = []

    def stages(cmd, log_path):
        commands.append(cmd)
        return fake_stages(cmd, log_path)

    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--resume", str(run_dir))
    verdict = safety_suite.main(argv, run_stage=stages, interactive=False)
    wrapper_cmd = next(c for c in commands if c[1] == "model_wrapper.py")
    assert "--resume" in wrapper_cmd
    assert wrapper_cmd[wrapper_cmd.index("--out") + 1] == str(run_dir / "responses_th_text.jsonl")
    assert wrapper_cmd[wrapper_cmd.index("--spend-ledger") + 1] == str(run_dir / "spend_ledger.json")
    assert (run_dir / "verdict.json").exists() and verdict["case_results"][0]["case_id"] == "th_text"
    from datetime import datetime

    assert datetime.fromisoformat(verdict["started_at"]) <= datetime.fromisoformat(verdict["finished_at"])


def test_resume_of_a_missing_folder_is_a_suite_error(sandbox):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation,
                   "--resume", str(sandbox / "nowhere"))
    with pytest.raises(SystemExit, match="is not a run folder"):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)


def test_summary_shows_the_effective_settings(sandbox, capsys):
    safety_suite.main(th_argv(sandbox, "--precision", "as_published"), run_stage=fake_stages, interactive=True,
                      ask_factory=lambda: FakeAsk(confirm_answer=False))
    out = capsys.readouterr().out
    assert "precision as_published" in out and "batch 8" in out and "max_new_tokens 256" in out
    assert "temperature 0, top_p 1 requested" in out
    assert "model@" + "a" * 12 in out


# ------------------------------------------------------------- sampling and prompt samples


def test_sampling_flags_are_forwarded_and_expected_back():
    args = parse(*BASE, "--backend", "api", "--max-thb", "3", "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20")
    forwarded = safety_suite.wrapper_args(args, None, {"max_new_tokens": 4096})
    assert forwarded[forwarded.index("--temperature") + 1] == "1.0"
    assert forwarded[forwarded.index("--top-p") + 1] == "0.95"
    assert forwarded[forwarded.index("--top-k") + 1] == "20"
    values = safety_suite.forwarded_values(args, None, {"max_new_tokens": 4096})
    assert (values["temperature"], values["top_p"], values["top_k"]) == (1.0, 0.95, 20)


def test_default_sampling_forwards_no_top_k():
    args = parse(*BASE, "--backend", "api", "--max-thb", "3")
    assert "--top-k" not in safety_suite.wrapper_args(args, None, {"max_new_tokens": 4096})
    assert "top_k" not in safety_suite.forwarded_values(args, None, {"max_new_tokens": 4096})


def test_presence_penalty_is_forwarded_and_expected_back():
    args = parse(*BASE, "--backend", "api", "--max-thb", "3", "--presence-penalty", "1.5")
    forwarded = safety_suite.wrapper_args(args, None, {"max_new_tokens": 4096})
    assert forwarded[forwarded.index("--presence-penalty") + 1] == "1.5"
    assert safety_suite.forwarded_values(args, None, {"max_new_tokens": 4096})["presence_penalty"] == 1.5
    assert safety_suite.protocol_deviations(args) == ["sampling"]


def test_default_sampling_forwards_no_presence_penalty():
    args = parse(*BASE, "--backend", "api", "--max-thb", "3")
    assert "--presence-penalty" not in safety_suite.wrapper_args(args, None, {"max_new_tokens": 4096})
    assert "presence_penalty" not in safety_suite.forwarded_values(args, None, {"max_new_tokens": 4096})


@pytest.mark.parametrize(
    "extra",
    [
        ["--temperature", "1.0"],  # hf backend
        ["--top-k", "20"],  # hf backend
        ["--presence-penalty", "1.5"],  # hf backend
        ["--backend", "api", "--max-thb", "3", "--presence-penalty", "2.5"],
        ["--backend", "api", "--max-thb", "3", "--presence-penalty", "-2.1"],
        ["--backend", "api", "--max-thb", "3", "--top-k", "0"],
        ["--backend", "api", "--max-thb", "3", "--top-p", "0"],
        ["--backend", "api", "--max-thb", "3", "--temperature", "-0.1"],
        ["--sample-per-case", "5"],  # no seed
        ["--sample-seed", "1"],  # no sample size
        ["--sample-per-case", "0", "--sample-seed", "1"],
    ],
)
def test_sampling_and_sample_flags_are_validated(extra):
    with pytest.raises(SystemExit):
        parse(*BASE, *extra)


def test_sample_records_are_seeded_paired_and_in_file_order(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text("".join(json.dumps({"id": f"r{i}"}) + "\n" for i in range(100)), encoding="utf-8")
    first = safety_suite.sample_records(path, 10, 7, "th_text")
    assert first == safety_suite.sample_records(path, 10, 7, "th_text")
    assert len(first) == 10 and len({r["id"] for r in first}) == 10
    positions = [int(r["id"][1:]) for r in first]
    assert positions == sorted(positions)
    assert first != safety_suite.sample_records(path, 10, 8, "th_text")
    assert first != safety_suite.sample_records(path, 10, 7, "en_text")
    assert len(safety_suite.sample_records(path, 500, 7, "th_text")) == 100


def test_sampled_vendor_run_records_sampling_and_sample_and_is_not_an_admission(sandbox):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    prompts = sandbox / "th_text.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"thaisafetybench_{i}", "model_input": f"q{i}"}) + "\n" for i in range(30)),
                       encoding="utf-8")
    seen = {}

    def stages(cmd, log_path):
        if cmd[1] == "model_wrapper.py":
            seen["prompts"] = Path(cmd[cmd.index("--prompts") + 1])
            seen["top_k"] = cmd[cmd.index("--top-k") + 1]
            n = len(seen["prompts"].read_text(encoding="utf-8").splitlines())
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text("{}\n", encoding="utf-8")
            run_config.write_stage_report(out, stage_report_for(cmd))
            seen["n"] = n
            return 0
        out = Path(cmd[cmd.index("--out") + 1])
        record = {"label": 0, "excluded": False, "exclusion_reason": None, "scoring_method": "classifier", "truncated": False}
        out.write_text((json.dumps(record) + "\n") * seen["n"], encoding="utf-8")
        return 0

    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--temperature", "1.0",
                   "--top-p", "0.95", "--top-k", "20", "--sample-per-case", "5", "--sample-seed", "11")
    verdict = safety_suite.main(argv, run_stage=stages, interactive=False)
    run_dir = next((sandbox / "results").iterdir())
    assert seen["prompts"] == run_dir / "prompts_th_text.jsonl" and seen["n"] == 5 and seen["top_k"] == "20"
    ra = verdict["run_args"]
    assert ra["sampling"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "top_k_location": "extra_body",
                              "presence_penalty": None, "presence_penalty_location": None, "matches_protocol": False}
    assert ra["protocol"]["temperature"] == 0.0  # the declared protocol is still recorded beside it
    assert ra["prompt_sample"]["per_case"] == 5 and ra["prompt_sample"]["seed"] == 11
    assert len(ra["prompt_sample"]["ids"]["th_text"]) == 5
    assert ra["provenance"]["prompt_sets"]["th_text"] == run_config.file_sha256(seen["prompts"])
    assert verdict["protocol_deviations"] == ["sampling", "prompt_sample"] and verdict["admitted"] is False
    assert verdict["case_results"][0]["n_prompts"] == 5
    index = (sandbox / "_index.md").read_text(encoding="utf-8")
    assert "non-protocol: sampling+prompt_sample" in index


def test_protocol_run_lists_no_deviations(sandbox):
    verdict = safety_suite.main(th_argv(sandbox), run_stage=fake_stages, interactive=False)
    assert verdict["protocol_deviations"] == []
    assert verdict["run_args"]["sampling"]["matches_protocol"] is True and verdict["run_args"]["prompt_sample"] is None


def test_summary_marks_non_protocol_sampling(sandbox, capsys):
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--dry-run",
                   "--generation-config", write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}}))
    safety_suite.main(argv, run_stage=lambda *a: pytest.fail("a dry run sends nothing"), interactive=False)
    out = capsys.readouterr().out
    assert "temperature 1, top_p 0.95, top_k 20 (extra_body) requested [NOT the protocol" in out


def test_summary_shows_presence_penalty(sandbox, capsys):
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20",
                   "--presence-penalty", "1.5", "--dry-run",
                   "--generation-config", write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}}))
    safety_suite.main(argv, run_stage=lambda *a: pytest.fail("a dry run sends nothing"), interactive=False)
    out = capsys.readouterr().out
    assert "top_k 20 (extra_body), presence_penalty 1.5 requested [NOT the protocol" in out


def test_dry_run_bounds_follow_the_sample_size(sandbox, capsys):
    generation = write_generation(sandbox, x={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--dry-run",
                   "--sample-per-case", "2", "--sample-seed", "3")
    safety_suite.main(argv, run_stage=lambda *a: pytest.fail("a dry run sends nothing"), interactive=False)
    out = capsys.readouterr().out
    assert "seeded sample of 2 per case, seed 3" in out
    assert "th_text (2 prompts)" in out and "total        2 prompts" in out
    line = next(l for l in out.splitlines() if l.strip().startswith("th_text"))
    assert line.split()[1] == "2"
    assert not (sandbox / "results").exists()


# ------------------------------------------------------------- per-model sampling protocol


MANGO = {**VENDOR, "presence_penalty": 1.5}


def api_args(model_id, *extra):
    return parse("--model", "m", "--model-id", model_id, "--capabilities", "c.json", "--thresholds", "t.json",
                 "--backend", "api", "--max-thb", "3", *extra)


def committed_protocols():
    return json.loads(safety_suite.DEFAULT_GENERATION_CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("model_id, expected", [
    ("Qwen 3.8-27B", VENDOR), ("Qwen 3.8-Flash-Next", VENDOR), ("Mango 1.5-9B", MANGO)])
def test_committed_protocol_fills_unset_sampling_and_matches(model_id, expected):
    args = api_args(model_id)
    safety_suite.apply_protocol_sampling(args, run_config.resolve_protocol_sampling(committed_protocols(), model_id, "api"))
    want = {"top_k": None, "presence_penalty": None, **expected}
    assert {k: getattr(args, k) for k in want} == want
    assert safety_suite.sampling_of(args)["matches_protocol"] is True
    assert safety_suite.protocol_deviations(args) == []


def test_presence_penalty_is_part_of_the_mango_protocol_only():
    config = committed_protocols()
    mango = api_args("Mango 1.5-9B")
    safety_suite.apply_protocol_sampling(mango, run_config.resolve_protocol_sampling(config, "Mango 1.5-9B", "api"))
    assert safety_suite.forwarded_values(mango)["presence_penalty"] == 1.5
    # Qwen's protocol has none, so Mango's setting on a Qwen run is a deviation
    qwen = api_args("Qwen 3.8-27B", "--presence-penalty", "1.5")
    safety_suite.apply_protocol_sampling(qwen, run_config.resolve_protocol_sampling(config, "Qwen 3.8-27B", "api"))
    assert safety_suite.protocol_deviations(qwen) == ["sampling"]


def test_hf_backend_stays_greedy_without_a_protocol_entry():
    args = parse(*BASE)
    assert (args.temperature, args.top_p) == (0.0, 1.0)
    assert run_config.resolve_protocol_sampling({}, "anything", "hf") == run_config.GREEDY_SAMPLING
    safety_suite.apply_protocol_sampling(args, run_config.resolve_protocol_sampling({}, "anything", "hf"))
    assert safety_suite.sampling_of(args)["matches_protocol"] is True


def test_api_run_without_a_protocol_entry_is_refused_before_any_stage(sandbox):
    generation = write_generation(sandbox, protocols={"other": GREEDY}, x={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation)
    with pytest.raises(SystemExit, match="no sampling protocol for model 'x'"):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)


def test_api_run_without_a_cap_still_refuses_when_the_protocol_exists(sandbox):
    generation = write_generation(sandbox, protocols={"x": VENDOR}, other={"th_text": {"max_new_tokens": 4096}})
    argv = th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation)
    with pytest.raises(SystemExit, match="no max_new_tokens for case 'th_text' and model 'x'"):
        safety_suite.main(argv, run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)


def test_malformed_protocol_entry_is_refused():
    with pytest.raises(run_config.GenerationConfigError, match="needs numeric temperature and top_p"):
        run_config.resolve_protocol_sampling({"protocols": {"m": {"temperature": 1.0, "seed": 3}}}, "m", "api")


def test_run_with_no_sampling_flags_uses_the_protocol_and_is_not_a_deviation(sandbox):
    generation = write_generation(sandbox, protocols={"x": {**VENDOR, "presence_penalty": 1.5}},
                                  x={"th_text": {"max_new_tokens": 4096}})
    seen = {}

    def stages(cmd, log_path):
        if cmd[1] == "model_wrapper.py":
            seen["cmd"] = cmd
        return fake_stages(cmd, log_path)

    verdict = safety_suite.main(th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation),
                                run_stage=stages, interactive=False)
    cmd = seen["cmd"]
    assert [cmd[cmd.index(f) + 1] for f in ("--temperature", "--top-p", "--top-k", "--presence-penalty")] == [
        "1.0", "0.95", "20", "1.5"]
    assert verdict["run_args"]["sampling"]["matches_protocol"] is True
    assert verdict["protocol_deviations"] == []


def test_a_flag_overriding_the_protocol_is_recorded_as_a_deviation(sandbox):
    generation = write_generation(sandbox, protocols={"x": VENDOR}, x={"th_text": {"max_new_tokens": 4096}})
    verdict = safety_suite.main(
        th_argv(sandbox, *API, "--max-thb", "40", "--generation-config", generation, "--temperature", "0"),
        run_stage=fake_stages, interactive=False)
    sampling = verdict["run_args"]["sampling"]
    assert (sampling["temperature"], sampling["top_p"], sampling["top_k"]) == (0.0, 0.95, 20)
    assert sampling["matches_protocol"] is False
    assert verdict["protocol_deviations"] == ["sampling"] and verdict["admitted"] is False


# --- derived cases: en_integrity scores the saved replies of over_refusal and en_text ---

REPLY = "Here is a clear and helpful answer to the question that was asked."


def english_sandbox(sandbox, monkeypatch):
    for case_id in ("en_text", "over_refusal"):
        prompts = sandbox / f"{case_id}.jsonl"
        prompts.write_text("".join(json.dumps({"id": f"{case_id}_{i}"}) + "\n" for i in range(4)), encoding="utf-8")
        monkeypatch.setitem(orchestrator.CASE_RUNNERS, case_id, CaseRunner(prompts, "score_th_text", "classifier"))
    path = sandbox / "t_en.json"
    entry = {"threshold": 0.2, "min_scored": 1}
    path.write_text(json.dumps({"provisional": True, "cases": {
        "en_text": entry,
        "over_refusal": entry,
        "en_integrity": {"threshold": 0, "min_scored": 1, "max_excluded_fraction": 0.10, "line_rule": "any_line", "share_threshold": 0.25},
    }}))
    return path


class EnglishStages:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, log_path):
        self.calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        if cmd[1] == "model_wrapper.py":
            out.write_text("".join(json.dumps({"id": f"r{i}", "response": REPLY, "finish_reason": "stop"}) + "\n" for i in range(4)), encoding="utf-8")
            run_config.write_stage_report(out, stage_report_for(cmd))
            return 0
        if cmd[1] == "score_en_integrity.py":
            import score_en_integrity

            score_en_integrity.main(cmd[2:], identify=lambda line: "en")
            return 0
        return fake_stages(cmd, log_path)

    def wrapper_outs(self):
        return [Path(c[c.index("--out") + 1]).name for c in self.calls if c[1] == "model_wrapper.py"]


def english_argv(sandbox, thresholds, *extra, languages=("en",)):
    return ["run", "--model", "m", "--model-id", "x", "--modalities", "text", "--languages", *languages,
            "--thresholds", str(thresholds), *extra]


def test_english_run_scores_the_saved_replies_without_generating_for_the_derived_case(sandbox, monkeypatch, capsys):
    thresholds = english_sandbox(sandbox, monkeypatch)
    stages = EnglishStages()
    verdict = safety_suite.main(english_argv(sandbox, thresholds), run_stage=stages, interactive=False)

    assert verdict["cases_run"] == ["en_text", "over_refusal", "en_integrity"]
    assert stages.wrapper_outs() == ["responses_en_text.jsonl", "responses_over_refusal.jsonl"]
    integrity = verdict["case_results"][-1]
    assert (integrity["status"], integrity["score"], integrity["n_scored"], integrity["n_prompts"]) == ("pass", 1.0, 8, 8)
    assert integrity["per_language"] == {"en": {"pass": True, "n_samples": 8, "pass_rate": 1.0}}
    assert verdict["languages_verified"] == {"en": True}
    assert "capabilities_verified" not in verdict
    assert verdict["smartness_gate"] == "pass"
    assert verdict["gate_incomplete"] == {}
    run_dir = next((sandbox / "results").glob("x_*"))
    assert (run_dir / "scored_en_integrity.jsonl").exists() and (run_dir / "review_en_integrity.csv").exists()
    # The derived case wrote no stage report, and the verdict still records one configuration.
    assert not (run_dir / "responses_en_integrity.jsonl.run_config.json").exists()
    out = capsys.readouterr().out
    assert "total        8 prompts" in out
    assert "en_integrity (replies of over_refusal, en_text; no prompts sent)" in out
    assert set(verdict["run_args"]["provenance"]["prompt_sets"]) == {"en_text", "over_refusal"}


def test_only_the_derived_case_is_refused_before_anything_runs(sandbox, monkeypatch):
    thresholds = english_sandbox(sandbox, monkeypatch)
    with pytest.raises(SystemExit, match="also name over_refusal, en_text"):
        safety_suite.main(english_argv(sandbox, thresholds, "--only", "en_integrity"),
                          run_stage=lambda *a: pytest.fail("no stage may run"), interactive=False)
    assert not (sandbox / "results").exists()


def test_only_naming_the_derived_case_with_its_sources_runs_all_three(sandbox, monkeypatch):
    thresholds = english_sandbox(sandbox, monkeypatch)
    verdict = safety_suite.main(english_argv(sandbox, thresholds, "--only", "en_text,over_refusal,en_integrity"),
                                run_stage=EnglishStages(), interactive=False)
    assert verdict["cases_run"] == ["en_text", "over_refusal", "en_integrity"]
    assert verdict["run_scope"] == "full"


def test_english_and_thai_run_leaves_thai_integrity_unattempted(sandbox, monkeypatch):
    thresholds = english_sandbox(sandbox, monkeypatch)
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_text", orchestrator.CASE_RUNNERS["en_text"])
    data = json.loads(thresholds.read_text())
    data["cases"]["th_text"] = {"threshold": 0.2, "min_scored": 1}
    thresholds.write_text(json.dumps(data))
    verdict = safety_suite.main(english_argv(sandbox, thresholds, languages=("en", "th")), run_stage=EnglishStages(), interactive=False)
    assert verdict["languages_verified"] == {"en": True, "th": None}
    assert verdict["gate_incomplete"] == {"smartness": ["th_integrity", "th_knowledge"]}


def test_api_dry_run_skips_the_derived_case(sandbox, monkeypatch, capsys):
    thresholds = english_sandbox(sandbox, monkeypatch)
    generation = write_generation(sandbox, x={"en_text": {"max_new_tokens": 4096}, "over_refusal": {"max_new_tokens": 1024}})
    argv = english_argv(sandbox, thresholds, *API, "--max-thb", "40", "--generation-config", generation, "--dry-run")
    safety_suite.main(argv, run_stage=lambda *a: pytest.fail("a dry run sends nothing"), interactive=False)
    out = capsys.readouterr().out
    assert "en_text" in out and "over_refusal" in out and "en_integrity   " not in out


def test_resume_with_a_relative_folder_forwards_absolute_paths(sandbox, monkeypatch):
    thresholds = english_sandbox(sandbox, monkeypatch)
    generation = write_generation(sandbox, x={"en_text": {"max_new_tokens": 4096}, "over_refusal": {"max_new_tokens": 1024}})
    (sandbox / "results" / "earlier").mkdir(parents=True)
    monkeypatch.chdir(sandbox)
    stages = EnglishStages()
    argv = english_argv(sandbox, thresholds, *API, "--max-thb", "40", "--generation-config", generation,
                        "--resume", "results/earlier")
    verdict = safety_suite.main(argv, run_stage=stages, interactive=False)
    outs = [Path(c[c.index("--out") + 1]) for c in stages.calls]
    assert outs and all(p.is_absolute() for p in outs)
    scorer = next(c for c in stages.calls if c[1] == "score_en_integrity.py")
    assert all(Path(a.split("=", 1)[1]).is_absolute() for a in scorer if a.startswith(("over_refusal=", "en_text=")))
    assert (sandbox / "results" / "earlier" / "verdict.json").exists() and verdict["cases_run"][-1] == "en_integrity"
    assert "results/earlier/" in (sandbox / "_index.md").read_text(encoding="utf-8").replace("\\", "/")


def test_resume_folder_outside_the_repo_is_allowed(sandbox, monkeypatch, tmp_path_factory):
    thresholds = english_sandbox(sandbox, monkeypatch)
    generation = write_generation(sandbox, x={"en_text": {"max_new_tokens": 4096}, "over_refusal": {"max_new_tokens": 1024}})
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.setattr(safety_suite, "REPO_ROOT", sandbox / "repo")
    argv = english_argv(sandbox, thresholds, *API, "--max-thb", "40", "--generation-config", generation, "--resume", str(elsewhere))
    safety_suite.main(argv, run_stage=EnglishStages(), interactive=False)
    assert str(elsewhere.resolve()) in (sandbox / "_index.md").read_text(encoding="utf-8")


def test_format_verdict_summary_lists_cases_and_banners():
    verdict = {
        "case_results": [
            {"case_id": "en_text", "gate": "safety", "score": 0.15, "direction": "lower_is_better",
             "threshold": 0.2, "n_scored": 300, "n_prompts": 400, "status": "pass"},
            {"case_id": "th_knowledge", "gate": "smartness", "score": None, "direction": "higher_is_better",
             "threshold": 0, "n_scored": 0, "n_prompts": 565, "status": "insufficient_data"},
        ],
        "declared_scope": {"not_attempted": [
            {"case_id": "en_image", "gate": "safety", "reason_code": "unimplemented"},
            {"case_id": None, "gate": "safety", "reason_code": "no_case_applies"},
        ]},
        "safety_gate": "pass", "smartness_gate": "insufficient_data", "admitted": False,
        "thresholds_provisional": True, "run_scope": "partial",
    }
    out = safety_suite.format_verdict_summary(verdict)
    assert "en_text" in out and "0.150" in out and "<= 0.2" in out and "300/400" in out
    assert ">= 0" in out and "insufficient_data" in out
    assert "en_image" in out and "not run (unimplemented)" in out
    assert "admitted: False" in out and "provisional" in out and "Partial run" in out
