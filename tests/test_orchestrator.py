import json
import subprocess
import sys
from pathlib import Path

import pytest

import orchestrator
import run_config
from capabilities import parse_capabilities
from conftest import REAL_LID_PROBLEMS, REAL_PYTHAINLP_PROBLEMS, stage_report_for
from cases import REGISTRY, OnlyError, select_cases
from orchestrator import (
    ABORTED_EXIT_CODE,
    BUDGET_EXIT_CODE,
    CONFIG_ERROR_EXIT_CODE,
    STOPPED_NO_ANSWER_EXIT_CODE,
    CaseRunner,
    SuiteError,
    preflight,
    run_case,
)

TH_TEXT = next(case for case in REGISTRY if case.case_id == "th_text")
THRESHOLDS = {"provisional": True, "cases": {"th_text": {"threshold": 0.3, "min_scored": 1}}}


@pytest.fixture
def th_prompts(tmp_path, monkeypatch):
    prompts = tmp_path / "th_text.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"thaisafetybench_{i}"}) + "\n" for i in range(4)), encoding="utf-8")
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_text", CaseRunner(prompts, "score_th_text", "classifier"))
    return prompts


class FakeStages:
    """Stands in for the subprocesses: records commands and writes canned outputs."""

    def __init__(self, gen_code=0, sidecar=None, scored_labels=(1, 0, 0, 0), report=True, n_truncated=0):
        self.gen_code, self.sidecar, self.scored_labels = gen_code, sidecar, scored_labels
        self.report, self.n_truncated = report, n_truncated
        self.calls = []

    def __call__(self, cmd, log_path):
        self.calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        if cmd[1] == "model_wrapper.py":
            out.write_text("{}\n", encoding="utf-8")
            if self.report:
                run_config.write_stage_report(out, stage_report_for(cmd, n_truncated_inputs=self.n_truncated))
            if self.sidecar is not None:
                Path(str(out) + ".stopped.json").write_text(json.dumps(self.sidecar), encoding="utf-8")
            return self.gen_code
        records = [
            {"label": label, "excluded": False, "exclusion_reason": None, "scoring_method": "classifier", "truncated": False}
            for label in self.scored_labels
        ]
        out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return 0


def scorer_was_called(stages):
    return any(cmd[1] == "score_th_text.py" for cmd in stages.calls)


def test_completed_case_is_scored(tmp_path, th_prompts):
    stages = FakeStages()
    result = run_case(TH_TEXT, tmp_path, ["--model", "m"], THRESHOLDS, run_stage=stages, timestamp="t")
    assert (result["status"], result["score"], result["n_scored"], result["n_prompts"]) == ("pass", 0.25, 4, 4)
    assert result["scoring_method"] == "classifier"
    gen_cmd = stages.calls[0]
    assert "--out" in gen_cmd and gen_cmd[-2:] == ["--model", "m"]


def test_stopped_case_is_insufficient_and_skips_scoring(tmp_path, th_prompts):
    sidecar = {"reason": "3 batches no answer", "n_prompts": 4, "n_attempted": 3, "n_answered": 0}
    stages = FakeStages(gen_code=STOPPED_NO_ANSWER_EXIT_CODE, sidecar=sidecar)
    result = run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stages, timestamp="t")
    assert (result["status"], result["score"], result["n_scored"]) == ("insufficient_data", None, 0)
    assert result["scoring_method"] is None
    assert "3/4 prompts attempted, 0 answered" in result["notes"]
    assert not scorer_was_called(stages)


def test_stop_code_without_sidecar_is_a_suite_error(tmp_path, th_prompts):
    stages = FakeStages(gen_code=STOPPED_NO_ANSWER_EXIT_CODE, sidecar=None)
    with pytest.raises(SuiteError):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stages, timestamp="t")


def test_config_error_exit_carries_the_wrappers_message(tmp_path, th_prompts):
    def refuses(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        Path(str(out) + ".error.json").write_text(json.dumps({"kind": "PrecisionError", "message": "pre-quantized (awq)"}))
        return CONFIG_ERROR_EXIT_CODE

    with pytest.raises(SuiteError, match="th_text: pre-quantized"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=refuses, timestamp="t")
    assert not list(tmp_path.glob("*.error.json"))


def test_config_error_code_without_its_sidecar_is_a_generic_suite_error(tmp_path, th_prompts):
    with pytest.raises(SuiteError, match="exited with code 4"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=FakeStages(gen_code=CONFIG_ERROR_EXIT_CODE), timestamp="t")


def test_other_generation_failure_is_a_suite_error(tmp_path, th_prompts):
    with pytest.raises(SuiteError):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=FakeStages(gen_code=1), timestamp="t")


def selection(languages):
    return select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": languages}))


def test_preflight_rejects_missing_threshold_entry(th_prompts):
    with pytest.raises(SuiteError, match="th_text: thresholds config"):
        preflight(selection(["th"]), {"cases": {}})


def test_preflight_rejects_missing_prompt_file(tmp_path, monkeypatch):
    missing = CaseRunner(tmp_path / "absent.jsonl", "score_th_text", "classifier")
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_text", missing)
    with pytest.raises(SuiteError, match="prompt file missing"):
        preflight(selection(["th"]), THRESHOLDS)


def test_scoring_failure_is_a_suite_error(tmp_path, th_prompts):
    stages = FakeStages()
    original = stages.__call__

    def failing_scorer(cmd, log_path):
        return 1 if cmd[1] == "score_th_text.py" else original(cmd, log_path)

    with pytest.raises(SuiteError, match="scoring exited"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=failing_scorer, timestamp="t")


def test_scorer_modules_import_without_torch():
    # Only meaningful where torch is installed; elsewhere the import can't happen at all.
    pytest.importorskip("torch")
    # The orchestrator imports the scorers for aggregate(); that must not pull in the model stack.
    code = (
        "import sys; import score_en_text, score_th_text, score_over_refusal, run_config; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    assert subprocess.run([sys.executable, "-c", code], cwd=scripts).returncode == 0


def test_every_implemented_case_has_a_runner():
    missing = [c.case_id for c in REGISTRY if c.implemented and c.case_id not in orchestrator.CASE_RUNNERS]
    assert missing == []


def test_case_result_carries_the_truncated_input_count(tmp_path, th_prompts):
    stages = FakeStages(n_truncated=3)
    result = run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stages, timestamp="t")
    assert result["n_truncated_inputs"] == 3


def test_stopped_case_still_reports_truncated_inputs(tmp_path, th_prompts):
    sidecar = {"reason": "r", "n_prompts": 4, "n_attempted": 3, "n_answered": 0}
    stages = FakeStages(gen_code=STOPPED_NO_ANSWER_EXIT_CODE, sidecar=sidecar, n_truncated=2)
    result = run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stages, timestamp="t")
    assert result["n_truncated_inputs"] == 2


def test_missing_run_config_is_a_suite_error(tmp_path, th_prompts):
    stages = FakeStages(report=False)
    with pytest.raises(SuiteError, match="th_text.*run configuration"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stages, timestamp="t")
    assert not scorer_was_called(stages)


def test_unparsable_run_config_is_a_suite_error(tmp_path, th_prompts):
    class Corrupt(FakeStages):
        def __call__(self, cmd, log_path):
            code = super().__call__(cmd, log_path)
            if cmd[1] == "model_wrapper.py":
                run_config.sidecar_path(cmd[cmd.index("--out") + 1]).write_text("{not json", encoding="utf-8")
            return code

    with pytest.raises(SuiteError, match="not valid JSON"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=Corrupt(), timestamp="t")


def test_responses_path_matches_the_stage_output(tmp_path, th_prompts):
    stages = FakeStages()
    run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stages, timestamp="t")
    assert run_config.sidecar_path(orchestrator.responses_path(tmp_path, TH_TEXT)).exists()


def test_suite_modules_import_without_the_model_stack():
    # Holds with or without torch installed: the suite process must never load the model stack.
    code = (
        "import sys; import safety_suite, orchestrator, run_config, verdict, cases; "
        "bad = [m for m in ('torch', 'transformers', 'peft', 'bitsandbytes', 'huggingface_hub') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    assert subprocess.run([sys.executable, "-c", code], cwd=scripts).returncode == 0


def test_budget_stop_is_a_suite_error_that_keeps_the_partial_responses(tmp_path, th_prompts):
    def stops(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text('{"id": "thaisafetybench_0"}\n', encoding="utf-8")
        Path(str(out) + ".budget.json").write_text(
            json.dumps({"reason": "spend limit 5 THB", "n_prompts": 4, "n_completed": 1}), encoding="utf-8"
        )
        return BUDGET_EXIT_CODE

    with pytest.raises(SuiteError, match="th_text: spend limit reached after 1/4.*--resume"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=stops, timestamp="t")
    assert (tmp_path / "responses_th_text.jsonl").exists()


def test_case_result_carries_exclusions_cap_and_cost_from_the_responses(tmp_path, th_prompts):
    thresholds = {"provisional": True, "cases": {"th_text": {"threshold": 0.3, "min_scored": 1, "max_excluded_fraction": 0.10}}}

    def stage(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        if cmd[1] == "model_wrapper.py":
            rows = [
                {"id": "a", "no_answer": False, "cost_thb": 0.5},
                {"id": "b", "no_answer": True, "cost_thb": 1.0},
                {"id": "c", "no_answer": True, "cost_thb": 1.0},
                {"id": "d", "no_answer": False, "cost_thb": 0.5},
            ]
            out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            run_config.write_stage_report(out, stage_report_for(cmd))
            return 0
        scored = [
            {"label": 0, "excluded": e, "exclusion_reason": "generate_failed" if e else None,
             "scoring_method": "classifier", "truncated": False}
            for e in (False, True, True, False)
        ]
        out.write_text("".join(json.dumps(r) + "\n" for r in scored), encoding="utf-8")
        return 0

    result = run_case(TH_TEXT, tmp_path, [], thresholds, run_stage=stage, timestamp="t", generation={"max_new_tokens": 2048})
    assert (result["n_excluded"], result["n_no_answer"], result["excluded_fraction"]) == (2, 2, 0.5)
    assert (result["max_new_tokens"], result["cost_thb"], result["max_excluded_fraction"]) == (2048, 3.0, 0.10)
    assert result["status"] == "insufficient_data" and result["score"] == 0.0


def test_aborted_api_generation_is_a_suite_error_that_keeps_the_partial_responses(tmp_path, th_prompts):
    def aborts(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text('{"id": "thaisafetybench_0"}\n', encoding="utf-8")
        Path(str(out) + ".aborted.json").write_text(
            json.dumps({"reason": "PermissionError: ledger is locked", "n_prompts": 4, "n_completed": 1}), encoding="utf-8"
        )
        return ABORTED_EXIT_CODE

    with pytest.raises(SuiteError, match="th_text: generation aborted after 1/4.*PermissionError.*--resume"):
        run_case(TH_TEXT, tmp_path, [], THRESHOLDS, run_stage=aborts, timestamp="t")
    assert (tmp_path / "responses_th_text.jsonl").exists()
    assert not list(tmp_path.glob("*.aborted.json"))


# --- derived cases ---

EN_INTEGRITY = next(case for case in REGISTRY if case.case_id == "en_integrity")
INTEGRITY_THRESHOLDS = {
    "provisional": True,
    "cases": {"en_integrity": {"threshold": 0, "min_scored": 1, "line_rule": "any_line", "share_threshold": 0.25}},
}
GOOD = "Here is a clear and helpful answer to the question that was asked."


def write_replies(run_dir, case_id, replies):
    path = run_dir / f"responses_{case_id}.jsonl"
    path.write_text(
        "".join(json.dumps({"id": f"{case_id}_{i}", **reply}) + "\n" for i, reply in enumerate(replies)),
        encoding="utf-8",
    )
    return path


def ok(text=GOOD):
    return {"response": text, "finish_reason": "stop"}


class IntegrityStages:
    """Runs the real scorer in-process with a stub language identifier, so the command line is parsed for real."""

    def __init__(self, code=0):
        self.code, self.calls = code, []

    def __call__(self, cmd, log_path):
        self.calls.append(cmd)
        if self.code:
            return self.code
        import score_en_integrity

        score_en_integrity.main(cmd[2:], identify=lambda line: "en")
        return 0


@pytest.fixture
def replies(tmp_path):
    write_replies(tmp_path, "over_refusal", [ok(), ok(), {"response": "", "finish_reason": "stop", "no_answer": True, "generation_failed": True}])
    write_replies(tmp_path, "en_text", [ok(), {"response": "", "finish_reason": "stop", "generation_failed": True}])
    return tmp_path


def test_derived_case_scores_the_saved_replies_and_sends_nothing(replies):
    stages = IntegrityStages()
    result = run_case(EN_INTEGRITY, replies, ["--model", "m"], INTEGRITY_THRESHOLDS, run_stage=stages, timestamp="t")

    assert len(stages.calls) == 1 and stages.calls[0][1] == "score_en_integrity.py"
    cmd = stages.calls[0]
    assert cmd[cmd.index("--responses") + 1 : cmd.index("--out")] == [
        f"over_refusal={replies / 'responses_over_refusal.jsonl'}",
        f"en_text={replies / 'responses_en_text.jsonl'}",
    ]
    assert cmd[cmd.index("--line-rule") + 1] == "any_line" and cmd[cmd.index("--share-threshold") + 1] == "0.25"
    assert (replies / "scored_en_integrity.jsonl").exists() and (replies / "review_en_integrity.csv").exists()
    # 5 replies in the pool: 2 good over_refusal replies and an empty answer (a failed sample, scored), plus
    # 1 good and 1 HTTP failure (excluded) from en_text.
    assert result["n_prompts"] == 5
    assert (result["n_scored"], result["n_excluded"], result["n_no_answer"]) == (4, 1, 1)
    assert result["score"] == 0.75
    assert result["scoring_method"] == "fasttext_line_script_degeneracy"
    assert result["status"] == "pass"  # the placeholder threshold 0 is met by any score
    assert result["per_language"] == {"en": {"pass": True, "n_samples": 4, "pass_rate": 0.75}}
    assert result["max_new_tokens"] is None and result["cost_thb"] is None


def test_derived_case_notes_record_the_rule_and_the_source_hashes(replies):
    result = run_case(EN_INTEGRITY, replies, [], INTEGRITY_THRESHOLDS, run_stage=IntegrityStages(), timestamp="t")
    assert "line_rule=any_line, share_threshold=0.25" in result["notes"]
    for case_id in ("over_refusal", "en_text"):
        digest = run_config.file_sha256(replies / f"responses_{case_id}.jsonl")
        assert f"{case_id} sha256={digest}" in result["notes"]


def test_a_threshold_from_the_config_decides_pass_or_fail(replies):
    strict = {"cases": {"en_integrity": {"threshold": 0.95, "min_scored": 1}}}
    result = run_case(EN_INTEGRITY, replies, [], strict, run_stage=IntegrityStages(), timestamp="t")
    assert (result["score"], result["status"]) == (0.75, "fail")


def test_line_rule_and_share_threshold_come_from_the_config(replies):
    config = {"cases": {"en_integrity": {"threshold": 0, "min_scored": 1, "line_rule": "share_of_lines", "share_threshold": 0.5}}}
    stages = IntegrityStages()
    run_case(EN_INTEGRITY, replies, [], config, run_stage=stages, timestamp="t")
    cmd = stages.calls[0]
    assert cmd[cmd.index("--line-rule") + 1] == "share_of_lines" and cmd[cmd.index("--share-threshold") + 1] == "0.5"


def test_a_stopped_source_makes_the_derived_case_insufficient_without_scoring(replies):
    (replies / "responses_over_refusal.jsonl.stopped.json").write_text("{}", encoding="utf-8")
    stages = IntegrityStages()
    result = run_case(EN_INTEGRITY, replies, [], INTEGRITY_THRESHOLDS, run_stage=stages, timestamp="t")
    assert stages.calls == []
    assert (result["status"], result["score"], result["n_scored"]) == ("insufficient_data", None, 0)
    assert "over_refusal stopped" in result["notes"]


def test_a_source_that_is_insufficient_for_other_reasons_does_not_block_scoring(replies):
    # The sources' own verdicts are not read: integrity is about the replies, which are all there.
    result = run_case(EN_INTEGRITY, replies, [], INTEGRITY_THRESHOLDS, run_stage=IntegrityStages(), timestamp="t")
    assert result["score"] is not None


def test_a_missing_source_file_is_a_suite_error(replies):
    (replies / "responses_en_text.jsonl").unlink()
    with pytest.raises(SuiteError, match="no responses file for en_text"):
        run_case(EN_INTEGRITY, replies, [], INTEGRITY_THRESHOLDS, run_stage=IntegrityStages(), timestamp="t")


def test_a_failing_integrity_scorer_is_a_suite_error(replies):
    with pytest.raises(SuiteError, match="scoring exited with code 1"):
        run_case(EN_INTEGRITY, replies, [], INTEGRITY_THRESHOLDS, run_stage=IntegrityStages(code=1), timestamp="t")


def test_an_all_excluded_pool_is_insufficient(tmp_path):
    failed = {"response": "", "generation_failed": True}
    write_replies(tmp_path, "over_refusal", [failed])
    write_replies(tmp_path, "en_text", [failed])
    result = run_case(EN_INTEGRITY, tmp_path, [], INTEGRITY_THRESHOLDS, run_stage=IntegrityStages(), timestamp="t")
    assert (result["score"], result["n_scored"], result["status"]) == (None, 0, "insufficient_data")


# --- preflight for derived cases ---


@pytest.fixture
def english_runners(tmp_path, monkeypatch):
    for case_id, scorer in (("en_text", "score_en_text"), ("over_refusal", "score_over_refusal")):
        prompts = tmp_path / f"{case_id}.jsonl"
        prompts.write_text(json.dumps({"id": "a"}) + "\n", encoding="utf-8")
        monkeypatch.setitem(orchestrator.CASE_RUNNERS, case_id, CaseRunner(prompts, scorer, "x"))


def english_selection(only=None):
    capabilities = parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["en"]})
    return select_cases(capabilities, only=only)


def english_thresholds(**integrity):
    entry = {"threshold": 0, "min_scored": 100, "line_rule": "any_line", "share_threshold": 0.25, **integrity}
    plain = {"threshold": 0.2, "min_scored": 1}
    return {"cases": {"en_text": plain, "over_refusal": plain, "en_integrity": entry}}


def test_preflight_accepts_a_derived_case_with_no_prompt_file_and_a_zero_threshold(english_runners):
    preflight(english_selection(), english_thresholds())
    assert orchestrator.CASE_RUNNERS["en_integrity"].prompts is None


@pytest.mark.parametrize(
    "integrity, message",
    [
        ({"line_rule": "most_lines"}, "line_rule must be one of"),
        ({"share_threshold": 0}, "share_threshold must be a number in"),
        ({"share_threshold": 1.5}, "share_threshold must be a number in"),
        ({"share_threshold": True}, "share_threshold must be a number in"),
        ({"share_threshold": "0.25"}, "share_threshold must be a number in"),
    ],
)
def test_preflight_rejects_a_bad_integrity_rule_config(english_runners, integrity, message):
    with pytest.raises(SuiteError, match=message):
        preflight(english_selection(), english_thresholds(**integrity))


def test_preflight_requires_the_integrity_threshold_entry(english_runners):
    thresholds = english_thresholds()
    del thresholds["cases"]["en_integrity"]
    with pytest.raises(SuiteError, match="en_integrity: thresholds config needs"):
        preflight(english_selection(), thresholds)


def test_preflight_reports_a_derived_case_whose_source_is_not_selected(english_runners):
    selection = english_selection()
    broken = type(selection)(
        cases=tuple(c for c in selection.cases if c.case_id != "over_refusal"),
        not_testable_gates=selection.not_testable_gates,
        declared_scope=selection.declared_scope,
    )
    with pytest.raises(SuiteError, match="source cases are not in this run: over_refusal"):
        preflight(broken, english_thresholds())


def test_preflight_surfaces_the_language_id_problems(english_runners, monkeypatch):
    monkeypatch.setattr(orchestrator, "lid_problems", lambda: ["fasttext is not importable from /x/python"])
    with pytest.raises(SuiteError, match="en_integrity: fasttext is not importable from /x/python"):
        preflight(english_selection(), english_thresholds())


def test_preflight_skips_the_language_id_check_when_the_derived_case_is_not_selected(english_runners, monkeypatch):
    monkeypatch.setattr(orchestrator, "lid_problems", lambda: pytest.fail("no integrity case, no check"))
    preflight(english_selection(only=["en_text"]), english_thresholds())


def test_lid_problems_name_the_interpreter_and_the_build_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(orchestrator.score_en_integrity, "LID_MODEL_PATH", tmp_path / "lid.176.bin")
    problems = REAL_LID_PROBLEMS()
    assert len(problems) == 2
    assert sys.executable in problems[0] and "fasttext==0.9.3" in problems[0] and "/Dssize_t=ptrdiff_t" in problems[0]
    assert "lid.176.bin" in problems[1] and "fetch_data.sh" in problems[1]


def test_lid_problems_are_empty_when_both_are_present(monkeypatch, tmp_path):
    model = tmp_path / "lid.176.bin"
    model.write_bytes(b"x")
    monkeypatch.setattr(orchestrator.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(orchestrator.score_en_integrity, "LID_MODEL_PATH", model)
    assert REAL_LID_PROBLEMS() == []


def test_a_derived_case_never_goes_through_the_generation_path(replies, monkeypatch):
    monkeypatch.setattr(orchestrator, "responses_path", lambda *a: pytest.fail("derived cases write no responses"))
    run_case(EN_INTEGRITY, replies, [], INTEGRITY_THRESHOLDS, run_stage=IntegrityStages(), timestamp="t")


# --- th_integrity: the same derived-case path with the Thai scorer ---

TH_INTEGRITY = next(case for case in REGISTRY if case.case_id == "th_integrity")
THAI_REPLY = "ผู้ใช้งานที่ได้รับอนุญาตสามารถเข้าถึงระบบได้"
THAI_ENTRY = {
    "threshold": 0,
    "min_scored": 1,
    "line_rule": "any_line",
    "share_threshold": 0.25,
    "min_thai_ratio": 0.5,
    "min_line_letters": 5,
    "foreign_letter_share": 0.01,
}
THAI_THRESHOLDS = {"provisional": True, "cases": {"th_integrity": THAI_ENTRY}}


class ThaiIntegrityStages:
    """Runs the real Thai scorer in-process with a whitespace tokenizer, so the command line is parsed for real."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, log_path):
        self.calls.append(cmd)
        import score_th_integrity

        score_th_integrity.main(cmd[2:], tokenize=str.split)
        return 0


@pytest.fixture
def thai_replies(tmp_path):
    write_replies(tmp_path, "th_text", [ok(THAI_REPLY), ok(THAI_REPLY + "\nThis line is written entirely in English words"), ok("")])
    return tmp_path


def test_th_integrity_runner_is_a_derived_case_with_the_thai_scorer():
    runner = orchestrator.CASE_RUNNERS["th_integrity"]
    assert runner.prompts is None and runner.scorer == "score_th_integrity"
    assert TH_INTEGRITY.implemented and TH_INTEGRITY.sources == ("th_text",)


def test_th_integrity_scores_the_saved_th_text_replies(thai_replies):
    stages = ThaiIntegrityStages()
    result = run_case(TH_INTEGRITY, thai_replies, [], THAI_THRESHOLDS, run_stage=stages, timestamp="t")

    cmd = stages.calls[0]
    assert cmd[1] == "score_th_integrity.py"
    assert cmd[cmd.index("--responses") + 1 : cmd.index("--out")] == [f"th_text={thai_replies / 'responses_th_text.jsonl'}"]
    for flag, value in (("--min-thai-ratio", "0.5"), ("--min-line-letters", "5"), ("--foreign-letter-share", "0.01")):
        assert cmd[cmd.index(flag) + 1] == value
    assert (thai_replies / "scored_th_integrity.jsonl").exists() and (thai_replies / "review_th_integrity.csv").exists()
    assert (result["n_prompts"], result["n_scored"]) == (3, 3)
    assert result["score"] == pytest.approx(1 / 3)
    assert result["scoring_method"] == "thai_char_ratio_script_degeneracy"
    assert result["status"] == "pass"  # the placeholder threshold 0 is met by any score
    assert result["per_language"] == {"th": {"pass": True, "n_samples": 3, "pass_rate": pytest.approx(1 / 3)}}
    assert "min_thai_ratio=0.5" in result["notes"] and "pythainlp" in result["notes"]


def test_th_integrity_settings_come_from_the_config(thai_replies):
    config = {"cases": {"th_integrity": {**THAI_ENTRY, "line_rule": "share_of_lines", "share_threshold": 0.5, "min_thai_ratio": 0.3}}}
    stages = ThaiIntegrityStages()
    run_case(TH_INTEGRITY, thai_replies, [], config, run_stage=stages, timestamp="t")
    cmd = stages.calls[0]
    assert cmd[cmd.index("--line-rule") + 1] == "share_of_lines" and cmd[cmd.index("--min-thai-ratio") + 1] == "0.3"


def test_th_integrity_defaults_apply_when_the_config_omits_the_extra_keys(thai_replies):
    config = {"cases": {"th_integrity": {"threshold": 0, "min_scored": 1}}}
    stages = ThaiIntegrityStages()
    run_case(TH_INTEGRITY, thai_replies, [], config, run_stage=stages, timestamp="t")
    cmd = stages.calls[0]
    assert cmd[cmd.index("--min-line-letters") + 1] == "5" and cmd[cmd.index("--foreign-letter-share") + 1] == "0.01"


@pytest.fixture
def thai_runners(tmp_path, monkeypatch):
    prompts = tmp_path / "th_text.jsonl"
    prompts.write_text(json.dumps({"id": "a"}) + "\n", encoding="utf-8")
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_text", CaseRunner(prompts, "score_th_text", "x"))
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_knowledge", CaseRunner(prompts, "score_th_knowledge", "x"))


def thai_selection():
    return select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["th"]}))


def thai_thresholds(**integrity):
    plain = {"threshold": 0.2, "min_scored": 1}
    return {"cases": {"th_text": plain, "th_knowledge": plain, "th_integrity": {**THAI_ENTRY, **integrity}}}


def test_preflight_accepts_th_integrity(thai_runners):
    preflight(thai_selection(), thai_thresholds())


@pytest.mark.parametrize(
    "integrity, message",
    [
        ({"line_rule": "most_lines"}, "line_rule must be one of"),
        ({"share_threshold": 0}, "share_threshold must be a number in"),
        ({"min_thai_ratio": 0}, "min_thai_ratio must be a number in"),
        ({"min_thai_ratio": 1.5}, "min_thai_ratio must be a number in"),
        ({"foreign_letter_share": True}, "foreign_letter_share must be a number in"),
        ({"min_line_letters": 0}, "min_line_letters must be an integer of at least 1"),
        ({"min_line_letters": 2.5}, "min_line_letters must be an integer of at least 1"),
        ({"min_line_letters": True}, "min_line_letters must be an integer of at least 1"),
    ],
)
def test_preflight_rejects_a_bad_th_integrity_config(thai_runners, integrity, message):
    with pytest.raises(SuiteError, match=message):
        preflight(thai_selection(), thai_thresholds(**integrity))


def test_preflight_names_a_missing_pythainlp(thai_runners, monkeypatch):
    monkeypatch.setattr(orchestrator, "pythainlp_problems", lambda: ["pythainlp is not importable"])
    with pytest.raises(SuiteError, match="th_integrity: pythainlp is not importable"):
        preflight(thai_selection(), thai_thresholds())


def test_pythainlp_problems_reports_an_uninstalled_package(monkeypatch):
    monkeypatch.setattr(orchestrator.importlib.util, "find_spec", lambda name: None)
    (problem,) = REAL_PYTHAINLP_PROBLEMS()
    assert "pythainlp==5.4.0" in problem


# --- th_knowledge: a generating case scored by the letter-match scorer ---

TH_KNOWLEDGE = next(case for case in REGISTRY if case.case_id == "th_knowledge")
KNOWLEDGE_THRESHOLDS = {"provisional": True, "cases": {"th_knowledge": {"threshold": 0, "min_scored": 1}}}


def test_th_knowledge_runner_is_a_generating_case_with_the_letter_match_scorer():
    runner = orchestrator.CASE_RUNNERS["th_knowledge"]
    assert runner.prompts.name == "th_knowledge.jsonl" and runner.scorer == "score_th_knowledge"
    assert runner.scoring_method == "letter_match_v2" and TH_KNOWLEDGE.implemented and not TH_KNOWLEDGE.derived


def test_th_knowledge_is_generated_then_scored_by_the_real_scorer(tmp_path, monkeypatch):
    prompts = tmp_path / "th_knowledge.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"thai_exam_onet_{i}"}) + "\n" for i in range(4)), encoding="utf-8")
    monkeypatch.setitem(orchestrator.CASE_RUNNERS, "th_knowledge", CaseRunner(prompts, "score_th_knowledge", "letter_match_v2"))
    replies = ["C", "Answer: B", "no idea", ""]

    def stages(cmd, log_path):
        out = Path(cmd[cmd.index("--out") + 1])
        if cmd[1] == "model_wrapper.py":
            records = [
                {"id": f"thai_exam_onet_{i}", "native_category": "onet", "answer": "C", "response": reply, "generation_failed": False}
                for i, reply in enumerate(replies)
            ]
            out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            run_config.write_stage_report(out, stage_report_for(cmd))
            return 0
        import score_th_knowledge

        score_th_knowledge.main(cmd[2:])
        return 0

    result = run_case(TH_KNOWLEDGE, tmp_path, ["--model", "m"], KNOWLEDGE_THRESHOLDS, run_stage=stages, timestamp="t")
    assert (result["n_prompts"], result["n_scored"], result["score"]) == (4, 4, 0.25)
    assert result["scoring_method"] == "letter_match_v2" and result["status"] == "pass"
    assert "2/4 replies unparsable" in result["notes"]
    assert result["per_language"] is None


def lyric_reply():
    return {**ok("la la la la la la la la la la " * 6), "tags": ["lyrics", "hash_check"]}


def test_lyrics_replies_leave_the_prompt_count_and_the_exclusion_ceiling_alone(tmp_path):
    write_replies(tmp_path, "over_refusal", [ok(), ok()])
    write_replies(tmp_path, "en_text", [ok(), ok(), *[lyric_reply() for _ in range(6)]])
    config = {"cases": {"en_integrity": {"threshold": 0, "min_scored": 1, "max_excluded_fraction": 0.0}}}
    result = run_case(EN_INTEGRITY, tmp_path, [], config, run_stage=IntegrityStages(), timestamp="t")
    assert result["n_prompts"] == 4 and result["n_scored"] == 4
    assert result["n_excluded"] == 0 and result["excluded_fraction"] == 0
    assert result["status"] == "pass" and result["score"] == 1.0
    assert "6 replies not scored (lyrics_repeat_by_design)" in result["notes"]
    assert "excluded_tags=lyrics" in result["notes"]
