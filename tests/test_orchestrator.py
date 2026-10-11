import json
import subprocess
import sys
from pathlib import Path

import pytest

import orchestrator
import run_config
from capabilities import parse_capabilities
from conftest import stage_report_for
from cases import REGISTRY, select_cases
from orchestrator import CONFIG_ERROR_EXIT_CODE, STOPPED_NO_ANSWER_EXIT_CODE, CaseRunner, SuiteError, preflight, run_case

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
