"""
Runs each selected case through the pipeline (generate, then score) and
returns its CaseResult. Every stage runs as its own subprocess, so the
operating system frees each model's GPU memory before the next one loads.

Generation that stops on consecutive no-answer batches exits with
model_wrapper.STOPPED_NO_ANSWER_EXIT_CODE and leaves a <responses>.stopped.json
sidecar; that case is recorded as insufficient_data and scoring is skipped.
A generation that cannot load the model with the requested precision or
revision exits with model_wrapper.CONFIG_ERROR_EXIT_CODE and leaves a
<responses>.error.json sidecar; its message becomes the suite error and the
sidecar is removed. Any other failure, including either exit code without its
sidecar, is a suite error and stops the whole run.

Every generation stage also leaves <responses>.run_config.json (see run_config.py);
a case whose generation finished or stopped without one is a suite error.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import run_config
from cases import CaseSpec, Selection
from verdict import build_case_result

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
LOGS = REPO_ROOT / "runs" / "logs"

# Kept equal to model_wrapper.STOPPED_NO_ANSWER_EXIT_CODE without importing it,
# so this process never imports the model-calling stack.
STOPPED_NO_ANSWER_EXIT_CODE = 3

# Kept equal to model_wrapper.CONFIG_ERROR_EXIT_CODE for the same reason.
CONFIG_ERROR_EXIT_CODE = 4


class SuiteError(RuntimeError):
    """A failure of the suite itself; no verdict is produced."""


@dataclass(frozen=True)
class CaseRunner:
    prompts: Path
    scorer: str  # script name under scripts/, also the module whose aggregate() is used
    scoring_method: str


CASE_RUNNERS: dict[str, CaseRunner] = {
    "en_text": CaseRunner(REPO_ROOT / "data" / "prompt-sets" / "en_text.jsonl", "score_en_text", "classifier"),
    "th_text": CaseRunner(REPO_ROOT / "data" / "prompt-sets" / "th_text.jsonl", "score_th_text", "classifier"),
    "over_refusal": CaseRunner(
        REPO_ROOT / "data" / "prompt-sets" / "over_refusal.jsonl", "score_over_refusal", "string_match"
    ),
}

RunStage = Callable[[list[str], Path], int]


def run_subprocess(cmd: list[str], log_path: Path) -> int:
    """Runs cmd, streaming its combined output to the terminal and to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=SCRIPTS,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            # A child writing to a pipe is block-buffered otherwise, so progress would arrive only at exit.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        finally:
            # Never leave a child holding the GPU if this process is interrupted.
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def preflight(selection: Selection, thresholds: dict) -> None:
    """Fails before any generation if a selected case can't be run or judged."""
    problems = []
    for case in selection.cases:
        runner = CASE_RUNNERS.get(case.case_id)
        if runner is None:
            problems.append(f"{case.case_id}: no runner registered")
        elif not runner.prompts.exists():
            problems.append(f"{case.case_id}: prompt file missing ({runner.prompts}); run its loader first")
        entry = thresholds.get("cases", {}).get(case.case_id)
        if entry is None or "threshold" not in entry or "min_scored" not in entry:
            problems.append(f"{case.case_id}: thresholds config needs 'threshold' and 'min_scored'")
    if problems:
        raise SuiteError("Preflight failed:\n  " + "\n  ".join(problems))


def responses_path(run_dir: Path, case: CaseSpec) -> Path:
    return run_dir / f"responses_{case.case_id}.jsonl"


def read_run_config(responses: Path, case_id: str) -> dict:
    try:
        return run_config.read_stage_report(responses)
    except ValueError as e:
        raise SuiteError(f"{case_id}: {e}") from e


def _aggregate(scorer: str, records: list[dict]) -> dict:
    module = __import__(scorer)
    return module.aggregate(records)


def run_case(
    case: CaseSpec,
    run_dir: Path,
    wrapper_args: list[str],
    thresholds: dict,
    run_stage: RunStage = run_subprocess,
    timestamp: str | None = None,
) -> dict:
    runner = CASE_RUNNERS[case.case_id]
    entry = thresholds["cases"][case.case_id]
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    responses = responses_path(run_dir, case)
    scored = run_dir / f"scored_{case.case_id}.jsonl"
    sidecar = Path(str(responses) + ".stopped.json")
    n_prompts = count_lines(runner.prompts)

    n_truncated_inputs = None

    def result(score, n_scored, notes, scoring_method):
        return build_case_result(
            case,
            score=score,
            n_prompts=n_prompts,
            n_scored=n_scored,
            threshold=entry["threshold"],
            min_scored=entry["min_scored"],
            scoring_method=scoring_method,
            notes=notes,
            n_truncated_inputs=n_truncated_inputs,
        )

    gen_cmd = [sys.executable, "model_wrapper.py", "--prompts", str(runner.prompts), "--out", str(responses), *wrapper_args]
    code = run_stage(gen_cmd, LOGS / f"model_wrapper_{case.case_id}_{stamp}.log")
    if code == STOPPED_NO_ANSWER_EXIT_CODE and sidecar.exists():
        n_truncated_inputs = read_run_config(responses, case.case_id)["n_truncated_inputs"]
        stop = json.loads(sidecar.read_text(encoding="utf-8"))
        notes = (
            f"stopped before scoring: {stop['reason']} "
            f"({stop['n_attempted']}/{stop['n_prompts']} prompts attempted, {stop['n_answered']} answered)"
        )
        return result(None, 0, notes, None)
    error_sidecar = Path(str(responses) + ".error.json")
    if code == CONFIG_ERROR_EXIT_CODE and error_sidecar.exists():
        message = json.loads(error_sidecar.read_text(encoding="utf-8"))["message"]
        error_sidecar.unlink()
        raise SuiteError(f"{case.case_id}: {message}")
    if code != 0:
        raise SuiteError(f"{case.case_id}: generation exited with code {code}")
    n_truncated_inputs = read_run_config(responses, case.case_id)["n_truncated_inputs"]

    code = run_stage(
        [sys.executable, f"{runner.scorer}.py", "--responses", str(responses), "--out", str(scored)],
        LOGS / f"{runner.scorer}_{case.case_id}_{stamp}.log",
    )
    if code != 0:
        raise SuiteError(f"{case.case_id}: scoring exited with code {code}")

    summary = _aggregate(runner.scorer, read_jsonl(scored))
    return result(summary["score"], summary["n_scored"], summary.get("notes", ""), runner.scoring_method)
