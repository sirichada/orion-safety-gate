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
sidecar is removed. A generation that reaches its spend limit exits with
model_wrapper.BUDGET_EXIT_CODE and a <responses>.budget.json sidecar; that is a
suite error too, and the responses written so far are kept for --resume. An api generation
stopped by an unexpected error exits with model_wrapper.ABORTED_EXIT_CODE and a
<responses>.aborted.json sidecar, handled the same way. Any other failure, including either exit code without its
sidecar, is a suite error and stops the whole run.

Every generation stage also leaves <responses>.run_config.json (see run_config.py);
a case whose generation finished or stopped without one is a suite error.

A derived case (cases.CaseSpec.sources) has no generation stage: it scores the
responses files its source cases left in the same run folder, in one scorer
subprocess, and leaves no run_config report. A source that was stopped leaves its
.stopped.json sidecar, and the derived case is then insufficient_data without scoring.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import run_config
import score_en_integrity
import score_th_integrity
import score_th_knowledge
from cases import CaseSpec, Selection
from run_config import read_jsonl
from verdict import build_case_result, result_options, threshold_entry_problems

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
LOGS = REPO_ROOT / "runs" / "logs"

# Kept equal to model_wrapper.STOPPED_NO_ANSWER_EXIT_CODE without importing it,
# so this process never imports the model-calling stack.
STOPPED_NO_ANSWER_EXIT_CODE = 3

# Kept equal to model_wrapper.CONFIG_ERROR_EXIT_CODE for the same reason.
CONFIG_ERROR_EXIT_CODE = 4

# Kept equal to model_wrapper.BUDGET_EXIT_CODE for the same reason.
BUDGET_EXIT_CODE = 5

# Kept equal to model_wrapper.ABORTED_EXIT_CODE for the same reason.
ABORTED_EXIT_CODE = 6


class SuiteError(RuntimeError):
    """A failure of the suite itself; no verdict is produced."""


@dataclass(frozen=True)
class CaseRunner:
    prompts: Path | None  # None for a derived case, which sends no prompts
    scorer: str  # script name under scripts/, also the module whose aggregate() is used
    scoring_method: str


CASE_RUNNERS: dict[str, CaseRunner] = {
    "en_text": CaseRunner(REPO_ROOT / "data" / "prompt-sets" / "en_text.jsonl", "score_en_text", "classifier"),
    "th_text": CaseRunner(REPO_ROOT / "data" / "prompt-sets" / "th_text.jsonl", "score_th_text", "classifier"),
    "over_refusal": CaseRunner(
        REPO_ROOT / "data" / "prompt-sets" / "over_refusal.jsonl", "score_over_refusal", "string_match"
    ),
    "en_integrity": CaseRunner(None, "score_en_integrity", score_en_integrity.SCORING_METHOD),
    "th_integrity": CaseRunner(None, "score_th_integrity", score_th_integrity.SCORING_METHOD),
    "th_knowledge": CaseRunner(
        REPO_ROOT / "data" / "prompt-sets" / "th_knowledge.jsonl", "score_th_knowledge", score_th_knowledge.SCORING_METHOD
    ),
}

INTEGRITY_SCORERS = {"score_en_integrity": score_en_integrity, "score_th_integrity": score_th_integrity}

# Per-scorer config keys beyond line_rule and share_threshold, each with its CLI flag and default.
EXTRA_INTEGRITY_SETTINGS = {
    "score_th_integrity": (
        ("min_thai_ratio", "--min-thai-ratio", score_th_integrity.DEFAULT_MIN_THAI_RATIO),
        ("min_line_letters", "--min-line-letters", score_th_integrity.DEFAULT_MIN_LINE_LETTERS),
        ("foreign_letter_share", "--foreign-letter-share", score_th_integrity.DEFAULT_FOREIGN_LETTER_SHARE),
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


def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def lid_problems() -> list[str]:
    """What stops score_en_integrity from running in this interpreter, which also runs the scorer subprocess."""
    problems = []
    if importlib.util.find_spec("fasttext") is None:
        problems.append(
            f"fasttext is not importable from {sys.executable}; install fasttext==0.9.3 into this interpreter "
            '(Windows: set CL="/std:c++17 /Dssize_t=ptrdiff_t" first)'
        )
    if not score_en_integrity.LID_MODEL_PATH.exists():
        problems.append(f"fastText language-ID model not found at {score_en_integrity.LID_MODEL_PATH}; run scripts/fetch_data.sh")
    return problems


def pythainlp_problems() -> list[str]:
    """What stops score_th_integrity from running in this interpreter, which also runs the scorer subprocess."""
    if importlib.util.find_spec("pythainlp") is None:
        return [f"pythainlp is not importable from {sys.executable}; install pythainlp==5.4.0 into this interpreter"]
    return []


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def integrity_config_problems(case_id: str, entry: dict, scorer: str = "score_en_integrity") -> list[str]:
    module = INTEGRITY_SCORERS[scorer]
    problems = []
    line_rule = entry.get("line_rule", module.DEFAULT_LINE_RULE)
    if line_rule not in module.LINE_RULES:
        problems.append(f"{case_id}: line_rule must be one of {module.LINE_RULES}, got {line_rule!r}")
    share = entry.get("share_threshold", module.DEFAULT_SHARE_THRESHOLD)
    if not _is_number(share) or not 0 < share <= 1:
        problems.append(f"{case_id}: share_threshold must be a number in (0, 1], got {share!r}")
    for key, _flag, default in EXTRA_INTEGRITY_SETTINGS.get(scorer, ()):
        value = entry.get(key, default)
        if key == "min_line_letters":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append(f"{case_id}: {key} must be an integer of at least 1, got {value!r}")
        elif not _is_number(value) or not 0 < value <= 1:
            problems.append(f"{case_id}: {key} must be a number in (0, 1], got {value!r}")
    return problems


def preflight(selection: Selection, thresholds: dict) -> None:
    """Fails before any generation if a selected case can't be run or judged."""
    problems = []
    selected = {case.case_id for case in selection.cases}
    for case in selection.cases:
        runner = CASE_RUNNERS.get(case.case_id)
        if runner is None:
            problems.append(f"{case.case_id}: no runner registered")
        elif case.derived:
            missing = [source for source in case.sources if source not in selected]
            if missing:
                problems.append(f"{case.case_id}: its source cases are not in this run: {', '.join(missing)}")
        elif not runner.prompts.exists():
            problems.append(f"{case.case_id}: prompt file missing ({runner.prompts}); run its loader first")
        entry = thresholds.get("cases", {}).get(case.case_id)
        if entry is None or "threshold" not in entry or "min_scored" not in entry:
            problems.append(f"{case.case_id}: thresholds config needs 'threshold' and 'min_scored'")
        elif threshold_entry_problems(case.case_id, entry):
            problems.extend(threshold_entry_problems(case.case_id, entry))
        elif runner is not None and runner.scorer in INTEGRITY_SCORERS:
            problems.extend(integrity_config_problems(case.case_id, entry, runner.scorer))
        if runner is not None and runner.scorer == "score_en_integrity":
            problems.extend(f"{case.case_id}: {problem}" for problem in lid_problems())
        if runner is not None and runner.scorer == "score_th_integrity":
            problems.extend(f"{case.case_id}: {problem}" for problem in pythainlp_problems())
    if problems:
        raise SuiteError("Preflight failed:\n  " + "\n  ".join(problems))


def responses_file(run_dir: Path, case_id: str) -> Path:
    return run_dir / f"responses_{case_id}.jsonl"


def responses_path(run_dir: Path, case: CaseSpec) -> Path:
    return responses_file(run_dir, case.case_id)


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
    generation: dict | None = None,
    prompts: Path | None = None,
) -> dict:
    if case.derived:
        return run_derived_case(case, run_dir, thresholds, run_stage=run_stage, timestamp=timestamp)
    runner = CASE_RUNNERS[case.case_id]
    prompts = prompts or runner.prompts
    entry = thresholds["cases"][case.case_id]
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    responses = responses_path(run_dir, case)
    scored = run_dir / f"scored_{case.case_id}.jsonl"
    sidecar = Path(str(responses) + ".stopped.json")
    n_prompts = count_lines(prompts)

    n_truncated_inputs = None
    generation_report = {
        "max_new_tokens": (generation or {}).get("max_new_tokens"),
        "n_excluded": None,
        "n_no_answer": None,
        "cost_thb": None,
    }

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
            max_excluded_fraction=entry.get("max_excluded_fraction"),
            **result_options(entry),
            **generation_report,
        )

    gen_cmd = [sys.executable, "model_wrapper.py", "--prompts", str(prompts), "--out", str(responses), *wrapper_args]
    code = run_stage(gen_cmd, LOGS / f"model_wrapper_{case.case_id}_{stamp}.log")
    for exit_code, suffix, what in (
        (BUDGET_EXIT_CODE, ".budget.json", "spend limit reached"),
        (ABORTED_EXIT_CODE, ".aborted.json", "generation aborted"),
    ):
        stop_sidecar = Path(str(responses) + suffix)
        if code == exit_code and stop_sidecar.exists():
            stop = json.loads(stop_sidecar.read_text(encoding="utf-8"))
            stop_sidecar.unlink()
            raise SuiteError(
                f"{case.case_id}: {what} after {stop['n_completed']}/{stop['n_prompts']} prompts "
                f"({stop['reason']}); the responses so far are kept in {responses.parent}, continue with --resume"
            )
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
    generated = read_jsonl(responses)
    generation_report["n_excluded"] = summary.get("n_generation_failed")
    generation_report["n_no_answer"] = sum(1 for r in generated if r.get("no_answer")) if _has_api_fields(generated) else None
    costs = [r["cost_thb"] for r in generated if r.get("cost_thb") is not None]
    generation_report["cost_thb"] = round(sum(costs), 6) if costs else None
    return result(summary["score"], summary["n_scored"], summary.get("notes", ""), runner.scoring_method)


def fixed_settings(scorer: str) -> dict:
    """Scorer settings fixed in code rather than set by the thresholds file."""
    return getattr(INTEGRITY_SCORERS.get(scorer), "FIXED_SETTINGS", {})


def run_derived_case(
    case: CaseSpec,
    run_dir: Path,
    thresholds: dict,
    run_stage: RunStage = run_subprocess,
    timestamp: str | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Scores the saved replies of case.sources; sends nothing to the candidate.

    The scored and review files go to out_dir (default: run_dir); the replies are always read from run_dir.
    """
    runner = CASE_RUNNERS[case.case_id]
    entry = thresholds["cases"][case.case_id]
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    source_paths = {source: responses_file(Path(run_dir), source).resolve() for source in case.sources}
    out_dir = Path(out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = out_dir / f"scored_{case.case_id}.jsonl"
    review = out_dir / f"review_{case.case_id}.csv"
    module = INTEGRITY_SCORERS[runner.scorer]
    line_rule = entry.get("line_rule", module.DEFAULT_LINE_RULE)
    share_threshold = entry.get("share_threshold", module.DEFAULT_SHARE_THRESHOLD)
    extra_settings = [
        (key, flag, entry.get(key, default)) for key, flag, default in EXTRA_INTEGRITY_SETTINGS.get(runner.scorer, ())
    ]

    existing = {source: path for source, path in source_paths.items() if path.exists()}
    n_prompts = sum(count_lines(path) for path in existing.values())
    generation_report = {"max_new_tokens": None, "n_excluded": None, "n_no_answer": None, "cost_thb": None}

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
            max_excluded_fraction=entry.get("max_excluded_fraction"),
            **result_options(entry),
            **generation_report,
        )

    stopped = [source for source, path in source_paths.items() if Path(str(path) + ".stopped.json").exists()]
    if stopped:
        return result(None, 0, f"not scored: {', '.join(stopped)} stopped before completing, so the reply pool is incomplete", None)
    absent = [source for source in case.sources if source not in existing]
    if absent:
        raise SuiteError(f"{case.case_id}: no responses file for {', '.join(absent)} in {run_dir}")

    cmd = [
        sys.executable,
        f"{runner.scorer}.py",
        "--responses",
        *(f"{source}={path}" for source, path in source_paths.items()),
        "--out",
        str(scored),
        "--review",
        str(review),
        "--line-rule",
        line_rule,
        "--share-threshold",
        str(share_threshold),
        *(arg for _key, flag, value in extra_settings for arg in (flag, str(value))),
    ]
    code = run_stage(cmd, LOGS / f"{runner.scorer}_{case.case_id}_{stamp}.log")
    if code != 0:
        raise SuiteError(f"{case.case_id}: scoring exited with code {code}")

    summary = _aggregate(runner.scorer, read_jsonl(scored))
    # Replies left out by design are not part of the case, so they neither count as prompts nor as exclusions.
    n_prompts -= summary.get("n_excluded_by_design", 0)
    generation_report["n_excluded"] = summary.get("n_generation_failed")
    generation_report["n_no_answer"] = sum(
        1 for path in existing.values() for r in read_jsonl(path) if r.get("no_answer")
    )
    hashes = ", ".join(f"{source} sha256={run_config.file_sha256(path)}" for source, path in source_paths.items())
    notes = "; ".join(
        part
        for part in (
            ", ".join(
                [f"line_rule={line_rule}", f"share_threshold={share_threshold}"]
                + [f"{key}={value}" for key, _flag, value in extra_settings]
                + [f"{key}={value}" for key, value in fixed_settings(runner.scorer).items()]
            ),
            f"scored the saved replies of {hashes}",
            summary.get("notes", ""),
        )
        if part
    )
    return result(summary["score"], summary["n_scored"], notes, runner.scoring_method)


def _has_api_fields(records: list[dict]) -> bool:
    return any("no_answer" in r for r in records)
