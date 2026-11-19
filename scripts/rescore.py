"""
Re-derives a recorded run's verdict from its run folder: no model, no API key, no network.

Default level: reads the scored_<case>.jsonl files the run left, re-aggregates them and
re-applies a thresholds file. Needs no classifier. An integrity case's scorer settings
(line_rule, share_threshold, ...) shape its scored file, so a thresholds file that sets
them differently from the run is refused at this level.

--rerun-scorers level: scores responses_<case>.jsonl again into rescore-<timestamp>/ in the
run folder. The en_text and th_text classifiers need a GPU.

Declared scope, model id, run scope, protocol deviations and what each case measured
(prompt count, token cap, cost) come from the run's verdict.json, so a partial or sampled
run stays non-admitting. The new verdict is written beside the old one and marked
"rescored": true; the original verdict.json is never touched.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import orchestrator
from cases import NOT_TESTED_REASON, REGISTRY, UNTESTED_CAPABILITIES, Selection
from orchestrator import (
    CASE_RUNNERS,
    EXTRA_INTEGRITY_SETTINGS,
    INTEGRITY_SCORERS,
    LOGS,
    REPO_ROOT,
    RunStage,
    SuiteError,
    integrity_config_problems,
    read_jsonl,
    responses_file,
    run_derived_case,
    run_subprocess,
)
from verdict import build_case_result, build_verdict

DEFAULT_THRESHOLDS = REPO_ROOT / "config" / "thresholds.json"
VERDICT_NAME = "verdict.json"


def load_recorded_verdict(run_dir: Path) -> dict:
    path = run_dir / VERDICT_NAME
    if not path.is_file():
        raise SuiteError(f"{run_dir} has no {VERDICT_NAME}, so its declared scope and run scope cannot be recovered")
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SuiteError(f"{path} cannot be read as JSON: {e}") from e
    if not isinstance(verdict, dict):
        raise SuiteError(f"{path} is not a JSON object")
    if verdict.get("rescored"):
        raise SuiteError(f"{path} is itself a re-score; point at the run's own {VERDICT_NAME}")
    required = (
        "model_id", "run_timestamp", "declared_scope", "run_scope", "cases_run", "case_results", "run_args",
        "safety_gate", "smartness_gate", "admitted", "protocol_deviations",
    )
    missing = [key for key in required if key not in verdict]
    if missing:
        raise SuiteError(f"{path} lacks {', '.join(missing)}")
    if not isinstance(verdict["declared_scope"], dict) or "not_attempted" not in verdict["declared_scope"]:
        raise SuiteError(f"{path}: declared_scope lacks not_attempted")
    for record in verdict["case_results"]:
        gaps = [key for key in ("case_id", "status", "score", "threshold", "n_prompts", "notes") if not isinstance(record, dict) or key not in record]
        if gaps:
            raise SuiteError(f"{path}: a case record lacks {', '.join(gaps)}")
    return verdict


def current_scope(recorded_scope: dict) -> tuple[dict, list[str]]:
    """The recorded scope under today's rule that untested capabilities are not cases.

    A not_attempted entry for one (tool_calling was once a smartness case) is dropped so it no longer
    holds a gate at insufficient_data; the ids dropped are returned. Every other entry is kept as recorded.
    """
    kept = [e for e in recorded_scope["not_attempted"] if e["case_id"] not in UNTESTED_CAPABILITIES]
    dropped = [e["case_id"] for e in recorded_scope["not_attempted"] if e["case_id"] in UNTESTED_CAPABILITIES]
    scope = {**recorded_scope, "not_attempted": kept}
    scope["not_tested"] = [{"capability": name, "reason": NOT_TESTED_REASON} for name in UNTESTED_CAPABILITIES]
    return scope, dropped


def selection_from_verdict(verdict: dict) -> Selection:
    """The run's own selection, not today's: a case that was unimplemented or left out then stays so."""
    by_id = {case.case_id: case for case in REGISTRY}
    unknown = [case_id for case_id in verdict["cases_run"] if case_id not in by_id]
    if unknown:
        raise SuiteError(f"the run's verdict names cases this suite no longer knows: {', '.join(unknown)}")
    scope, _ = current_scope(verdict["declared_scope"])
    return Selection(
        cases=tuple(by_id[case_id] for case_id in verdict["cases_run"]),
        not_testable_gates=frozenset(entry["gate"] for entry in scope["not_attempted"] if entry["case_id"] is None),
        declared_scope=scope,
        run_scope=verdict["run_scope"],
    )


def threshold_problems(selection: Selection, thresholds: dict) -> list[str]:
    problems = []
    for case in selection.cases:
        entry = thresholds.get("cases", {}).get(case.case_id)
        if entry is None or "threshold" not in entry or "min_scored" not in entry:
            problems.append(f"{case.case_id}: thresholds config needs 'threshold' and 'min_scored'")
        elif CASE_RUNNERS[case.case_id].scorer in INTEGRITY_SCORERS:
            problems.extend(integrity_config_problems(case.case_id, entry, CASE_RUNNERS[case.case_id].scorer))
    return problems


def scorer_setting_names(scorer: str) -> list[tuple[str, object]]:
    """(key, default) of every setting that shapes an integrity scorer's output."""
    module = INTEGRITY_SCORERS[scorer]
    return [("line_rule", module.DEFAULT_LINE_RULE), ("share_threshold", module.DEFAULT_SHARE_THRESHOLD)] + [
        (key, default) for key, _flag, default in EXTRA_INTEGRITY_SETTINGS.get(scorer, ())
    ]


def recorded_setting(notes: str, key: str) -> str | None:
    found = re.search(rf"(?:^|[;,]\s*){re.escape(key)}=([^,;]+)", notes or "")
    return found.group(1).strip() if found else None


def setting_mismatches(case_id: str, scorer: str, entry: dict, recorded_notes: str) -> list[str]:
    """Where the scored file was made under settings other than the thresholds file now asks for."""
    problems = []
    for key, default in scorer_setting_names(scorer):
        wanted, recorded = str(entry.get(key, default)), recorded_setting(recorded_notes, key)
        if recorded != wanted:
            problems.append(f"{case_id}: {key} is {wanted} here but {recorded or 'not recorded'} in the run")
    return problems


def _derived_notes(case, entry: dict, recorded_notes: str, summary: dict) -> str:
    scorer = CASE_RUNNERS[case.case_id].scorer
    settings = ", ".join(f"{key}={entry.get(key, default)}" for key, default in scorer_setting_names(scorer))
    saved = next((part for part in recorded_notes.split("; ") if part.startswith("scored the saved replies of")), "")
    return "; ".join(part for part in (settings, saved, summary.get("notes", "")) if part)


def _case_result(case, entry: dict, recorded: dict, summary: dict, notes: str) -> dict:
    return build_case_result(
        case,
        score=summary["score"],
        n_prompts=recorded["n_prompts"],
        n_scored=summary["n_scored"],
        threshold=entry["threshold"],
        min_scored=entry["min_scored"],
        scoring_method=recorded.get("scoring_method") or CASE_RUNNERS[case.case_id].scoring_method,
        notes=notes,
        n_truncated_inputs=recorded.get("n_truncated_inputs"),
        n_excluded=summary.get("n_generation_failed"),
        n_no_answer=recorded.get("n_no_answer"),
        max_excluded_fraction=entry.get("max_excluded_fraction"),
        max_new_tokens=recorded.get("max_new_tokens"),
        cost_thb=recorded.get("cost_thb"),
    )


def _unscored_result(case, entry: dict, recorded: dict) -> dict:
    """A case the run stopped before scoring keeps its recorded reason and stays insufficient_data."""
    return build_case_result(
        case,
        score=None,
        n_prompts=recorded["n_prompts"],
        n_scored=0,
        threshold=entry["threshold"],
        min_scored=entry["min_scored"],
        scoring_method=None,
        notes=recorded.get("notes", ""),
        n_truncated_inputs=recorded.get("n_truncated_inputs"),
        max_excluded_fraction=entry.get("max_excluded_fraction"),
        max_new_tokens=recorded.get("max_new_tokens"),
    )


def rerun_problems(selection: Selection, run_dir: Path, recorded: dict) -> list[str]:
    problems = []
    scoring = [case for case in selection.cases if case.case_id in recorded and recorded[case.case_id]["score"] is not None]
    scorers = {CASE_RUNNERS[case.case_id].scorer for case in scoring}
    for case in scoring:
        sources = case.sources or (case.case_id,)
        for source in sources:
            if not responses_file(run_dir, source).is_file():
                problems.append(f"{case.case_id}: no {responses_file(run_dir, source).name} in the run folder")
    if "score_en_integrity" in scorers:
        problems.extend(f"en_integrity: {p}" for p in orchestrator.lid_problems())
    if "score_th_integrity" in scorers:
        problems.extend(f"th_integrity: {p}" for p in orchestrator.pythainlp_problems())
    return problems


def rescore_case(case, run_dir: Path, thresholds: dict, recorded: dict, *, rerun: bool, stamp: str, scored_dir: Path, run_stage: RunStage) -> dict:
    runner = CASE_RUNNERS[case.case_id]
    entry = thresholds["cases"][case.case_id]
    if recorded["score"] is None:
        return _unscored_result(case, entry, recorded)
    if rerun and case.derived:
        return run_derived_case(case, run_dir, thresholds, run_stage=run_stage, timestamp=stamp, out_dir=scored_dir)
    if rerun:
        scored = scored_dir / f"scored_{case.case_id}.jsonl"
        scored_dir.mkdir(parents=True, exist_ok=True)
        code = run_stage(
            [sys.executable, f"{runner.scorer}.py", "--responses", str(responses_file(run_dir, case.case_id).resolve()), "--out", str(scored)],
            LOGS / f"rescore_{runner.scorer}_{case.case_id}_{stamp}.log",
        )
        if code != 0:
            raise SuiteError(f"{case.case_id}: scoring exited with code {code}")
    else:
        scored = run_dir / f"scored_{case.case_id}.jsonl"
        if not scored.is_file():
            raise SuiteError(f"{case.case_id}: no {scored.name} in the run folder; use --rerun-scorers to score its responses")
    summary = orchestrator._aggregate(runner.scorer, read_jsonl(scored))
    notes = summary.get("notes", "")
    if case.derived:
        notes = _derived_notes(case, entry, recorded.get("notes", ""), summary)
    return _case_result(case, entry, recorded, summary, notes)


def rescore_run(
    run_dir: Path,
    thresholds_path: Path,
    *,
    out: Path | None = None,
    rerun_scorers: bool = False,
    force: bool = False,
    run_stage: RunStage = run_subprocess,
    stamp: str | None = None,
) -> tuple[dict, Path]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise SuiteError(f"{run_dir} is not a run folder")
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out) if out else run_dir / f"verdict.rescore-{stamp}.json"
    if out.resolve() == (run_dir / VERDICT_NAME).resolve():
        raise SuiteError(f"{out} is the run's own verdict; a re-score never replaces it, even with --force")
    if out.exists() and not force:
        raise SuiteError(f"{out} already exists; pass --force to replace it")

    old = load_recorded_verdict(run_dir)
    selection = selection_from_verdict(old)
    try:
        thresholds_bytes = Path(thresholds_path).read_bytes()
        thresholds = json.loads(thresholds_bytes.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise SuiteError(f"thresholds file {thresholds_path} cannot be read as JSON: {e}") from e
    if not isinstance(thresholds, dict) or not isinstance(thresholds.get("cases"), dict):
        raise SuiteError(f"thresholds file {thresholds_path} has no 'cases' object")
    problems = threshold_problems(selection, thresholds)
    recorded = {r["case_id"]: r for r in old["case_results"]}
    absent = [case.case_id for case in selection.cases if case.case_id not in recorded]
    problems.extend(f"{case_id}: the run's verdict has no record of this case" for case_id in absent)
    if rerun_scorers:
        problems.extend(rerun_problems(selection, run_dir, recorded))
    elif not problems:
        mismatches = []
        for case in selection.cases:
            scorer = CASE_RUNNERS[case.case_id].scorer
            if scorer in INTEGRITY_SCORERS and recorded[case.case_id]["score"] is not None:
                mismatches.extend(setting_mismatches(case.case_id, scorer, thresholds["cases"][case.case_id], recorded[case.case_id]["notes"]))
        if mismatches:
            problems = mismatches + ["the scored files were made under other scorer settings; use --rerun-scorers"]
    if problems:
        raise SuiteError("Re-score refused:\n  " + "\n  ".join(problems))

    scored_dir = run_dir / f"rescore-{stamp}"
    if rerun_scorers and scored_dir.exists():
        raise SuiteError(f"{scored_dir} already exists; wait a second and run again")
    case_results = [
        rescore_case(case, run_dir, thresholds, recorded[case.case_id], rerun=rerun_scorers, stamp=stamp, scored_dir=scored_dir, run_stage=run_stage)
        for case in selection.cases
    ]
    run_args = dict(old["run_args"])
    run_args["thresholds_file"] = str(Path(thresholds_path).resolve())
    run_args["thresholds_sha256"] = hashlib.sha256(thresholds_bytes).hexdigest()
    verdict = build_verdict(
        old["model_id"],
        selection,
        case_results,
        thresholds_provisional=bool(thresholds.get("provisional", True)),
        run_args=run_args,
        run_timestamp=old["run_timestamp"],
        deviations=old["protocol_deviations"],
    )
    verdict["rescored"] = True
    verdict["rescore"] = {
        "source_run_folder": str(run_dir),
        "source_verdict_sha256": hashlib.sha256((run_dir / VERDICT_NAME).read_bytes()).hexdigest(),
        "thresholds_file": str(Path(thresholds_path).resolve()),
        "thresholds_sha256": run_args["thresholds_sha256"],
        "dropped_not_attempted": current_scope(old["declared_scope"])[1],
        "level": "rerun_scorers" if rerun_scorers else "recorded_scores",
        "rescored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "previous": {
            "safety_gate": old["safety_gate"],
            "smartness_gate": old["smartness_gate"],
            "admitted": old["admitted"],
            "cases": {r["case_id"]: {"status": r["status"], "score": r["score"], "threshold": r["threshold"], "notes": r["notes"]} for r in old["case_results"]},
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    return verdict, out


def format_changes(verdict: dict) -> str:
    """What moved against the verdict the run itself recorded."""
    rescore = verdict["rescore"]
    previous = rescore["previous"]
    lines = [f"Re-score of {rescore['source_run_folder']} ({rescore['level'].replace('_', ' ')}, thresholds {rescore['thresholds_file']})"]
    changed = []
    for r in verdict["case_results"]:
        before = previous["cases"].get(r["case_id"])
        if before and (before["status"] != r["status"] or before["score"] != r["score"]):
            changed.append(f"  {r['case_id']}: {before['status']} ({before['score']}) -> {r['status']} ({r['score']})")
    for gate in ("safety_gate", "smartness_gate", "admitted"):
        if previous[gate] != verdict[gate]:
            changed.append(f"  {gate}: {previous[gate]} -> {verdict[gate]}")
    reworded = [r["case_id"] for r in verdict["case_results"] if r["case_id"] in previous["cases"] and r["notes"] != previous["cases"][r["case_id"]].get("notes", r["notes"])]
    if changed:
        lines.append("Differs from the recorded verdict:")
    elif reworded:
        lines.append(f"Same cases, gates and admitted as the recorded verdict; notes differ for {', '.join(reworded)}.")
    else:
        lines.append("Matches the recorded verdict (cases, gates, admitted, notes).")
    return "\n".join(lines + changed)
