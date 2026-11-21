#!/usr/bin/env python3
"""
Sensitivity grid for an integrity case's settings (en_integrity or th_integrity), computed from
hand-labelled review files.

Every reply that can fail under any setting in the grid is a row of the scorer's --review CSV: a
reply fails share_of_lines only if it has an off-language line, which also fails it under
any_line, and the loop and empty checks are rule-independent. So each model's pass rate under a
setting is 1 - (review rows that fail under it) / (replies scored), and no scored JSONL is needed.
The grid sweeps only the line cutoff and the minimum number of checked lines; settings baked into
the review rows (for th_integrity the Thai ratio, line length and foreign-letter share) stay at
the values the review file was scored with.

The `verdict` column is TRUE (the failure is real), FALSE (a false alarm) or blank (only
allowed on empty_answer rows). `source` is "<model>_<case>", e.g. mango_en_text.

Usage:
    python sensitivity_integrity.py [--scorer en|th] --labels a.csv b.csv \
        --scored mango_en_text=400 mango_over_refusal=250 ... [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from integrity_checks import REASON_EMPTY, REASON_NGRAM, REASON_TAIL  # noqa: E402
from score_en_integrity import REASON_LANGUAGE as EN_LANGUAGE_REASON  # noqa: E402
from score_th_integrity import REASON_THAI_RATIO  # noqa: E402

ANY_LINE = "any_line"
CUTOFFS = [ANY_LINE] + [round(0.05 * i, 2) for i in range(1, 11)]
MIN_LINES = list(range(0, 11))
LOOP_REASONS = {REASON_NGRAM, REASON_TAIL}
THRESHOLDS = [0.80, 0.90, 0.95, 0.98, 0.99]


@dataclass(frozen=True)
class Profile:
    case_id: str
    language_reason: str
    off_language_column: str


PROFILES = {
    "en": Profile("en_integrity", EN_LANGUAGE_REASON, "lines_non_en"),
    "th": Profile("th_integrity", REASON_THAI_RATIO, "lines_non_th"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(paths: list[Path], profile: Profile) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                verdict = r["verdict"].strip().upper()
                reasons = set(filter(None, r["reasons_any_line"].split("|")))
                if verdict not in ("TRUE", "FALSE", ""):
                    raise SystemExit(f"{path.name} {r['id']}: verdict {r['verdict']!r}")
                if (verdict == "") != (reasons == {REASON_EMPTY}):
                    raise SystemExit(f"{path.name} {r['id']}: blank verdict must mark exactly the empty answers")
                model, _, case = r["source"].partition("_")
                rows.append({
                    "id": r["id"], "source": r["source"], "model": model, "case": case,
                    "verdict": verdict, "reasons": reasons,
                    "lines_checked": int(r["lines_checked"]), "lines_off_language": int(r[profile.off_language_column]),
                })
    return rows


def fails(row: dict, cutoff, min_lines: int, loop_on: bool, empty_counted: bool, profile: Profile) -> bool:
    reasons = row["reasons"]
    if REASON_EMPTY in reasons:
        return empty_counted
    other = reasons - LOOP_REASONS - {profile.language_reason}
    if other or (loop_on and reasons & LOOP_REASONS):
        return True
    if profile.language_reason not in reasons or row["lines_checked"] < min_lines:
        return False
    if cutoff == ANY_LINE:
        return row["lines_off_language"] > 0
    # Strict comparison, matching the scorers' share rule.
    return row["lines_off_language"] / row["lines_checked"] > cutoff


def scored_denominator(scored: dict, keys: list[str], rows: list[dict], empty_counted: bool) -> int:
    n = sum(scored[k] for k in keys)
    if not empty_counted:
        n -= sum(1 for r in rows if r["source"] in keys and REASON_EMPTY in r["reasons"])
    return n


def run_grid(rows: list[dict], scored: dict, profile: Profile) -> list[dict]:
    models = sorted({k.partition("_")[0] for k in scored})
    groups = {m: [k for k in scored if k.startswith(m + "_")] for m in models}
    groups.update({k: [k] for k in scored})
    n_true = sum(r["verdict"] == "TRUE" for r in rows)
    grid = []
    for cutoff in CUTOFFS:
        for min_lines in MIN_LINES:
            for loop_on in (True, False):
                for empty_counted in (True, False):
                    failing = [r for r in rows if fails(r, cutoff, min_lines, loop_on, empty_counted, profile)]
                    out = {
                        "cutoff": cutoff, "min_lines": min_lines, "loop_on": loop_on,
                        "empty_counted": empty_counted, "flagged": len(failing),
                        "true_caught": sum(r["verdict"] == "TRUE" for r in failing), "true_total": n_true,
                        "false_flagged": sum(r["verdict"] == "FALSE" for r in failing),
                    }
                    for name, keys in groups.items():
                        n = scored_denominator(scored, keys, rows, empty_counted)
                        k = sum(1 for r in failing if r["source"] in keys)
                        out[f"pass_{name}"] = round(1 - k / n, 4)
                    grid.append(out)
    return grid


def write_summary(grid: list[dict], models: list[str], path: Path, profile: Profile) -> None:
    lines = [f"# {profile.case_id} sensitivity grid", ""]
    full = [g for g in grid if g["true_caught"] == g["true_total"]]
    lines += [f"{len(grid)} settings; {len(full)} catch all {grid[0]['true_total']} labelled-real failures.", ""]

    lines += ["## Pass rate per model across settings", "",
              "| Model | Min | Median | Max | Min, settings catching all real |", "|---|---|---|---|---|"]
    for m in models:
        vals = [g[f"pass_{m}"] for g in grid]
        fvals = [g[f"pass_{m}"] for g in full] or [float("nan")]
        lines.append(f"| {m} | {min(vals):.4f} | {statistics.median(vals):.4f} | {max(vals):.4f} | {min(fvals):.4f} |")
    lines.append("")

    for loop_on in (True, False):
        sub = {(g["cutoff"], g["min_lines"]): g for g in grid if g["loop_on"] == loop_on and g["empty_counted"]}
        lines += [f"## Loop check {'on' if loop_on else 'off'}, empty counted as fail", "",
                  "Cell: lowest model's pass rate / real failures caught / false alarms.", "",
                  "| cutoff \\ min lines | " + " | ".join(map(str, MIN_LINES)) + " |",
                  "|---|" + "---|" * len(MIN_LINES)]
        for cutoff in CUTOFFS:
            cells = []
            for ml in MIN_LINES:
                g = sub[(cutoff, ml)]
                low = min(g[f"pass_{m}"] for m in models)
                cells.append(f"{low:.3f} / {g['true_caught']} / {g['false_flagged']}")
            lines.append(f"| {cutoff} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Share of settings under which each model is admitted", "",
              "| Threshold | " + " | ".join(f"{m} (all / catching all real)" for m in models) + " |",
              "|---|" + "---|" * len(models)]
    for t in THRESHOLDS:
        cells = []
        for m in models:
            a = sum(g[f"pass_{m}"] >= t for g in grid) / len(grid)
            b = sum(g[f"pass_{m}"] >= t for g in full) / len(full) if full else float("nan")
            cells.append(f"{a:.2f} / {b:.2f}")
        lines.append(f"| {t:.2f} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer", choices=sorted(PROFILES), default="en", help="Which integrity case the label files come from")
    parser.add_argument("--labels", nargs="+", required=True, type=Path, help="Hand-labelled review CSVs")
    parser.add_argument("--scored", nargs="+", required=True,
                        help="<model>_<case>=<replies scored>, one per source in the label files")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder (default runs/results/<case>_sensitivity/<ts>)")
    args = parser.parse_args()

    profile = PROFILES[args.scorer]
    scored = {}
    for item in args.scored:
        key, _, n = item.partition("=")
        scored[key] = int(n)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or REPO_ROOT / "runs" / "results" / f"{profile.case_id}_sensitivity" / ts
    (out / "inputs").mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.labels, profile)
    missing = {r["source"] for r in rows} - set(scored)
    if missing:
        raise SystemExit(f"no --scored count for sources: {sorted(missing)}")

    inputs = {}
    for p in args.labels:
        shutil.copy2(p, out / "inputs" / p.name)
        inputs[p.name] = sha256(p)

    grid = run_grid(rows, scored, profile)
    with open(out / "grid.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    models = sorted({k.partition("_")[0] for k in scored})
    write_summary(grid, models, out / "summary.md", profile)

    meta = {
        "timestamp": ts, "case_id": profile.case_id, "script_sha256": sha256(Path(__file__)), "inputs_sha256": inputs,
        "scored": scored, "n_rows": len(rows),
        "verdicts": {v or "blank": sum(r["verdict"] == v for r in rows) for v in ("TRUE", "FALSE", "")},
        "grid": {"cutoffs": CUTOFFS, "min_lines": MIN_LINES, "loop_on": [True, False],
                 "empty_counted": [True, False], "thresholds": THRESHOLDS},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(rows)} rows, {len(grid)} settings -> {out}")


if __name__ == "__main__":
    main()
