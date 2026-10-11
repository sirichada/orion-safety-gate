#!/usr/bin/env python3
"""
Tally pairwise-comparison results from the "Hazard Priority Ranking" survey
(severity + likelihood, MLCommons AILuminate's 12 hazard categories,
"in your field" framing) sent to CMKL faculty for interviews.

Usage:
    python3 tally.py response1.json response2.json ...

Each input file holds one respondent's pasted result block (raw JSON, as
copied off the survey's final "Done" screen) — one file per professor.
Prints a per-respondent severity/likelihood ranking, plus a combined
aggregate ranking across all respondents once more than one is given.
"""

import json
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

CATEGORIES = [
    "violent-crimes", "sex-crimes", "cse", "self-harm", "cbrne", "ip",
    "defamation", "nonviolent-crimes", "hate", "privacy",
    "specialized-advice", "sexual-content",
]
CATEGORY_SET = set(CATEGORIES)

LABELS = {
    "violent-crimes": "Violent Crimes",
    "sex-crimes": "Sex-Related Crimes",
    "cse": "Child Sexual Exploitation",
    "self-harm": "Suicide & Self-Harm",
    "cbrne": "CBRNE",
    "ip": "IP Violations",
    "defamation": "Defamation",
    "nonviolent-crimes": "Non-Violent Crimes",
    "hate": "Hate",
    "privacy": "Privacy Violations",
    "specialized-advice": "Specialized Advice",
    "sexual-content": "Sexual Content",
}

EXPECTED_PAIRS = len(CATEGORIES) * (len(CATEGORIES) - 1) // 2  # 66
LOW_N_THRESHOLD = 6  # fewer comparisons than this -> flag the rate as low-confidence

CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
SMART_QUOTES = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
})


def parse_pasted_json(raw_text, source_label):
    """Parse a hand-pasted JSON blob, tolerating common paste artifacts
    (``` code fences from Slack/Markdown, smart quotes from some clients).
    Returns the parsed object, or raises ValueError with a clear message."""
    text = CODE_FENCE_RE.sub("", raw_text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text.translate(SMART_QUOTES))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{source_label}: couldn't parse as JSON even after stripping "
            f"code fences / normalizing smart quotes ({e}). Check the paste "
            f"wasn't truncated or mixed with other text."
        )


def parse_completed_at(ts):
    """Best-effort parse of an ISO timestamp for ordering resubmissions.
    Unparseable/missing timestamps sort as the oldest possible value, so a
    record with a real timestamp always beats one without in a tie-break."""
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def dedupe(choices, respondent, round_name, skipped_count, warnings):
    """Validate and keep only the first valid answer for each unique
    unordered pair. Reports (does not silently drop) invalid entries and
    duplicate re-answers, and explains any gap between the answered count
    and the full 66-pair round robin."""
    if not isinstance(choices, list):
        warnings.append(f"{respondent}: {round_name} 'choices' wasn't a list — treated as empty.")
        choices = []

    seen = set()
    out = []
    dropped_invalid = 0
    dropped_duplicate = 0
    for c in choices:
        if not isinstance(c, dict):
            dropped_invalid += 1
            continue
        winner, loser = c.get("winner"), c.get("loser")
        if winner not in CATEGORY_SET or loser not in CATEGORY_SET:
            dropped_invalid += 1
            continue
        if winner == loser:
            dropped_invalid += 1
            continue
        key = tuple(sorted([winner, loser]))
        if key in seen:
            dropped_duplicate += 1
            continue
        seen.add(key)
        out.append({"winner": winner, "loser": loser})

    if not isinstance(skipped_count, int) or isinstance(skipped_count, bool):
        if skipped_count is not None:
            warnings.append(f"{respondent}: {round_name} 'skipped' field wasn't a number — ignored.")
        skipped_count = 0

    if dropped_invalid:
        warnings.append(
            f"{respondent}: {round_name} had {dropped_invalid} invalid answer(s) "
            f"(unknown category, winner==loser, or a malformed entry) — dropped."
        )
    if dropped_duplicate:
        warnings.append(
            f"{respondent}: {round_name} had {dropped_duplicate} duplicate "
            f"re-answer(s) of an already-seen pair — kept the first, dropped the rest."
        )

    missing = EXPECTED_PAIRS - len(out)
    # Explain the gap in priority order so the same shortfall isn't blamed
    # on two different causes at once: invalid entries (already reported
    # above) first, then declared skips, then whatever's left over.
    explained_by_invalid = min(dropped_invalid, missing) if missing > 0 else 0
    remaining = missing - explained_by_invalid
    explained_by_skip = min(max(skipped_count, 0), remaining) if remaining > 0 else 0
    unexplained = remaining - explained_by_skip

    if explained_by_skip:
        warnings.append(f"{respondent}: {round_name} — {explained_by_skip} pair(s) intentionally skipped.")
    if unexplained:
        warnings.append(
            f"{respondent}: {round_name} — {unexplained} pair(s) missing with no "
            f"matching skip — likely a truncated or incomplete paste."
        )
    if skipped_count > remaining:
        warnings.append(
            f"{respondent}: {round_name} — record claims {skipped_count} skipped "
            f"pair(s) but only {remaining} pair(s) are actually unaccounted for — "
            f"skip count looks inconsistent with the choices list."
        )
    return out


def count_choices(choices):
    wins = {c: 0 for c in CATEGORIES}
    appearances = {c: 0 for c in CATEGORIES}
    for c in choices:
        wins[c["winner"]] += 1
        appearances[c["winner"]] += 1
        appearances[c["loser"]] += 1
    return wins, appearances


def competition_rank(rate_by_category):
    """rate_by_category: {category: Fraction|None}. Returns (rank_num,
    category) rows with ties sharing a rank. Rates are exact Fractions, so
    two genuinely-equal rates never diverge from float summation error."""
    ordered = sorted(
        CATEGORIES,
        key=lambda c: (rate_by_category[c] is None, -(rate_by_category[c] or Fraction(0))),
    )
    rows = []
    prev = object()  # sentinel that != any real rate or None
    rank_num = 0
    for i, c in enumerate(ordered, 1):
        r = rate_by_category[c]
        if r != prev:
            rank_num = i
            prev = r
        rows.append((rank_num, c))
    return rows


def rank(choices):
    wins, appearances = count_choices(choices)
    rate = {
        c: (Fraction(wins[c], appearances[c]) if appearances[c] else None)
        for c in CATEGORIES
    }
    return [(rn, c, wins[c], appearances[c], rate[c]) for rn, c in competition_rank(rate)]


def print_ranking(title, rows):
    print(f"\n{title}")
    for rank_num, cat, w, t, r in rows:
        if r is None:
            cell = "n/a"
        else:
            cell = f"{100 * float(r):.1f}%"
            if t < LOW_N_THRESHOLD:
                cell += "  (low n)"
        print(f"{rank_num:2d}. {LABELS[cat]:28s} {w}/{t}  {cell}")


def mean_of_rates(per_respondent_choices):
    """Mean win rate per category across respondents, each respondent
    weighted equally (as opposed to pooling raw choices, where a
    respondent who answered fewer pairs counts for less)."""
    sums = {c: Fraction(0) for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    for choices in per_respondent_choices:
        wins, appearances = count_choices(choices)
        for c in CATEGORIES:
            if appearances[c]:
                sums[c] += Fraction(wins[c], appearances[c])
                counts[c] += 1
    rate = {c: (sums[c] / counts[c] if counts[c] else None) for c in CATEGORIES}
    return [(rn, c, counts[c], rate[c]) for rn, c in competition_rank(rate)]


def print_mean_ranking(title, rows, n_respondents):
    print(f"\n{title}")
    for rank_num, cat, n, r in rows:
        cell = f"{100 * float(r):.1f}%" if r is not None else "n/a"
        print(f"{rank_num:2d}. {LABELS[cat]:28s} (in {n}/{n_respondents} respondents)  {cell}")


def load_record(path, warnings):
    with open(path, encoding="utf-8") as f:
        raw_text = f.read()
    record = parse_pasted_json(raw_text, str(path))

    if not isinstance(record, dict):
        raise ValueError(f"{path}: top-level JSON was a {type(record).__name__}, expected an object.")

    raw_name = record.get("name")
    stripped_name = raw_name.strip() if isinstance(raw_name, str) else ""
    name = stripped_name if stripped_name else Path(path).stem
    norm_name = " ".join(name.split()).lower()  # collapse internal whitespace too
    respondent_id = record.get("respondentId")
    respondent_id = respondent_id if isinstance(respondent_id, str) and respondent_id else None
    completed_at = record.get("completedAt")

    severity_block = record.get("severity")
    likelihood_block = record.get("likelihood")
    severity_block = severity_block if isinstance(severity_block, dict) else {}
    likelihood_block = likelihood_block if isinstance(likelihood_block, dict) else {}

    sev = dedupe(severity_block.get("choices", []), name, "severity", severity_block.get("skipped", 0), warnings)
    lik = dedupe(likelihood_block.get("choices", []), name, "likelihood", likelihood_block.get("skipped", 0), warnings)

    return {
        "path": str(path),
        "id": respondent_id,
        "norm_name": norm_name,
        "name": name,
        "completed_at": completed_at,
        "sev": sev,
        "lik": lik,
    }


def resolve_duplicates(records, warnings):
    """Groups records that share a respondentId OR a normalized name via
    union-find over (id, name) identity keys, then keeps the
    latest-completedAt record per group. Union-find handles the case
    where a record bridges two otherwise-unconnected id/name pairs — e.g.
    A=(id1,"jane"), B=(id2,"j. doe"), C=(id2,"jane"): C shares an id with
    B and a name with A, so all three end up in one group regardless of
    input order."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rec in records:
        name_key = ("name", rec["norm_name"])
        find(name_key)
        if rec["id"]:
            union(("id", rec["id"]), name_key)

    groups = {}
    for rec in records:
        key = ("id", rec["id"]) if rec["id"] else ("name", rec["norm_name"])
        groups.setdefault(find(key), []).append(rec)

    resolved = []
    for members in groups.values():
        if len(members) == 1:
            resolved.append(members[0])
            continue

        times = [parse_completed_at(m["completed_at"]) for m in members]
        max_time = max(times)
        chosen = members[times.index(max_time)]
        tied_at_max = sum(1 for t in times if t == max_time)

        id_values = {m["id"] for m in members if m["id"]}
        detail = (
            "different respondentIds, same name — verify manually"
            if len(id_values) > 1 else "matched by respondentId or name"
        )
        timing = (
            "timestamps tie or are both missing/unparseable — kept the first file seen among the tied ones"
            if tied_at_max > 1 else f"kept the later submission ({chosen['path']})"
        )
        paths = ", ".join(m["path"] for m in members)
        warnings.append(
            f"{chosen['name']}: {paths} look like the same respondent "
            f"({detail}) — {timing}."
        )
        resolved.append(chosen)
    return resolved


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    warnings = []
    loaded = []
    for path in sys.argv[1:]:
        try:
            loaded.append(load_record(path, warnings))
        except (ValueError, OSError) as e:
            warnings.append(f"{path}: couldn't load — {e}")

    respondents = resolve_duplicates(loaded, warnings)

    n_files = len(sys.argv) - 1
    print(
        f"Loaded {len(respondents)} respondent(s) from {n_files} file(s)"
        + (f" — {len(warnings)} warning(s), see below." if warnings else ".")
    )

    for rec in respondents:
        print(f"\n{'=' * 60}\n{rec['name']}\n{'=' * 60}")
        print_ranking("Severity (in their field)", rank(rec["sev"]))
        print_ranking("Likelihood (in their field)", rank(rec["lik"]))

    if warnings:
        print(f"\n\n{'-' * 60}\nWarnings\n{'-' * 60}")
        for w in warnings:
            print(f"  [warning] {w}")

    if not respondents:
        print("\nNo valid respondents loaded — nothing to tally.")
        sys.exit(1)

    if len(respondents) > 1:
        all_sev = [c for r in respondents for c in r["sev"]]
        all_lik = [c for r in respondents for c in r["lik"]]
        n = len(respondents)

        print(f"\n\n{'#' * 60}\nCOMBINED ACROSS {n} RESPONDENTS\n{'#' * 60}")
        print_ranking("Severity — pooled (each answered pair counted equally)", rank(all_sev))
        print_ranking("Likelihood — pooled (each answered pair counted equally)", rank(all_lik))
        print_mean_ranking(
            "Severity — mean of per-respondent rates (each respondent counted equally)",
            mean_of_rates([r["sev"] for r in respondents]), n,
        )
        print_mean_ranking(
            "Likelihood — mean of per-respondent rates (each respondent counted equally)",
            mean_of_rates([r["lik"] for r in respondents]), n,
        )


if __name__ == "__main__":
    main()
