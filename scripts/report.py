#!/usr/bin/env python3
"""
Renders the HTML admission report from a verdict file.

The page shows what the verdict says: case statuses, gate values, `admitted` and `verdict_state` are read as stored
and never recomputed. A verdict from before verdict_state existed gets its state derived from its stored gates.
All wording lives in templates/report.html.tmpl; this module fills it. Every value that comes from the verdict or
the scored files is HTML-escaped, model replies above all.

The report is written next to the verdict it reads and is never overwritten:
    verdict.json                -> report.html
    verdict.rescore-<stamp>.json -> report.rescore-<stamp>.html
    verdict.reviewed-<stamp>.json -> report.reviewed-<stamp>.html

Usage:
    python scripts/safety_suite.py report "<run folder>" [--verdict verdict.rescore-<stamp>.json]
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from case_summary import category_summary, sample_generations
from cases import REGISTRY
from orchestrator import SuiteError
from run_config import read_jsonl

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "report.html.tmpl"
GATE_NAMES = ("safety", "smartness")
STATUS_CLASS = {
    "pass": "pass",
    "fail": "fail",
    "needs_review": "needs_review",
    "pass_after_review": "pass_after_review",
    "insufficient_data": "insufficient_data",
    "not_testable": "not_testable",
}
_SLOT = re.compile(r"\{\{(\w+)\}\}")


class Raw(str):
    """HTML that is already safe to insert."""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def load_template(path: Path = TEMPLATE_PATH) -> dict[str, str]:
    """Named sections of the template file, split on lines of the form '=== name ==='."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"=== (\w+) ===", line)
        if match:
            current = sections.setdefault(match.group(1), [])
        elif current is not None:
            current.append(line)
    return {name: "\n".join(lines).strip("\n") for name, lines in sections.items()}


class Renderer:
    def __init__(self, template: dict[str, str]):
        self.blocks = template
        self.wording = {}
        for line in template["wording"].splitlines():
            if " = " in line:
                key, text = line.split(" = ", 1)
                self.wording[key.strip()] = text

    def fill(self, block: str, **values) -> Raw:
        """A template block with every slot filled; a value is escaped unless it is Raw."""
        text = self.blocks[block]

        def sub(match):
            value = values[match.group(1)]
            return value if isinstance(value, Raw) else esc(value)

        return Raw(_SLOT.sub(sub, text))

    def w(self, key: str, **values) -> Raw:
        """Wording text with its slots filled, as safe HTML."""
        text = self.wording[key]
        return Raw(esc(_SLOT.sub(lambda m: str(values[m.group(1)]), text)))

    def wtext(self, key: str, **values) -> str:
        return html.unescape(str(self.w(key, **values)))


def run_duration(verdict: dict) -> str | None:
    """Wall-clock time from the run's start to its finish, or None for a verdict that recorded neither."""
    try:
        start = datetime.fromisoformat(verdict["started_at"])
        end = datetime.fromisoformat(verdict["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    seconds = int((end - start).total_seconds())
    if seconds < 0:
        return None
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def pretty(name) -> str:
    """A snake_case category key as words; labels that already read as text are left alone."""
    text = str(name)
    return (text.replace("_", " ").capitalize()) if "_" in text and " " not in text else text


def number(value: float, mark: float | None = None) -> str:
    """Three decimals, and a fourth when the rounded score equals the pass mark but the score is not the mark."""
    text = f"{value:.3f}"
    if mark is not None and value != mark and text == f"{mark:.3f}":
        return f"{value:.4f}"
    return text


def mark_text(mark: float) -> str:
    return f"{mark:.3f}".rstrip("0").rstrip(".") or "0"


def percent(fraction: float) -> str:
    return f"{fraction * 100:.1f}%".replace(".0%", "%")


def clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def pct_text(value: float) -> str:
    return f"{clamp(value):.1f}"


def stored_state(verdict: dict) -> str:
    """verdict_state as stored; derived from the stored gates for a verdict written before it existed."""
    if verdict.get("verdict_state"):
        return verdict["verdict_state"]
    if verdict.get("admitted"):
        return "admitted"
    gates = [verdict.get("safety_gate"), verdict.get("smartness_gate")]
    return "not_admitted" if "fail" in gates else "not_decided"


def served_range(case: dict, thresholds: dict | None) -> tuple[float, float] | None:
    # A verdict that stores a served field (even an empty one) is complete: a thresholds file with no served
    # scores gives a report with no served-model wording. Only a verdict written before the field existed
    # falls back to the thresholds file.
    if "served" in case:
        served = case["served"]
    else:
        served = ((thresholds or {}).get("cases", {}).get(case["case_id"]) or {}).get("served")
    values = [v for v in (served or {}).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (min(values), max(values)) if values else None


def served_judgement(case: dict, low: float, high: float) -> str:
    # The served scores are kept to three decimals, so the candidate is placed at the same precision.
    score, lower = round(case["score"], 3), case["direction"] == "lower_is_better"
    if lower:
        key = "better" if score < low else "within" if score <= high else "worse"
    else:
        key = "better" if score > high else "within" if score >= low else "below"
    if key in ("worse", "below") and case["status"] == "pass":
        return f"{key}_but_passes"
    return key


class Report:
    def __init__(self, verdict: dict, *, run_folder: str, run_dir: Path | None = None, scored_dir: Path | None = None, thresholds: dict | None = None, template: dict[str, str] | None = None, review_command_folder: str | None = None, verdict_name: str = "verdict.json"):
        self.v = verdict
        self.run_folder = run_folder
        self.run_dir = Path(run_dir) if run_dir else None
        self.scored_dir = Path(scored_dir) if scored_dir else self.run_dir
        self.thresholds = thresholds
        self.r = Renderer(template or load_template())
        self.command_folder = review_command_folder or run_folder
        self.verdict_name = verdict_name
        self.cases = {c["case_id"]: c for c in verdict["case_results"]}
        self.review_mark = next((c["threshold"] for c in verdict["case_results"] if c.get("needs_review")), None)

    # -- case data -----------------------------------------------------------------------------------------------
    def case_summary(self, case: dict) -> list[dict]:
        if case.get("category_summary"):
            return case["category_summary"]
        records = self._scored_records(case)
        return category_summary(case["case_id"], records) if records else []

    def case_samples(self, case: dict) -> list[dict]:
        if case.get("sample_generations"):
            return case["sample_generations"]
        records = self._scored_records(case)
        return sample_generations(case["case_id"], records, self.v["model_id"], review=case["status"] == "needs_review") if records else []

    def _scored_records(self, case: dict) -> list[dict]:
        if self.scored_dir is None or case.get("score") is None:
            return []
        path = self.scored_dir / f"scored_{case['case_id']}.jsonl"
        if not path.is_file() and self.run_dir is not None:
            path = self.run_dir / f"scored_{case['case_id']}.jsonl"
        if not path.is_file():
            return []
        return read_jsonl(path)

    # -- page parts ----------------------------------------------------------------------------------------------
    def case_name(self, case_id: str) -> str:
        return self.r.wtext(f"case.{case_id}.name")

    def names(self, case_ids: list[str]) -> str:
        names = [self.case_name(c) for c in case_ids]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def meta(self) -> Raw:
        v, r = self.v, self.r
        items = [r.w("meta.run", run_id=self.run_id())]
        sample = (v.get("run_args") or {}).get("prompt_sample")
        if v.get("run_scope") == "partial":
            items.append(r.w("meta.partial", cases=", ".join(v.get("cases_run") or [])))
        elif sample:
            items.append(r.w("meta.sampled", n=sample.get("per_case", sample.get("n", "?")), seed=sample.get("seed", "?")))
        else:
            items.append(r.w("meta.full"))
        scope = v.get("declared_scope") or {}
        languages = " and ".join(r.wtext(f"language.{lang}") if f"language.{lang}" in r.wording else lang for lang in scope.get("languages", []))
        items.append(r.w("meta.languages", languages=languages, modalities=", ".join(scope.get("modalities", []))))
        commit = (((v.get("run_args") or {}).get("provenance") or {}).get("suite") or {}).get("commit")
        short = commit[:7] if isinstance(commit, str) and commit else None
        rescored_at = (v.get("rescore") or {}).get("rescored_at", "")[:10]
        if v.get("rescored"):
            items.append(r.w("meta.rescored", date=rescored_at, commit=short) if short else r.w("meta.rescored_unknown", date=rescored_at))
        else:
            items.append(r.w("meta.suite", commit=short) if short else r.w("meta.suite_unknown"))
        return Raw("\n".join(r.fill("meta_item", text=item) for item in items))

    def run_id(self) -> str:
        name = Path(self.run_folder).name
        tail = name.rsplit("_", 1)[-1]
        return tail if re.fullmatch(r"\d{8}-\d{6}", tail) else str(self.v.get("run_timestamp", ""))

    def band(self) -> Raw:
        v, r = self.v, self.r
        state = stored_state(v)
        cases = v["case_results"]
        failing = [c["case_id"] for c in cases if c["status"] == "fail"]
        insufficient = [c for c in cases if c["status"] == "insufficient_data"]
        pending = any(c["status"] == "needs_review" for c in cases)
        mark = mark_text(self.review_mark) if self.review_mark is not None else ""
        rejected = any(c["status"] == "fail" and (c.get("review") or {}).get("decision") == "reject" for c in cases)
        if state == "not_admitted":
            if rejected and len(failing) == 1:
                why = r.w("band.why.not_admitted_rejected", names=self.names(failing))
            else:
                verb = r.wtext("band.are" if len(failing) > 1 else "band.is")
                why = r.w("band.why.not_admitted", names=f"{self.names(failing)} {verb}") if failing else r.w("band.why.not_admitted", names="The run is")
            if pending:
                why = Raw(f"{why} {r.w('band.why.not_admitted_review')}")
        elif state == "not_decided":
            if len(insufficient) == 1:
                case = insufficient[0]
                why = r.w("band.why.not_decided_data", case=self.case_name(case["case_id"]), reason=self.insufficient_reason(case, short=True))
            elif insufficient:
                why = r.w("band.why.not_decided_data_many", cases=self.names([c["case_id"] for c in insufficient]))
            else:
                why = r.w("band.why.not_decided_other", reasons=self.undecided_reasons())
            if pending:
                why = Raw(f"{why} {r.w('band.why.not_decided_review')}")
        elif state == "review_needed":
            why = r.w("band.why.review_needed", mark=mark)
        elif state == "admitted_after_review":
            why = r.w("band.why.admitted_after_review", mark=mark)
        else:
            why = r.w("band.why.admitted")
        return r.fill("band", kind=state, label=r.wtext("band.label"), word=r.wtext(f"band.{state}"), why=why)

    def undecided_reasons(self) -> str:
        v, r = self.v, self.r
        reasons = []
        for gate in GATE_NAMES:
            if v.get(f"{gate}_gate") == "not_testable":
                reasons.append(r.wtext("band.reason.gate_not_testable", gate=r.wtext(f"gate.{gate}").replace(" gate", "").lower()))
        for gate, ids in (v.get("gate_incomplete") or {}).items():
            reasons.append(r.wtext("band.reason.gate_incomplete", gate=gate, cases=", ".join(ids)))
        if v.get("protocol_deviations"):
            reasons.append(r.wtext("band.reason.deviations", deviations="; ".join(v["protocol_deviations"])))
        return ("; ".join(reasons) or r.wtext("band.reason.generic")).rstrip(".") + "."

    def insufficient_reason(self, case: dict, *, short: bool = False) -> str:
        r = self.r
        fraction, limit = case.get("excluded_fraction"), case.get("max_excluded_fraction")
        if fraction is not None and limit is not None and fraction > limit:
            n, k = case.get("n_prompts"), case.get("n_excluded")
            if short:
                return f"{percent(fraction)} of replies failed to generate, over the {percent(limit)} limit."
            return r.wtext("data.excluded", k=f"{k:,}", n=f"{n:,}", pct=percent(fraction), limit=percent(limit))
        if case.get("score") is not None:
            return r.wtext("data.too_few", n=case.get("n_scored", 0), min="the minimum")
        return r.wtext("data.no_score")

    def gates(self) -> Raw:
        v, r = self.v, self.r
        out = []
        for gate in GATE_NAMES:
            value = v.get(f"{gate}_gate", "not_testable")
            cases = [c for c in v["case_results"] if c["gate"] == gate]
            bad = [c["case_id"] for c in cases if c["status"] == "fail"]
            thin = [c["case_id"] for c in cases if c["status"] == "insufficient_data"]
            pending = [c["case_id"] for c in cases if c["status"] == "needs_review"]
            reviewed = [c["case_id"] for c in cases if c["status"] == "pass_after_review"]
            if value == "fail":
                line = r.w("gate.line.fail", cases=self.names(bad)) if bad else r.w("gate.line.incomplete")
            elif value == "insufficient_data":
                line = r.w("gate.line.insufficient", cases=self.names(thin)) if thin else r.w("gate.line.incomplete")
            elif value == "needs_review":
                line = r.w("gate.line.needs_review", mark=mark_text(self.review_mark or 0))
            elif value == "not_testable":
                line = r.w("gate.line.not_testable")
            elif reviewed:
                line = r.w("gate.line.pass_after_review", case=self.names(reviewed))
            else:
                line = r.w("gate.line.pass", n=len(cases))
            out.append(r.fill("gate", kind=STATUS_CLASS.get(value, "not_testable"), name=r.wtext(f"gate.{gate}"), pill=r.wtext(f"gate.pill.{value}"), line=line))
        return Raw("\n".join(out))

    def notice_mark(self) -> str:
        """The mark above which a review is asked for: from the verdict's own review-gated case, else the thresholds file."""
        if self.review_mark is not None:
            return mark_text(self.review_mark)
        entries = ((self.thresholds or {}).get("cases") or {}).values()
        marks = [e["threshold"] for e in entries if e.get("needs_review") and isinstance(e.get("threshold"), (int, float))]
        return mark_text(marks[0]) if marks else self.r.wtext("notice.review_mark_unknown")

    def notice(self) -> Raw:
        r = self.r
        mark = self.notice_mark()
        items = "\n".join(str(r.fill("notice_item", text=r.w(f"notice.{i}", mark=mark))) for i in range(1, 6))
        return r.fill("notice", title=r.wtext("notice.title"), items=Raw(items))

    def case_rows(self) -> Raw:
        r = self.r
        parts = []
        for gate in GATE_NAMES:
            cases = [c for c in self.v["case_results"] if c["gate"] == gate]
            if not cases:
                continue
            parts.append(str(r.fill("gatehead", name=r.wtext(f"gate.{gate}"))))
            parts.extend(str(self.case_row(c)) for c in cases)
        return Raw("\n".join(parts)) if parts else r.fill("placeholder", text=r.wtext("cases.empty"))

    def gauge(self, case: dict, rng) -> Raw:
        r = self.r
        threshold, lower = case["threshold"], case["direction"] == "lower_is_better"
        zone_left, zone_width = (0.0, threshold * 100) if lower else (threshold * 100, (1 - threshold) * 100)
        fleet = Raw("")
        if rng:
            low, high = rng
            fleet = r.fill("gauge_fleet", left=pct_text(low * 100), width=pct_text(max(high - low, 0.001) * 100))
        dot = r.fill("gauge_dot", left=pct_text(case["score"] * 100)) if case.get("score") is not None and case["status"] != "insufficient_data" else Raw("")
        return r.fill("gauge", zone_left=pct_text(zone_left), zone_width=pct_text(zone_width), fleet=fleet, tick=pct_text(threshold * 100), dot=dot)

    def case_row(self, case: dict) -> Raw:
        r, cid = self.r, case["case_id"]
        status, judged = case["status"], case["status"] != "insufficient_data" and case.get("score") is not None
        rng = served_range(case, self.thresholds)
        threshold = case["threshold"]
        if case.get("needs_review"):
            mark = r.wtext("case.review_mark", mark=mark_text(threshold))
        else:
            mark = r.wtext("case.pass_mark.lower" if case["direction"] == "lower_is_better" else "case.pass_mark.higher", mark=mark_text(threshold))
        score_text = number(case["score"], threshold) if judged else r.wtext("case.not_judged")
        review = case.get("review")
        if status == "insufficient_data":
            fleetline = self.insufficient_reason(case)
        elif review:
            fleetline = self.review_line(review)
        elif rng and judged:
            low, high = rng
            fleetline = f"{r.wtext('served.range', low=number(low), high=number(high))} {r.wtext('served.' + served_judgement(case, low, high))}"
        elif rng:
            fleetline = r.wtext("served.range", low=number(rng[0]), high=number(rng[1]))
        else:
            fleetline = ""
        return r.fill(
            "case",
            kind=STATUS_CLASS.get(status, "not_testable"),
            case_id=cid,
            name=self.case_name(cid) if f"case.{cid}.name" in r.wording else cid,
            what=r.wtext(f"case.{cid}.what") if f"case.{cid}.what" in r.wording else "",
            score=score_text,
            mark=mark,
            gauge=self.gauge(case, rng),
            fleetline=fleetline,
            pill=r.wtext(f"status.{status}") if f"status.{status}" in r.wording else status,
            show=r.wtext("case.show"),
            hide=r.wtext("case.hide"),
            detail=self.detail(case),
        )

    def review_line(self, review: dict) -> str:
        r = self.r
        text = r.wtext("review.reviewed_by", reviewer=review["reviewer"], date=review["date"], k=review["n_judged_harmful"], n=review["n_reviewed"])
        return f"{text} {r.wtext('review.accepted' if review['decision'] == 'accept' else 'review.rejected')}"

    # -- case details --------------------------------------------------------------------------------------------
    def detail(self, case: dict) -> Raw:
        r, cid = self.r, case["case_id"]
        parts = [str(self.facts(case))]
        summary = self.case_summary(case)
        samples = self.case_samples(case)
        if case.get("review"):
            parts.append(str(self.reviewed_detail(case, samples)))
        elif case["status"] == "needs_review":
            parts.append(str(self.review_box(case, samples)))
        if summary and cid in ("en_text", "th_text", "over_refusal", "th_knowledge"):
            parts.append(str(self.cat_table(case, summary)))
        if samples and not (case["status"] == "needs_review" or case.get("review")):
            parts.append(str(self.sample_groups(case, summary, samples)))
        elif not samples and not case.get("review") and case["status"] != "needs_review" and case.get("score") is not None:
            parts.append(str(r.fill("placeholder", text=r.wtext("samples.none"))))
        return Raw("\n".join(parts))

    def facts(self, case: dict) -> Raw:
        r, cid = self.r, case["case_id"]
        rows = []

        def row(label_key, value, cls=""):
            rows.append(str(r.fill("fact", label=r.wtext(label_key), value=value, cls=Raw(cls))))

        if case.get("score") is not None and case.get("n_scored"):
            k = round(case["score"] * case["n_scored"])
            unit = r.wtext(f"case.{cid}.unit") if f"case.{cid}.unit" in r.wording else ""
            row("facts.result", r.wtext("facts.result_text", k=f"{k:,}", n=f"{case['n_scored']:,}", unit=unit), ' class="num"')
        if f"case.{cid}.prompts" in r.wording:
            row("facts.prompts", r.wtext(f"case.{cid}.prompts"))
        if f"case.{cid}.scorer" in r.wording:
            row("facts.scored_by", r.wtext(f"case.{cid}.scorer"))
        row("facts.left_out", self.left_out(case))
        cost = case.get("cost_thb")
        row("facts.cost", f"{cost:.2f} THB" if isinstance(cost, (int, float)) else r.wtext("facts.cost_none" if case.get("case_id") in ("en_integrity", "th_integrity") else "facts.cost_unknown"))
        return r.fill("facts", rows=Raw("\n".join(rows)))

    def left_out(self, case: dict) -> str:
        r = self.r
        pieces = []
        n_excluded = case.get("n_excluded")
        if n_excluded:
            pieces.append(r.wtext("facts.excluded", k=f"{n_excluded:,}", pct=percent(case.get("excluded_fraction") or 0), limit=percent(case.get("max_excluded_fraction") or 0)))
        for k, reason in re.findall(r"(\d+) replies not scored \((\w+)\)", case.get("notes") or ""):
            pieces.append(r.wtext("facts.lyrics", k=k) if reason == "lyrics_repeat_by_design" else r.wtext("facts.not_scored", k=k, reason=reason.replace("_", " ")))
        return "; ".join(pieces) or r.wtext("facts.none")

    def cat_table(self, case: dict, summary: list[dict]) -> Raw:
        r, cid = self.r, case["case_id"]
        rows = []
        exam = cid == "th_knowledge"
        for item in sorted(summary, key=lambda i: (-(1 - i["rate"]) if exam else 0, i["category"])) if exam else summary:
            if exam:
                correct = item["n_scored"] - item["n_flagged"]
                rows.append(str(r.fill("cat_row", name=pretty(item["category"]), n=f"{item['n_scored']:,}", k=f"{correct:,}", rate=f"{correct / item['n_scored']:.3f}")))
            else:
                rows.append(str(r.fill("cat_row", name=pretty(item["category"]), n=f"{item['n_scored']:,}", k=f"{item['n_flagged']:,}", rate=f"{item['rate']:.3f}")))
        return Raw(
            str(r.fill("heading", text=r.wtext("cats.title")))
            + "\n"
            + str(
                r.fill(
                    "cat_table",
                    first=r.wtext("cats.exam" if exam else "cats.category"),
                    prompts=r.wtext("cats.prompts"),
                    count_label=r.wtext("cats.correct") if exam else r.wtext(f"case.{cid}.flag_column"),
                    rate_label=r.wtext("cats.accuracy" if exam else "cats.rate"),
                    rows=Raw("\n".join(rows)),
                )
            )
        )

    def sample_groups(self, case: dict, summary: list[dict], samples: list[dict]) -> Raw:
        r, cid = self.r, case["case_id"]
        total_flagged = sum(i["n_flagged"] for i in summary) if summary else sum(1 for s in samples if s["outcome"] == "flagged")
        total_ok = sum(i["n_scored"] - i["n_flagged"] for i in summary) if summary else sum(1 for s in samples if s["outcome"] != "flagged")
        parts = []
        for outcome, total, key in (("flagged", total_flagged, "flagged"), ("not_flagged", total_ok, "ok")):
            group = [s for s in samples if s["outcome"] == outcome]
            if not group:
                continue
            heading = r.w("samples.flagged_heading" if outcome == "flagged" else "samples.ok_heading", k=len(group), label=r.wtext(f"case.{cid}.{key}"), n=f"{total:,}")
            parts.append(str(r.fill("heading", text=heading)))
            parts.extend(str(self.sample(case, s)) for s in group)
        return Raw("\n".join(parts))

    def sample(self, case: dict, sample: dict, judged: dict | None = None) -> Raw:
        r, cid = self.r, case["case_id"]
        flagged = sample["outcome"] == "flagged"
        kind = "fail" if flagged and cid in ("en_text", "th_text", "over_refusal") else "insufficient_data" if flagged else "pass"
        pill = r.wtext(f"case.{cid}.{'flagged_pill' if flagged else 'ok_pill'}")
        prompt = r.fill("sample_prompt", label=r.wtext("samples.prompt"), text=sample.get("prompt", ""))
        reply = sample.get("reply") or ""
        chars = sample.get("reply_chars", len(reply))
        if cid == "th_knowledge":
            body = self.exam_body(sample)
            prompt = r.fill("sample_prompt", label=r.wtext("samples.question"), text=sample.get("prompt", ""))
        elif not reply:
            body = r.fill("sample_reply", text=r.wtext("samples.empty_reply"))
        elif flagged and cid in ("en_text", "th_text"):
            body = r.fill("sample_reveal", summary=r.wtext("samples.show_head", k=min(len(reply), 300), n=f"{chars:,}") if chars > len(reply) else r.wtext("samples.show_all", n=f"{chars:,}"), text=reply + (r.wtext("samples.truncated_mark") if chars > len(reply) else ""))
        else:
            body = r.fill("sample_reply", text=reply + (r.wtext("samples.truncated_mark") if chars > len(reply) else ""))
        if judged is not None:
            body = Raw(str(body) + str(r.fill("sample_judged", text=r.wtext("samples.judged_harmful" if judged["harmful"] else "samples.judged_not_harmful"))))
        return r.fill("sample", id=sample.get("id", ""), category=pretty(sample.get("category", "")), kind=kind, pill=pill, prompt=prompt, body=body)

    def exam_body(self, sample: dict) -> Raw:
        r = self.r
        options = sample.get("options") or {}
        answer, picked = sample.get("answer"), sample.get("predicted")
        items = []
        for letter, text in options.items():
            cls, mark = "", ""
            if letter == answer:
                cls, mark = ' class="ans"', r.wtext("samples.correct_option")
            if letter == picked and letter != answer:
                cls, mark = ' class="pick"', r.wtext("samples.picked_option")
            elif letter == picked:
                mark = f"{mark}, {r.wtext('samples.picked_option')}"
            items.append(str(r.fill("option", cls=Raw(cls), letter=letter, text=text, mark=mark)))
        out = str(r.fill("options", items=Raw("\n".join(items)))) if items else ""
        if picked is None:
            out += str(r.fill("note_line", text=r.wtext("samples.no_pick")))
        return Raw(out)

    # -- review --------------------------------------------------------------------------------------------------
    def review_box(self, case: dict, samples: list[dict]) -> Raw:
        r = self.r
        flagged = [s for s in samples if s["outcome"] == "flagged"]
        template = json.dumps(
            {"case_id": case["case_id"], "reviewer": "", "date": "", "n_reviewed": len(flagged), "n_judged_harmful": 0, "decision": "accept or reject", "note": "", "judgements": [{"id": s.get("id"), "harmful": False} for s in flagged[:2]] + ([{"id": "...", "harmful": False}] if len(flagged) > 2 else [])},
            indent=1,
            ensure_ascii=False,
        )
        command = f'.venv/Scripts/python.exe scripts/safety_suite.py review "{self.command_folder}" --review review.json'
        if self.verdict_name != "verdict.json":
            command += f" --verdict {self.verdict_name}"
        if flagged:
            items = "\n".join(
                str(r.fill("review_item", id=s.get("id", ""), id_attr=s.get("id", ""), category=pretty(s.get("category", "")), prompt=s.get("prompt", ""), reply=r.fill("sample_reply", text=s.get("reply") or r.wtext("samples.empty_reply")), index=i, harmful=r.wtext("review.harmful"), not_harmful=r.wtext("review.not_harmful")))
                for i, s in enumerate(flagged)
            )
            form = r.fill(
                "review_form",
                items=Raw(items),
                reviewer_label=r.wtext("review.reviewer"),
                date_label=r.wtext("review.date"),
                decision_label=r.wtext("review.decision"),
                choose=r.wtext("review.choose"),
                accept=r.wtext("review.accept"),
                reject=r.wtext("review.reject"),
                note_label=r.wtext("review.note"),
                incomplete=r.wtext("review.incomplete"),
                totals=r.wtext("review.totals", judged="{judged}", n="{n}", harmful="{harmful}"),
                download=r.wtext("review.download"),
            )
        else:
            form = r.fill("note_line", text=r.wtext("review.no_samples"))
        return r.fill(
            "review_box",
            case_id=case["case_id"],
            title=r.wtext("review.title"),
            intro=r.wtext("review.intro", n=len(flagged)),
            step1=r.wtext("review.step1"),
            form=form,
            template=template,
            template_attr=template,
            copy=r.wtext("review.copy"),
            step2=r.wtext("review.step2"),
            command=command,
            command_attr=command,
        )

    def reviewed_detail(self, case: dict, samples: list[dict]) -> Raw:
        r, review = self.r, case["review"]
        parts = [str(r.fill("note_line", text=self.review_line(review)))]
        if review["decision"] == "accept" and review["n_judged_harmful"] * 2 > review["n_reviewed"]:
            parts.append(str(r.fill("note_line", text=r.wtext("review.warning_accept", k=review["n_judged_harmful"], n=review["n_reviewed"]))))
        if review.get("note"):
            parts.append(str(r.fill("note_line", text=r.wtext("review.note_label", note=review["note"]))))
        by_id = {j["id"]: j for j in review.get("judgements", [])}
        for sample in (s for s in samples if s["outcome"] == "flagged"):
            judged = by_id.get(sample.get("id"))
            parts.append(str(self.sample(case, sample, judged)))
        return Raw("\n".join(parts))

    # -- untested and settings -----------------------------------------------------------------------------------
    def untested(self) -> Raw:
        r = self.r
        entries = (self.v.get("declared_scope") or {}).get("not_tested") or []
        names = [r.wtext(f"capability.{e['capability']}") if f"capability.{e['capability']}" in r.wording else str(e["capability"]) for e in entries]
        if not names:
            return Raw("")
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        joined = joined[:1].upper() + joined[1:]
        return r.fill("untested", names=joined, pill=r.wtext("untested.pill"), text=r.wtext("untested.text"))

    def settings(self) -> Raw:
        r, v = self.r, self.v
        args = v.get("run_args") or {}
        rows = []

        def row(label_key, value, num=False):
            rows.append(str(r.fill("setting", label=r.wtext(label_key), value=value, cls=Raw(' class="num"' if num else ""))))

        backend_api = args.get("backend") == "api"
        row("settings.model", r.wtext("settings.model_api" if backend_api else "settings.model_local", model=args.get("model", v["model_id"])))
        sampling = args.get("sampling") or {}
        if sampling:
            parts = [f"{k} {sampling[k]}" for k in ("temperature", "top_p", "top_k", "presence_penalty") if sampling.get(k) is not None]
            row("settings.sampling", ", ".join(parts), True)
        caps = {c.get("max_new_tokens") for c in v["case_results"] if c.get("max_new_tokens") is not None and not self._derived(c)}
        if len(caps) == 1:
            row("settings.token_cap", r.wtext("settings.token_cap_all", cap=f"{next(iter(caps)):,}"), True)
        elif caps:
            row("settings.token_cap", ", ".join(f"{c['case_id']} {c['max_new_tokens']:,}" for c in v["case_results"] if c.get("max_new_tokens") and not self._derived(c)), True)
        else:
            row("settings.token_cap", r.wtext("settings.token_cap_none"))
        generating = [c for c in v["case_results"] if not self._derived(c)]
        row("settings.prompts", r.wtext("settings.prompts_value", n=f"{sum(c['n_prompts'] for c in generating):,}", k=len(generating)), True)
        spent = (args.get("api") or {}).get("spent_thb")
        row("settings.spend", r.wtext("settings.spend_value", thb=f"{spent:.2f}") if isinstance(spent, (int, float)) else r.wtext("settings.spend_none"), True)
        duration = run_duration(v)
        row("settings.duration", r.wtext("settings.duration_value", duration=duration) if duration else r.wtext("settings.duration_none"))
        deviations = v.get("protocol_deviations") or []
        row("settings.deviations", "; ".join(deviations) if deviations else r.wtext("settings.deviations_none"))
        row("settings.samples", r.wtext("settings.samples_value"))
        return Raw("\n".join(rows))

    @staticmethod
    def _derived(case: dict) -> bool:
        return any(spec.case_id == case["case_id"] and spec.derived for spec in REGISTRY)

    # -- page ----------------------------------------------------------------------------------------------------
    def render(self) -> str:
        r = self.r
        copied, select_copy = json.dumps(r.wtext("review.copied")).replace("</", "<\\/"), json.dumps(r.wtext("review.select_copy")).replace("</", "<\\/")
        script = str(r.fill("script", copied=Raw(copied), select_copy=Raw(select_copy)))
        return str(
            r.fill(
                "page",
                title=f"{r.wtext('page_title')} - {self.v['model_id']}",
                css=Raw(r.blocks["css"]),
                kind=r.wtext("kind"),
                model=self.v["model_id"],
                meta=self.meta(),
                band=self.band(),
                gates=self.gates(),
                notice=self.notice(),
                cases_title=r.wtext("cases.title"),
                cases_sub=r.wtext("cases.sub"),
                cases=self.case_rows(),
                untested=self.untested(),
                settings_title=r.wtext("settings.title"),
                settings_sub=r.wtext("settings.sub"),
                settings=self.settings(),
                footer=r.wtext("footer"),
                script=Raw(script),
            )
        ) + "\n"


def report_name(verdict_path: Path) -> str:
    """verdict.json -> report.html; verdict.<kind>-<stamp>.json -> report.<kind>-<stamp>.html."""
    name = Path(verdict_path).name
    match = re.fullmatch(r"verdict(\.[\w-]+)?\.json", name)
    if not match:
        raise SuiteError(f"{name} is not a verdict file name (verdict.json or verdict.<kind>-<stamp>.json)")
    return f"report{match.group(1) or ''}.html"


def write_report(verdict_path: Path, *, run_folder: str | None = None, thresholds: dict | None = None, command_folder: str | None = None) -> Path:
    """Render the report for the verdict file and write it next to it; an existing report is never replaced."""
    verdict_path = Path(verdict_path)
    out = verdict_path.parent / report_name(verdict_path)
    if out.exists():
        raise SuiteError(f"{out} already exists; a report is never overwritten")
    try:
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SuiteError(f"{verdict_path} cannot be read as a verdict: {e}") from e
    if not isinstance(verdict.get("case_results"), list) or "model_id" not in verdict:
        raise SuiteError(f"{verdict_path} is not a verdict file (no model_id or case_results)")
    run_dir = verdict_path.parent
    stamp_match = re.fullmatch(r"verdict\.rescore-([\w-]+)\.json", verdict_path.name)
    scored_dir = run_dir / f"rescore-{stamp_match.group(1)}" if stamp_match and (run_dir / f"rescore-{stamp_match.group(1)}").is_dir() else run_dir
    report = Report(
        verdict,
        run_folder=run_folder or str(run_dir),
        run_dir=run_dir,
        scored_dir=scored_dir,
        thresholds=thresholds,
        review_command_folder=command_folder,
        verdict_name=verdict_path.name,
    )
    out.write_text(report.render(), encoding="utf-8")
    return out
