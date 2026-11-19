import json

from case_summary import category_summary, sample_generations
from verdict import build_case_result, build_verdict
from cases import REGISTRY, select_cases
from capabilities import parse_capabilities


def records(n=40, harmful_every=4):
    return [
        {
            "id": f"r{i:03d}",
            "native_category": "a" if i % 2 else "b",
            "prompt": f"p{i}",
            "response": f"reply {i} " + "x" * 400,
            "label": int(i % harmful_every == 0),
            "excluded": False,
        }
        for i in range(n)
    ]


def test_category_summary_counts_flagged_per_category_most_flagged_first():
    rows = category_summary("en_text", records())
    assert rows[0] == {"category": "b", "n_scored": 20, "n_flagged": 10, "rate": 0.5}
    assert rows[1] == {"category": "a", "n_scored": 20, "n_flagged": 0, "rate": 0.0}


def test_integrity_and_knowledge_flag_label_zero():
    rows = category_summary("th_knowledge", records())
    assert sum(r["n_flagged"] for r in rows) == 30


def test_excluded_records_are_not_counted_or_sampled():
    recs = records()
    recs[0]["excluded"] = True
    recs[0]["label"] = None
    assert sum(r["n_scored"] for r in category_summary("en_text", recs)) == 39
    assert "r000" not in [s["id"] for s in sample_generations("en_text", recs, "m")]


def test_samples_are_seeded_by_model_and_case_and_independent_of_record_order():
    recs = records()
    first = sample_generations("en_text", recs, "model-a")
    assert first == sample_generations("en_text", list(reversed(recs)), "model-a")
    assert first != sample_generations("en_text", recs, "model-b")
    assert [s["outcome"] for s in first].count("flagged") == 5 and len(first) == 10
    assert all(len(s["reply"]) == 300 and s["reply_chars"] > 300 for s in first)


def test_a_review_sample_has_thirty_flagged_replies_in_full():
    sample = sample_generations("th_text", records(n=200, harmful_every=2), "m", review=True)
    flagged = [s for s in sample if s["outcome"] == "flagged"]
    assert len(flagged) == 30 and all(len(s["reply"]) > 300 for s in flagged)


def test_build_verdict_attaches_summaries_from_scored_files(tmp_path):
    scored = tmp_path / "scored_en_text.jsonl"
    scored.write_text("\n".join(json.dumps(r) for r in records()), encoding="utf-8")
    selection = select_cases(parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["en"]}))
    case = next(c for c in REGISTRY if c.case_id == "en_text")
    result = build_case_result(case, score=0.25, n_prompts=40, n_scored=40, threshold=0.3, min_scored=1, scoring_method="m", notes="")
    verdict = build_verdict("m", selection, [result], thresholds_provisional=True, run_args={}, scored_files={"en_text": scored})
    case_result = verdict["case_results"][0]
    assert case_result["category_summary"][0]["category"] == "b"
    assert len(case_result["sample_generations"]) == 10
