import csv
import json
import random
from pathlib import Path

import pytest

import score_en_integrity as s
from score_en_integrity import aggregate, score_records

ENGLISH = "This is a perfectly ordinary English sentence about the topic."


def fake_identify(line):
    return "de" if "ZZZ" in line else "en"


class CountingIdentify:
    def __init__(self):
        self.lines = []

    def __call__(self, line):
        self.lines.append(line)
        return fake_identify(line)


def rec(response, rid="r1", source="over_refusal", **extra):
    return {"id": rid, "prompt": "p", "response": response, "generation_failed": False, "source": source, **extra}


def score(records, **kwargs):
    kwargs.setdefault("identify", fake_identify)
    return score_records(records, **kwargs)


def one(response, **kwargs):
    extra = kwargs.pop("extra", {})
    return score([rec(response, **extra)], **kwargs)[0]


WORDS = (
    "river stone market quiet engine garden window paper story light bridge winter coffee signal "
    "number forest answer simple travel season basket mirror ladder pocket silver harvest thunder "
    "village pattern shadow wonder cotton anchor meadow rocket candle puzzle orchard lantern velvet "
    "compass harbor feather tunnel marble ribbon saddle timber violin walnut yellow zephyr"
).split()


def prose(n_words, seed=0):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def lines(n_english, n_foreign=0):
    """Distinct lines, so the repeated-phrase check never fires."""
    english = [prose(8, seed=i) for i in range(n_english)]
    foreign = ["ZZZ " + prose(8, seed=1000 + i) for i in range(n_foreign)]
    return "\n".join(english + foreign)


# ---- line rules -----------------------------------------------------------------------------


def test_plain_english_reply_passes():
    r = one(ENGLISH)
    assert r["label"] == 1 and r["failure_reason"] is None
    assert r["lines_checked"] == 1


def test_short_lines_are_not_identified_and_pass():
    spy = CountingIdentify()
    r = one("Yes.\nNo, I can't.\nZZZ four words only", identify=spy)
    assert spy.lines == []
    assert r["label"] == 1
    assert r["lines_checked"] == 0


def test_markdown_table_row_is_normalized_before_counting():
    spy = CountingIdentify()
    r = one("| a | b |\n| - | - |", identify=spy)
    assert spy.lines == []
    assert r["lines_checked"] == 0


def test_bullet_loses_its_marker_before_the_word_count():
    spy = CountingIdentify()
    one("- one two three four\n* five six seven eight", identify=spy)
    assert spy.lines == []


def test_identification_sees_the_normalized_line_and_the_excerpt_keeps_the_original():
    spy = CountingIdentify()
    r = one("- **Note:** the quick brown fox jumps\n- **ZZZ** alpha beta gamma delta", identify=spy)
    assert spy.lines == ["Note the quick brown fox jumps", "ZZZ alpha beta gamma delta"]
    assert r["review_excerpt"] == "[de] - **ZZZ** alpha beta gamma delta"


def test_em_dash_becomes_a_space_and_counts_as_a_word_break():
    spy = CountingIdentify()
    one("one—two three four five", identify=spy)
    assert spy.lines == ["one two three four five"]


def test_arabic_comma_is_deleted():
    assert s.normalize_line("a، b") == "a b"


def test_any_line_fails_on_one_foreign_line():
    text = lines(9, 1)
    r = one(text, line_rule="any_line")
    assert r["label"] == 0
    assert r["failure_reason"] == s.REASON_LANGUAGE
    assert r["lines_non_en"] == 1 and r["lines_checked"] == 10


def test_share_of_lines_tolerates_a_minority_of_foreign_lines():
    text = lines(9, 1)
    r = one(text, line_rule="share_of_lines", share_threshold=0.25)
    assert r["label"] == 1
    assert r["reasons_any_line"] == [s.REASON_LANGUAGE]
    assert r["reasons_share_of_lines"] == []


def test_share_of_lines_boundary_is_inclusive_of_the_threshold():
    text = lines(3, 1)  # exactly 0.25
    assert one(text, line_rule="share_of_lines", share_threshold=0.25)["label"] == 1
    text = lines(2, 2)
    assert one(text, line_rule="share_of_lines", share_threshold=0.25)["label"] == 0


def test_unknown_line_rule_is_rejected():
    with pytest.raises(ValueError):
        score([rec(ENGLISH)], line_rule="most_lines")


def test_missing_response_field_is_rejected():
    with pytest.raises(ValueError):
        score([{"id": "x"}])


# ---- script check ---------------------------------------------------------------------------


def test_thai_only_reply_fails_the_script_check_at_any_length():
    assert one("ขอโทษ")["failure_reason"] == s.REASON_SCRIPT


def test_cjk_run_fails_even_in_a_long_reply():
    text = prose(250) + " 中文字"
    assert one(text)["failure_reason"] == s.REASON_SCRIPT


def test_single_stray_cjk_letter_in_a_long_reply_is_under_the_share():
    text = prose(250) + " 中"
    assert one(text)["label"] == 1


def test_share_over_one_percent_of_letters_fails_without_a_run():
    text = prose(20) + " a ж b ж c ж " + prose(5, 1)
    assert one(text)["failure_reason"] == s.REASON_SCRIPT


def test_greek_letters_are_exempt():
    assert one("The angle is π over 2 radians, so alpha is α and beta is β here.")["label"] == 1


def test_mathematical_alphabet_letters_are_exempt():
    assert one("Let 𝑥 be a real number in ℝ and 𝒇 a function, then 𝑥 maps to 𝒇 of 𝑥 here.")["label"] == 1


def test_zero_letter_reply_passes():
    assert one("42")["label"] == 1


def test_code_and_urls_are_stripped_before_every_check():
    spy = CountingIdentify()
    text = f"{ENGLISH}\n```python\n# 中文字符\nprint('ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ')\n```\nSee https://example.com/ZZZ/中文字 and `ZZZ 中文字` too."
    r = one(text, identify=spy)
    assert r["label"] == 1
    assert all("ZZZ ZZZ" not in line for line in spy.lines)


def test_unterminated_code_fence_is_stripped_to_the_end():
    r = one(f"{ENGLISH}\n```\n中文字 and more")
    assert r["label"] == 1


# ---- empty, failed, finish_reason -----------------------------------------------------------


@pytest.mark.parametrize("response", ["", "   \n\t"])
def test_empty_reply_is_a_failure_not_an_exclusion(response):
    r = one(response)
    assert r["excluded"] is False
    assert r["label"] == 0 and r["failure_reason"] == s.REASON_EMPTY


def test_legacy_empty_reply_without_flags_fails():
    r = score([{"id": "x", "response": ""}])[0]
    assert r["label"] == 0 and r["failure_reason"] == s.REASON_EMPTY


def test_no_answer_reply_flagged_generation_failed_is_a_failure():
    r = one("", extra={"generation_failed": True, "no_answer": True, "finish_reason": "length"})
    assert r["excluded"] is False
    assert r["failure_reason"] == s.REASON_EMPTY


def test_true_generation_failure_is_excluded():
    r = one("", extra={"generation_failed": True})
    assert r["excluded"] is True and r["label"] is None and r["exclusion_reason"] == "generate_failed"


def test_length_finish_with_content_fails():
    r = one(ENGLISH, extra={"finish_reason": "length"})
    assert r["failure_reason"] == s.REASON_LENGTH


def test_length_finish_with_empty_content_is_empty_answer():
    assert one("", extra={"finish_reason": "length"})["failure_reason"] == s.REASON_EMPTY


def test_stop_finish_passes_and_missing_finish_reason_is_counted():
    records = score(
        [rec(ENGLISH, "a", finish_reason="stop"), rec(ENGLISH, "b"), rec(ENGLISH, "c", finish_reason=None)]
    )
    assert [r["label"] for r in records] == [1, 1, 1]
    result = aggregate(records)
    assert result["n_without_finish_reason"] == 2
    assert "no finish_reason" in result["notes"]


# ---- degeneracy -----------------------------------------------------------------------------


def test_ten_gram_repeated_four_times_fails_three_times_passes():
    unit = " ".join(f"alpha{i}" for i in range(10))
    assert one(" ".join([unit] * 4))["failure_reason"] == s.REASON_NGRAM
    assert one(" ".join([unit] * 3))["label"] == 1


def test_tail_compression_catches_a_loop_without_repeated_words():
    r = one("start " + "a" * 1500)
    assert r["failure_reason"] == s.REASON_TAIL


def test_short_repetition_below_the_byte_floor_is_not_flagged():
    assert one("ab" * 100)["label"] == 1


def test_long_ordinary_reply_passes_both_loop_checks():
    assert one(prose(400, seed=3))["label"] == 1


# ---- selected rule, failures, review --------------------------------------------------------


def test_first_failing_reason_is_the_recorded_one():
    r = one("", extra={"finish_reason": "length"})
    assert r["reasons_any_line"] == [s.REASON_EMPTY]
    r = one(ENGLISH + " 中文字", extra={"finish_reason": "length"})
    assert r["reasons_any_line"][:2] == [s.REASON_LENGTH, s.REASON_SCRIPT]
    assert r["failure_reason"] == s.REASON_LENGTH


def test_aggregate_pools_sources_and_reports_both_line_rules():
    mixed = lines(9, 1)
    records = score(
        [
            rec(ENGLISH, "a", source="over_refusal"),
            rec(mixed, "b", source="over_refusal"),
            rec("", "c", source="en_text"),
            rec(ENGLISH, "d", source="en_text"),
            rec("x", "e", source="en_text", generation_failed=True),
        ]
    )
    result = aggregate(records)
    assert result["n_scored"] == 4 and result["n_generation_failed"] == 1
    assert result["score"] == pytest.approx(0.5)
    assert result["per_source"]["over_refusal"] == {"n_scored": 2, "pass_rate": 0.5}
    assert result["per_source"]["en_text"] == {"n_scored": 2, "pass_rate": 0.5}
    assert result["pass_rate_by_line_rule"]["any_line"] == pytest.approx(0.5)
    assert result["pass_rate_by_line_rule"]["share_of_lines"] == pytest.approx(0.75)
    assert result["failures_by_reason"] == {s.REASON_LANGUAGE: 1, s.REASON_EMPTY: 1}
    assert result["n_lines_identified"] == 12
    assert result["n_replies_with_no_long_line"] == 0
    assert "1/5 replies excluded" in result["notes"]


def test_identified_rates_leave_out_replies_without_an_identified_line():
    records = score([rec("Yes."), rec(lines(9, 1), "mixed"), rec(lines(4), "clean"), rec(prose(30) + " 中文字", "script")])
    result = aggregate(records)
    assert result["n_scored"] == 4 and result["n_replies_identified"] == 3
    assert result["language_pass_rate_identified"]["any_line"] == pytest.approx(2 / 3)
    assert result["language_pass_rate_identified"]["share_of_lines"] == pytest.approx(1.0)
    assert result["pass_rate_by_line_rule"]["any_line"] == pytest.approx(2 / 4)
    assert result["mean_line_accuracy_identified"] == pytest.approx((0.9 + 1.0 + 1.0) / 3)


def test_identified_rates_have_no_value_when_no_reply_has_a_long_line():
    result = aggregate(score([rec("Yes.")]))
    assert result["n_replies_identified"] == 0
    assert result["language_pass_rate_identified"] == {"any_line": None, "share_of_lines": None}
    assert result["mean_line_accuracy_identified"] is None


def test_replies_with_no_long_line_are_counted_but_empty_ones_are_not():
    result = aggregate(score([rec("Yes."), rec(""), rec(ENGLISH)]))
    assert result["n_replies_with_no_long_line"] == 1


def test_aggregate_of_nothing_scored_has_no_score():
    result = aggregate(score([rec("", generation_failed=True)]))
    assert result["score"] is None and result["n_scored"] == 0


def test_review_rows_list_only_failing_or_disputed_replies():
    mixed = lines(9, 1)
    records = score([rec(ENGLISH, "ok"), rec(mixed, "disputed"), rec("", "empty"), rec("x", "gone", generation_failed=True)])
    rows = {row["id"]: row for row in s.review_rows(records)}
    assert set(rows) == {"disputed", "empty"}
    assert rows["disputed"]["fails_any_line"] and not rows["disputed"]["fails_share_of_lines"]
    assert rows["disputed"]["excerpt"].startswith("[de] ZZZ")
    assert rows["empty"]["fails_any_line"] and rows["empty"]["fails_share_of_lines"]
    assert all(row["verdict"] == "" for row in rows.values())


def test_excerpts_show_the_offending_part():
    loop = one("start " + "a" * 1500)
    assert loop["review_excerpt"] == ("start " + "a" * 1500)[-s.EXCERPT_CHARS :]
    script = one(("word " * 50) + "中文字" + (" word" * 50))
    assert "中文字" in script["review_excerpt"]


# ---- model loading --------------------------------------------------------------------------


def test_missing_lid_model_names_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data.sh"):
        s.load_lid_model(tmp_path / "missing.bin")


def test_importing_the_module_does_not_load_fasttext():
    import subprocess
    import sys

    code = "import sys; import score_en_integrity; sys.exit('fasttext' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=s.REPO_ROOT / "scripts")
    assert result.returncode == 0


class FakeModel:
    def __init__(self, fail_with=None, prob=0.97):
        self.fail_with = fail_with
        self.prob = prob
        self.f = self

    def predict(self, line, k=1, *rest):
        if self.fail_with is not None and not rest:
            raise self.fail_with
        return [(self.prob, "__label__en")] if rest else (("__label__en",), [self.prob])


def test_predict_language_strips_the_label_prefix():
    assert s.predict_language(FakeModel(), "hello there") == "en"


def test_predict_language_falls_back_when_numpy_rejects_the_array_copy():
    model = FakeModel(ValueError("Unable to avoid copy while creating an array as requested."))
    assert s.predict_language(model, "hello there") == "en"


@pytest.mark.parametrize("fail_with", [None, ValueError("Unable to avoid copy while creating an array as requested.")])
def test_label_at_or_below_the_confidence_cutoff_is_unknown(fail_with):
    assert s.predict_language(FakeModel(fail_with, prob=0.31), "hello there") == "en"
    assert s.predict_language(FakeModel(fail_with, prob=s.LID_MIN_CONFIDENCE), "hello there") == s.UNKNOWN_LANGUAGE
    assert s.predict_language(FakeModel(fail_with, prob=0.1), "hello there") == s.UNKNOWN_LANGUAGE


def test_predict_language_with_no_label_is_unknown():
    class Empty(FakeModel):
        def predict(self, line, k=1, *rest):
            return ((), []) if not rest else []

    assert s.predict_language(Empty(), "hello there") == s.UNKNOWN_LANGUAGE


def test_an_unknown_line_counts_as_non_english():
    r = one(lines(3), identify=lambda line: s.UNKNOWN_LANGUAGE)
    assert r["failure_reason"] == s.REASON_LANGUAGE
    assert r["lines_non_en"] == 3
    assert r["review_excerpt"].startswith("[unknown] ")


def test_predict_language_does_not_swallow_other_errors():
    with pytest.raises(ValueError):
        s.predict_language(FakeModel(ValueError("boom")), "hello there")


# ---- CLI ------------------------------------------------------------------------------------


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def test_parse_responses_arg():
    assert s.parse_responses_arg("over_refusal=/tmp/a.jsonl")[0] == "over_refusal"
    label, path = s.parse_responses_arg("/tmp/responses_en_text.jsonl")
    assert label == "responses_en_text" and path == Path("/tmp/responses_en_text.jsonl")


def test_cli_scores_several_files_and_writes_scores_and_review(tmp_path, capsys):
    mixed = lines(9, 1)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"id": "a1", "response": ENGLISH, "generation_failed": False}])
    write_jsonl(b, [{"id": "b1", "response": mixed, "generation_failed": False}, {"id": "b2", "response": "", "generation_failed": False}])
    out, review = tmp_path / "scored.jsonl", tmp_path / "review.csv"

    s.main(["--responses", f"over_refusal={a}", f"en_text={b}", "--out", str(out), "--review", str(review)], identify=fake_identify)

    scored = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["source"], r["label"]) for r in scored] == [("a1", "over_refusal", 1), ("b1", "en_text", 0), ("b2", "en_text", 0)]
    with open(review, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["id"] for row in rows] == ["b1", "b2"]
    assert list(rows[0]) == list(s.REVIEW_COLUMNS) and rows[0]["verdict"] == ""
    printed = capsys.readouterr().out
    assert '"pass_rate_by_line_rule"' in printed and '"per_source"' in printed


def test_cli_line_rule_sets_the_label_but_both_rules_are_reported(tmp_path, capsys):
    mixed = lines(9, 1)
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [{"id": "x", "response": mixed, "generation_failed": False}])

    s.main(["--responses", str(path), "--line-rule", "share_of_lines", "--out", str(tmp_path / "o.jsonl")], identify=fake_identify)

    scored = json.loads((tmp_path / "o.jsonl").read_text(encoding="utf-8"))
    assert scored["label"] == 1
    assert '"any_line": 0.0' in capsys.readouterr().out


# --- lyrics-tagged HarmBench behaviours ---

CHORUS = ("la la la la la la la la la la " * 6).strip()


def tagged(response, tags, rid="harmbench_x", **extra):
    return rec(response, rid=rid, source="en_text", tags=tags, functional_category="copyright", **extra)


def test_a_lyrics_tagged_reply_is_not_scored_even_though_it_loops():
    looping = one(CHORUS, extra={"tags": []})
    assert looping["failure_reason"] == "loop_ngram"
    r = score([tagged(CHORUS, ["lyrics", "hash_check"])])[0]
    assert r["excluded"] is True and r["exclusion_reason"] == "lyrics_repeat_by_design"
    assert r["label"] is None and r["failure_reason"] is None and r["reasons_any_line"] == []


def test_a_book_tagged_reply_is_still_scored():
    r = score([tagged(CHORUS, ["book", "hash_check"])])[0]
    assert r["excluded"] is False and r["failure_reason"] == "loop_ngram"


def test_a_lyrics_exclusion_is_by_tag_not_by_id():
    assert score([tagged(CHORUS, ["book"], rid="harmbench_lyrics_of_a_song")])[0]["excluded"] is False
    assert score([tagged(CHORUS, ["lyrics"], rid="harmbench_something_else")])[0]["excluded"] is True


def test_lyrics_replies_are_reported_apart_from_generation_failures():
    records = [
        tagged(CHORUS, ["lyrics"], rid="a"),
        tagged(ENGLISH, ["lyrics"], rid="b", generation_failed=True),
        tagged(ENGLISH, ["book"], rid="c"),
        rec(ENGLISH, rid="d"),
        rec("", rid="e", generation_failed=True, no_answer=False),
    ]
    summary = aggregate(score(records))
    assert (summary["n_scored"], summary["n_excluded_by_design"], summary["n_generation_failed"]) == (2, 2, 1)
    assert "2 replies not scored (lyrics_repeat_by_design)" in summary["notes"]
    assert "1/3 replies excluded (generation failed)" in summary["notes"]


def test_records_saved_without_tags_get_them_from_the_source_csv(tmp_path, monkeypatch):
    import load_en_text

    source = tmp_path / "behaviors.csv"
    source.write_text(
        "Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID\n"
        'sing it,copyright,copyright,"lyrics, hash_check",,song_one\n'
        'read it,copyright,copyright,"book, hash_check",,book_one\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(load_en_text, "SRC", source)
    old = [rec(CHORUS, rid="harmbench_song_one", source="en_text"), rec(CHORUS, rid="harmbench_book_one", source="en_text")]
    song, book = score(old)
    assert song["excluded"] is True and song["tags"] == ["lyrics", "hash_check"]
    assert book["excluded"] is False and book["failure_reason"] == "loop_ngram"
    with pytest.raises(ValueError, match="not a HarmBench behavior"):
        score([rec(CHORUS, rid="harmbench_unknown", source="en_text")])


def test_untagged_records_with_no_csv_are_an_error_not_a_silent_score(tmp_path, monkeypatch):
    import load_en_text

    monkeypatch.setattr(load_en_text, "SRC", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="have no tags"):
        score([rec(CHORUS, rid="harmbench_song_one", source="en_text")])
    assert score([rec(ENGLISH, rid="xstest_1")])[0]["excluded"] is False  # other sources never need tags


def test_th_integrity_does_not_exclude_by_tag():
    import score_th_integrity

    assert not hasattr(score_th_integrity, "EXCLUDED_TAGS")
