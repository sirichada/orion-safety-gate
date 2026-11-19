import score_en_text
import load_en_text

HEADER = "Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID\n"


def test_the_prompt_set_carries_each_behaviours_tags(tmp_path, monkeypatch):
    source = tmp_path / "b.csv"
    source.write_text(
        HEADER + 'sing,copyright,copyright,"lyrics, hash_check",,song\nread,copyright,copyright,"book, hash_check",,book\nplain,standard,x,,,plain\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(load_en_text, "SRC", source)
    records = {r["native_id"]: r for r in load_en_text.load()}
    assert records["song"]["tags"] == ["lyrics", "hash_check"] and records["book"]["tags"] == ["book", "hash_check"]
    assert records["plain"]["tags"] == []
    assert records["song"]["functional_category"] == "copyright"  # en_text still sees it as a copyright behaviour
    assert load_en_text.behavior_tags()["harmbench_song"] == ["lyrics", "hash_check"]


def test_en_text_still_excludes_every_copyright_behaviour_whatever_its_tags():
    records = [
        {"excluded": True, "exclusion_reason": "copyright_hash_check_unimplemented", "scoring_method": "classifier",
         "functional_category": "copyright", "tags": tags, "label": None, "generation_failed": False}
        for tags in (["lyrics", "hash_check"], ["book", "hash_check"])
    ] + [{"excluded": False, "exclusion_reason": None, "scoring_method": "classifier", "functional_category": "standard", "tags": [], "label": 1, "generation_failed": False}]
    summary = score_en_text.aggregate(records)
    assert (summary["n_copyright_excluded"], summary["n_scored"], summary["score"]) == (2, 1, 1.0)
