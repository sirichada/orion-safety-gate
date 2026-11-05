import csv
import json

import pytest

import sensitivity_integrity as g

EN_COLUMNS = ["id", "source", "fails_any_line", "fails_share_of_lines", "reasons_any_line", "reasons_share_of_lines",
              "lines_checked", "lines_non_en", "excerpt", "verdict"]


def columns(profile):
    return [profile.off_language_column if c == "lines_non_en" else c for c in EN_COLUMNS]


def write_labels(path, profile, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns(profile))
        writer.writeheader()
        for rid, source, reasons, checked, off, verdict in rows:
            writer.writerow({"id": rid, "source": source, "reasons_any_line": reasons, "reasons_share_of_lines": "",
                             "lines_checked": checked, profile.off_language_column: off, "verdict": verdict,
                             "fails_any_line": True, "fails_share_of_lines": False, "excerpt": ""})


ROWS = {
    "en": [("a", "m_x", "language_line", 10, 1, "FALSE"), ("b", "m_x", "language_line", 4, 3, "TRUE"),
           ("c", "m_x", "empty_answer", 0, 0, ""), ("d", "m_x", "script_mix", 5, 0, "TRUE")],
    "th": [("a", "m_x", "thai_ratio", 10, 1, "FALSE"), ("b", "m_x", "thai_ratio", 4, 3, "TRUE"),
           ("c", "m_x", "empty_answer", 0, 0, ""), ("d", "m_x", "script_mix", 5, 0, "TRUE")],
}


@pytest.mark.parametrize("scorer", ["en", "th"])
def test_each_profile_reads_its_own_off_language_column_and_reason(tmp_path, scorer):
    profile = g.PROFILES[scorer]
    path = tmp_path / "l.csv"
    write_labels(path, profile, ROWS[scorer])
    rows = g.load_rows([path], profile)
    assert [r["lines_off_language"] for r in rows] == [1, 3, 0, 0]
    assert profile.language_reason in rows[0]["reasons"]


def test_a_review_file_from_the_other_scorer_is_rejected(tmp_path):
    path = tmp_path / "l.csv"
    write_labels(path, g.PROFILES["en"], ROWS["en"])
    with pytest.raises(KeyError):
        g.load_rows([path], g.PROFILES["th"])


def test_blank_verdict_is_only_allowed_on_empty_answers(tmp_path):
    path = tmp_path / "l.csv"
    write_labels(path, g.PROFILES["th"], [("a", "m_x", "thai_ratio", 10, 1, "")])
    with pytest.raises(SystemExit, match="blank verdict"):
        g.load_rows([path], g.PROFILES["th"])


def test_the_language_reason_decides_which_rows_the_cutoff_applies_to():
    th, en = g.PROFILES["th"], g.PROFILES["en"]
    row = {"reasons": {"thai_ratio"}, "lines_checked": 4, "lines_off_language": 1}
    assert g.fails(row, g.ANY_LINE, 0, True, True, th)
    assert not g.fails(row, 0.25, 0, True, True, th)  # strict: 1/4 is not above 0.25
    assert g.fails(row, 0.2, 0, True, True, th)
    assert not g.fails(row, g.ANY_LINE, 5, True, True, th)  # fewer checked lines than the minimum
    # Under the English profile the same reason is an unrelated check that always fails.
    assert g.fails(row, 0.99, 0, True, True, en)


def test_loop_and_empty_switches():
    th = g.PROFILES["th"]
    loop = {"reasons": {"loop_ngram"}, "lines_checked": 1, "lines_off_language": 0}
    empty = {"reasons": {"empty_answer"}, "lines_checked": 0, "lines_off_language": 0}
    assert g.fails(loop, g.ANY_LINE, 0, True, True, th) and not g.fails(loop, g.ANY_LINE, 0, False, True, th)
    assert g.fails(empty, g.ANY_LINE, 0, True, True, th) and not g.fails(empty, g.ANY_LINE, 0, True, False, th)


def test_cli_for_thai_writes_a_thai_titled_grid(tmp_path, monkeypatch):
    labels = tmp_path / "l.csv"
    write_labels(labels, g.PROFILES["th"], ROWS["th"])
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--scorer", "th", "--labels", str(labels), "--scored", "m_x=40", "--out", str(out)])
    g.main()
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("# th_integrity sensitivity grid")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["case_id"] == "th_integrity" and meta["n_rows"] == 4
    with open(out / "grid.csv", encoding="utf-8", newline="") as f:
        grid = list(csv.DictReader(f))
    strict = next(r for r in grid if r["cutoff"] == "any_line" and r["min_lines"] == "0" and r["loop_on"] == "True" and r["empty_counted"] == "True")
    assert strict["flagged"] == "4" and float(strict["pass_m"]) == pytest.approx(1 - 4 / 40)


def test_default_scorer_stays_english(tmp_path, monkeypatch):
    labels = tmp_path / "l.csv"
    write_labels(labels, g.PROFILES["en"], ROWS["en"])
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--labels", str(labels), "--scored", "m_x=40", "--out", str(out)])
    g.main()
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("# en_integrity sensitivity grid")
