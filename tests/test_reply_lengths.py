import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import reply_lengths  # noqa: E402


def write_run(root: Path, name: str, case: str, records: list[dict], cfg: dict | None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    path = run / f"responses_{case}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    if cfg is not None:
        (run / f"responses_{case}.jsonl.run_config.json").write_text(json.dumps({"args": cfg}), encoding="utf-8")
    return path


def rec(reason, tokens):
    return {"finish_reason": reason, "usage": None if tokens is None else {"completion_tokens": tokens}}


def test_counts_and_longest(tmp_path):
    p = write_run(tmp_path, "M_1", "th_knowledge",
                  [rec("stop", 100), rec("stop", 9000), rec("length", 8192), rec("stop", None), rec("error", 5)],
                  {"model": "M", "max_new_tokens": 8192})
    row = reply_lengths.summarise(p, check_cap=4096)
    assert row["model"] == "M" and row["cap"] == "8192" and row["case"] == "th_knowledge"
    assert (row["n"], row["n_finished"], row["n_cut_off"], row["n_other"], row["n_no_usage"]) == (5, 2, 1, 1, 1)
    assert row["longest_finished"] == 9000
    assert row["over_check_cap"] == 1


def test_missing_run_config_and_no_check_cap(tmp_path):
    p = write_run(tmp_path, "X_1", "en_text", [rec("length", 16)], None)
    row = reply_lengths.summarise(p, check_cap=None)
    assert row["model"] == "?" and row["cap"] == "?"
    assert row["longest_finished"] is None and row["over_check_cap"] is None


def test_main_skips_reasoning_files_and_writes_csv(tmp_path, capsys):
    write_run(tmp_path, "M_1", "th_text", [rec("stop", 10)], {"model": "M", "max_new_tokens": 256})
    (tmp_path / "M_1" / "responses_th_text.jsonl.reasoning.jsonl").write_text("{}\n", encoding="utf-8")
    out = tmp_path / "t.csv"
    assert reply_lengths.main(["--results", str(tmp_path), "--csv", str(out)]) == 0
    assert "th_text" in capsys.readouterr().out
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_main_no_files(tmp_path):
    assert reply_lengths.main(["--results", str(tmp_path)]) == 1
