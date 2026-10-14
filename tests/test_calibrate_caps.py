import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import calibrate_caps as cc  # noqa: E402
import run_config  # noqa: E402

MODEL = "Qwen 3.8-Flash-Next"


def row(completion, finish="stop", prompt=100, latency=10.0, content=True, failed=False):
    return {"failed": failed, "finish_reason": finish, "prompt_tokens": prompt, "completion_tokens": completion,
            "latency_s": latency, "has_content": content}


def test_sample_is_seeded_distinct_sorted_and_per_case():
    a = cc.sample_indices(400, 20, 1, "en_text")
    assert a == cc.sample_indices(400, 20, 1, "en_text")
    assert a == sorted(set(a)) and len(a) == 20 and max(a) < 400
    assert a != cc.sample_indices(400, 20, 2, "en_text")
    assert a != cc.sample_indices(400, 20, 1, "th_text")
    assert cc.sample_indices(5, 20, 1, "x") == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "max_tokens, cut, expected_cap",
    [
        (300, 0, 1024),  # 2 x 300 = 600 -> 1024
        (512, 0, 1024),  # exactly 2 x max fits the step
        (513, 0, 2048),
        (5000, 0, 16384),
        (9000, 0, None),  # 18000 is above the largest step
        (300, 1, None),  # a cut-off at the calibration cap means the cap was too low to learn anything
        (None, 0, None),
    ],
)
def test_proposed_cap(max_tokens, cut, expected_cap):
    cap, reason = cc.propose_cap(max_tokens, cut, 8192)
    assert cap == expected_cap and reason


def test_case_summary_counts_and_proposes():
    rows = [row(200), row(800, latency=20.0), row(400, content=False), row(None, finish=None, failed=True)]
    summary = cc.summarize_case(rows, 8192)
    assert (summary["n_sampled"], summary["n_replied"], summary["n_failed"], summary["n_cut_off"]) == (4, 3, 1, 0)
    assert summary["n_empty"] == 1 and summary["max_completion_tokens"] == 800
    assert summary["mean_completion_tokens"] == pytest.approx(466.666, abs=0.01)
    assert summary["proposed_max_new_tokens"] == 2048  # 2 x 800 = 1600
    assert summary["mean_latency_s"] == pytest.approx(40 / 3)


def test_case_summary_with_a_cut_off_proposes_nothing():
    summary = cc.summarize_case([row(100), row(8192, finish="length", content=False)], 8192)
    assert summary["n_cut_off"] == 1 and summary["proposed_max_new_tokens"] is None


def test_full_run_estimate_uses_prices_prompt_count_and_latency():
    summary = {"mean_prompt_tokens": 100, "mean_completion_tokens": 1000, "mean_latency_s": 30.0}
    estimate = cc.estimate_full_run(summary, ["x" * 100] * 600, MODEL, 4096, 3)
    per_prompt = (100 * 1.0 + 1000 * 1.0) / 1e6
    assert estimate["n_prompts"] == 600
    assert estimate["expected_thb"] == pytest.approx(per_prompt * 600, abs=1e-4)
    assert estimate["expected_hours"] == pytest.approx(600 * 30 / 3 / 3600, abs=0.01)
    worst_each = ((100 / 2 + 16) * 1.0 + 4096 * 1.0) / 1e6
    assert estimate["worst_case_thb_at_proposed_cap"] == pytest.approx(worst_each * 600, abs=1e-3)
    assert cc.estimate_full_run(summary, ["x"], MODEL, None, 3)["worst_case_thb_at_proposed_cap"] is None


@pytest.fixture
def prompt_files(tmp_path, monkeypatch):
    files = {}
    for case_id in cc.DEFAULT_CASES:
        path = tmp_path / f"{case_id}.jsonl"
        path.write_text(
            "".join(json.dumps({"id": f"{case_id}_{i}", "model_input": f"prompt {i}"}) + "\n" for i in range(30)),
            encoding="utf-8",
        )
        files[case_id] = path
    monkeypatch.setattr(cc, "PROMPT_FILES", files)
    monkeypatch.setattr(cc, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(cc, "LOG_DIR", tmp_path / "logs")
    return files


def fake_generate_factory(completion=300, budget_on_call=None, finish="stop"):
    state = {"calls": 0, "kwargs": None}

    def make(model, concurrency=3, no_answer_retries=0, spend=None):
        state["kwargs"] = {"model": model, "concurrency": concurrency, "no_answer_retries": no_answer_retries}

        def generate(prompts, max_new_tokens, temperature, top_p, seed=None, chat_template=None):
            state["calls"] += 1
            if budget_on_call and state["calls"] == budget_on_call:
                raise run_config.BudgetExceeded("limit")
            generate.meta = [
                {"finish_reason": finish, "attempts": 1, "prompt_tokens": 50, "completion_tokens": completion,
                 "latency_s": 12.0, "cost_thb": 0.01, "reasoning_content": "think", "failed": False}
                for _ in prompts
            ]
            return ["answer" for _ in prompts]

        generate.meta = []
        return generate

    return make, state


def test_dry_run_sends_nothing_and_writes_nothing(prompt_files, tmp_path, capsys):
    def no_generate(*a, **k):
        raise AssertionError("a dry run must not build a backend")

    code = cc.main(["--model", MODEL, "--model-id", "x", "--dry-run"], make_generate=no_generate)
    out = capsys.readouterr().out
    assert code == 0 and "worst case for the sample" in out and "Dry run: nothing was sent" in out
    assert not (tmp_path / "out").exists() and not (tmp_path / "logs").exists()


def test_a_real_run_requires_a_spend_limit_and_a_priced_model(prompt_files):
    with pytest.raises(SystemExit, match="--max-thb is required"):
        cc.main(["--model", MODEL, "--model-id", "x"])
    with pytest.raises(SystemExit, match="no price"):
        cc.main(["--model", "Unpriced", "--model-id", "x", "--dry-run"])
    with pytest.raises(SystemExit, match="between 1 and 3"):
        cc.main(["--model", MODEL, "--model-id", "x", "--dry-run", "--api-concurrency", "4"])


def test_real_run_writes_calibration_and_a_config_snippet(prompt_files, tmp_path, capsys):
    make, state = fake_generate_factory(completion=300)
    code = cc.main(["--model", MODEL, "--model-id", "Qwen Flash", "--max-thb", "5", "--n", "10"], make_generate=make)
    assert code == 0
    assert state["kwargs"] == {"model": MODEL, "concurrency": 3, "no_answer_retries": 0}
    run_dir = next((tmp_path / "out").iterdir())
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["model"] == MODEL and calibration["seed"] == cc.DEFAULT_SEED and calibration["n_per_case"] == 10
    assert set(calibration["cases"]) == set(cc.DEFAULT_CASES)
    case = calibration["cases"]["th_text"]
    assert case["n_sampled"] == 10 and case["proposed_max_new_tokens"] == 1024
    assert case["estimate_full_run"]["n_prompts"] == 30 and case["estimate_full_run"]["expected_thb"] is not None
    assert calibration["generation_config_snippet"] == {
        "Qwen Flash": {c: {"max_new_tokens": 1024} for c in cc.DEFAULT_CASES}
    }
    raw = [json.loads(line) for line in (run_dir / "raw_en_text.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw) == 10 and all("reasoning_content" not in r for r in raw)
    assert list((tmp_path / "logs").glob("calibrate_caps_*.log"))
    assert not isinstance(sys.stdout, cc.Tee)


def test_a_budget_stop_keeps_what_was_measured_and_exits_nonzero(prompt_files, tmp_path):
    make, state = fake_generate_factory(budget_on_call=2)
    code = cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--n", "20"], make_generate=make)
    assert code == 1
    run_dir = next((tmp_path / "out").iterdir())
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["stopped"].startswith("en_text") and list(calibration["cases"]) == ["en_text"]
    assert calibration["cases"]["en_text"]["n_sampled"] == 12  # the first chunk only (3 in flight x 4)


def test_cut_off_replies_propose_no_cap(prompt_files, tmp_path):
    make, _ = fake_generate_factory(completion=8192, finish="length")
    cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--n", "4", "--cases", "en_text"], make_generate=make)
    path = next((tmp_path / "out").iterdir()) / "calibration.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    assert calibration["cases"]["en_text"]["proposed_max_new_tokens"] is None
    assert calibration["generation_config_snippet"] == {"x": {}}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_full_reasoning_goes_to_a_sidecar_not_the_raw_rows(prompt_files, tmp_path):
    make, _ = fake_generate_factory(completion=300)
    cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--n", "4", "--cases", "en_text"], make_generate=make)
    run_dir = next((tmp_path / "out").iterdir())
    raw = read_jsonl(run_dir / "raw_en_text.jsonl")
    sidecar = read_jsonl(run_dir / "raw_en_text.reasoning.jsonl")
    assert [r["id"] for r in raw] == [e["id"] for e in sidecar] and len(sidecar) == 4
    assert all("reasoning_content" not in r and r["reasoning_chars"] == 5 for r in raw)
    assert sidecar[0] == {"id": raw[0]["id"], "finish_reason": "stop", "completion_tokens": 300, "reasoning_content": "think"}


def earlier_run(tmp_path, model=MODEL):
    folder = tmp_path / "earlier"
    folder.mkdir()
    rows = [
        {"id": "en_text_1", "finish_reason": "stop"},
        {"id": "en_text_4", "finish_reason": "length"},
        {"id": "en_text_7", "finish_reason": "length"},
    ]
    (folder / "raw_en_text.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (folder / "raw_th_text.jsonl").write_text(json.dumps({"id": "th_text_2", "finish_reason": "stop"}) + "\n", encoding="utf-8")
    calibration = {
        "model": model,
        "cases": {
            "en_text": {"sample_ids": ["en_text_1", "en_text_4", "en_text_7"]},
            "th_text": {"sample_ids": ["th_text_2"]},
        },
    }
    path = folder / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    return path


def test_ids_from_selects_only_the_cut_off_prompts(tmp_path):
    path = earlier_run(tmp_path)
    assert cc.ids_from_calibration(path, True, MODEL) == {"en_text": ["en_text_4", "en_text_7"]}
    assert cc.ids_from_calibration(path, False, MODEL) == {
        "en_text": ["en_text_1", "en_text_4", "en_text_7"], "th_text": ["th_text_2"],
    }
    with pytest.raises(ValueError, match="not 'Other'"):
        cc.ids_from_calibration(path, True, "Other")


def test_rerun_of_cut_off_prompts_sends_exactly_those_ids(prompt_files, tmp_path):
    path = earlier_run(tmp_path)
    sent = []
    make, _ = fake_generate_factory(completion=9000)

    def recording(model, **kwargs):
        inner = make(model, **kwargs)

        def generate(prompts, *a, **k):
            sent.extend(prompts)
            texts = inner(prompts, *a, **k)
            generate.meta = inner.meta
            return texts

        generate.meta = []
        return generate

    code = cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--ids-from", str(path), "--only-cut-off",
                    "--max-new-tokens", "16384"], make_generate=recording)
    assert code == 0
    assert sent == ["prompt 4", "prompt 7"]
    run_dir = next(d for d in (tmp_path / "out").iterdir())
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert list(calibration["cases"]) == ["en_text"] and calibration["cases"]["en_text"]["sample_ids"] == ["en_text_4", "en_text_7"]
    assert calibration["calibration_max_new_tokens"] == 16384 and calibration["ids_from"]["only_cut_off"] is True
    assert calibration["cases"]["en_text"]["proposed_max_new_tokens"] is None  # 2 x 9000 is above the largest step


def test_ids_from_errors_are_clear(prompt_files, tmp_path):
    path = earlier_run(tmp_path)
    with pytest.raises(SystemExit, match="--only-cut-off needs --ids-from"):
        cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--only-cut-off"])
    with pytest.raises(SystemExit, match="--ids-from: .*not 'Qwen 3.8-27B'"):
        cc.main(["--model", "Qwen 3.8-27B", "--model-id", "x", "--max-thb", "5", "--ids-from", str(path), "--dry-run"])


def sampling_factory(reject_top_k=False, error="HTTP 400: top_k is not supported"):
    state = {"calls": [], "kwargs": None, "sent_chunks": []}

    def make(model, concurrency=3, no_answer_retries=0, spend=None, extra_params=None, max_attempts=None):
        state["kwargs"] = {"extra_params": extra_params}
        state["max_attempts"] = max_attempts

        def generate(prompts, max_new_tokens, temperature, top_p, seed=None, chat_template=None):
            state["calls"].append((temperature, top_p))
            state["sent_chunks"].append(list(prompts))
            if reject_top_k and extra_params:
                generate.meta = [
                    {"finish_reason": None, "attempts": 0, "failed": True, "error": error}
                    for _ in prompts
                ]
                return ["" for _ in prompts]
            generate.meta = [
                {"finish_reason": "stop", "attempts": 1, "prompt_tokens": 5, "completion_tokens": 50, "latency_s": 1.0,
                 "cost_thb": 0.01, "reasoning_content": "r", "failed": False}
                for _ in prompts
            ]
            return ["answer" for _ in prompts]

        generate.meta = []
        return generate

    return make, state


def test_sampling_settings_repeats_and_top_k_are_applied_and_recorded(prompt_files, tmp_path):
    path = earlier_run(tmp_path)
    make, state = sampling_factory()
    code = cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--ids-from", str(path), "--only-cut-off",
                    "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--repeats", "2",
                    "--max-new-tokens", "16384"], make_generate=make)
    assert code == 0
    assert state["kwargs"] == {"extra_params": {"top_k": 20}}
    assert {c for c in state["calls"]} == {(1.0, 0.95)}
    assert state["sent_chunks"][0] == ["prompt 4"]  # the first request goes out alone
    assert [p for chunk in state["sent_chunks"] for p in chunk] == ["prompt 4", "prompt 4", "prompt 7", "prompt 7"]
    run_dir = next(d for d in (tmp_path / "out").iterdir())
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["sampling"] == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "top_k_location": "top_level",
        "extra_params_sent": {"top_k": 20}, "retries": True, "repeats": 2,
    }
    ids = [r["id"] for r in read_jsonl(run_dir / "raw_en_text.jsonl")]
    assert ids == ["en_text_4__r1", "en_text_4__r2", "en_text_7__r1", "en_text_7__r2"]


def test_a_gateway_rejection_stops_the_run_after_the_first_request(prompt_files, tmp_path, capsys):
    path = earlier_run(tmp_path)
    make, state = sampling_factory(reject_top_k=True)
    code = cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--ids-from", str(path), "--only-cut-off",
                    "--top-k", "20", "--repeats", "2"], make_generate=make)
    assert code == 1
    assert state["sent_chunks"] == [["prompt 4"]]  # nothing else was sent
    run_dir = next(d for d in (tmp_path / "out").iterdir())
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert "HTTP 400: top_k is not supported" in calibration["stopped"]


def test_defaults_keep_the_suite_protocol(prompt_files, tmp_path):
    make, state = sampling_factory()
    cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--n", "2", "--cases", "en_text"], make_generate=make)
    assert state["kwargs"] == {"extra_params": None} or state["kwargs"] is None
    assert state["calls"] == [(0.0, 1.0)]


def test_a_500_for_an_unsupported_field_also_stops_the_run_after_one_request(prompt_files, tmp_path):
    path = earlier_run(tmp_path)
    make, state = sampling_factory(reject_top_k=True, error="HTTP 500: unexpected keyword argument 'top_k'")
    code = cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--ids-from", str(path), "--only-cut-off",
                    "--top-k", "20", "--repeats", "2"], make_generate=make)
    assert code == 1 and state["sent_chunks"] == [["prompt 4"]]


def test_top_k_in_extra_body_and_no_retries_are_passed_and_recorded(prompt_files, tmp_path):
    path = earlier_run(tmp_path)
    make, state = sampling_factory()
    cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--ids-from", str(path), "--only-cut-off",
             "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--top-k-location", "extra_body",
             "--no-retries"], make_generate=make)
    assert state["kwargs"] == {"extra_params": {"extra_body": {"top_k": 20}}}
    assert state["max_attempts"] == 1
    run_dir = next(d for d in (tmp_path / "out").iterdir())
    sampling = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))["sampling"]
    assert sampling["top_k_location"] == "extra_body" and sampling["extra_params_sent"] == {"extra_body": {"top_k": 20}}
    assert sampling["retries"] is False


def test_no_sampling_extras_are_recorded_as_none(prompt_files, tmp_path):
    make, _ = sampling_factory()
    cc.main(["--model", MODEL, "--model-id", "x", "--max-thb", "5", "--n", "2", "--cases", "en_text"], make_generate=make)
    run_dir = next(d for d in (tmp_path / "out").iterdir())
    sampling = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))["sampling"]
    assert sampling["top_k"] is None and sampling["top_k_location"] is None and sampling["extra_params_sent"] is None
