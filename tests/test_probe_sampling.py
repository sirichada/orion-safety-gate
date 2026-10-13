import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import probe_thinking as pt  # noqa: E402
from test_probe_thinking import FAKE_KEY, FakeBackend, FUNCTIONAL_PROBLEM as FUNCTIONAL, manifest, ok_body, run, write_manifest  # noqa: E402


def ids(planned):
    return [p.request_id for p in planned]


# ------------------------------------------- sampling sets (planned/repeat/diversity)


def test_default_sets_are_todays_requests():
    planned = pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True)
    assert ids(planned) == ["smoke", "smoke_off", "long_hold", "abc999_a", "3333"]
    assert all(p.set_name == "planned" and p.sampling_mode == "explicit_t0" and p.max_tokens is None for p in planned)


def test_repeat_set_has_smoke_and_problems_with_shared_group_ids():
    planned = pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=("planned", "repeat"))
    repeat = [p for p in planned if p.set_name == "repeat"]
    assert ids(repeat) == ["smoke__r2", "abc999_a__r2", "3333__r2"]
    by_id = {p.request_id: p for p in planned}
    for p in repeat:
        original = by_id[p.request_id.removesuffix("__r2")]
        assert p.group_id == original.group_id and p.prompt == original.prompt and p.sampling_mode == "explicit_t0"


def test_diversity_set_conditions_cap_and_thinking_off():
    planned = pt.plan_requests(manifest(), thinking_off_check=False, long_hold=False, sets=("diversity",), diversity_k=4)
    assert len(planned) == 12
    modes = {}
    for p in planned:
        modes.setdefault(p.group_id, set()).add(p.sampling_mode)
        assert p.max_tokens == 64 and p.extra_body == pt.THINKING_OFF_EXTRA_BODY and p.prompt == pt.DIVERSITY_PROMPT
    assert modes == {
        "diversity:omitted": {"omitted"},
        "diversity:t0": {"explicit_t0"},
        "diversity:t1": {"explicit_t1"},
    }


def test_sets_are_sent_in_planned_repeat_diversity_order():
    planned = pt.plan_requests(
        manifest(), thinking_off_check=True, long_hold=True, sets=("diversity", "repeat", "planned"), diversity_k=1
    )
    order = [p.set_name for p in planned]
    assert order == sorted(order, key={"planned": 0, "repeat": 1, "diversity": 2}.get)


def test_unknown_or_cut_sets_and_bad_k_raise():
    for bad in ("default_sampling", "controls_t1", "nonsense"):
        with pytest.raises(ValueError):
            pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=(bad,))
    with pytest.raises(ValueError):
        pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=("diversity",), diversity_k=0)


def test_payload_sampling_modes():
    def request(mode):
        return pt.PlannedRequest(
            "x", "diversity", "p", None, pt.THINKING_OFF_EXTRA_BODY, sampling_mode=mode, max_tokens=64
        )

    omitted = pt.build_payload("Mango 1.5-9B", request("omitted"), 32768, stream=True)
    assert "temperature" not in omitted and "top_p" not in omitted and "seed" not in omitted
    assert omitted["max_tokens"] == 64 and omitted["stream"] is True
    assert omitted["stream_options"] == {"include_usage": True}
    assert omitted["extra_body"] == pt.THINKING_OFF_EXTRA_BODY
    t1 = pt.build_payload("Mango 1.5-9B", request("explicit_t1"), 32768)
    assert (t1["temperature"], t1["top_p"]) == (1.0, 0.95)
    t0 = pt.build_payload("Mango 1.5-9B", request("explicit_t0"), 32768)
    assert (t0["temperature"], t0["top_p"]) == (0.0, 1.0)
    uncapped = pt.build_payload("Mango 1.5-9B", pt.PlannedRequest("y", "smoke", "p", None), 32768)
    assert uncapped["max_tokens"] == 32768


def test_dry_run_lists_sampling_mode_cap_and_model_info_note(tmp_path, monkeypatch, capsys):
    import httpx

    def boom(*args, **kwargs):
        raise AssertionError("network in dry run")

    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.delenv("ORION_API_KEY", raising=False)
    args = pt.build_parser().parse_args(
        [
            "run", "--dry-run", "--stream", "--sets", "planned,repeat,diversity", "--diversity-k", "2",
            "--models", "Qwen 3.8-Flash-Next", "--manifest", str(write_manifest(tmp_path)),
        ]
    )
    assert pt.cmd_run(args) == 0
    printed = capsys.readouterr().out
    assert "omitted" in printed and "explicit_t1" in printed and "cap     64" in printed
    assert "would GET" in printed and "not performed" in printed


def test_cmd_run_rejects_unknown_set(tmp_path):
    args = pt.build_parser().parse_args(
        ["run", "--dry-run", "--sets", "default_sampling", "--manifest", str(write_manifest(tmp_path))]
    )
    assert pt.cmd_run(args) == 2


# ------------------------------------------------- variance and diversity


def row(request_id, status=200, completion=100, reasoning=10, finish="stop", set_name="planned", kind="problem", group="g"):
    return {
        "model": "Qwen 3.8-27B", "request_id": request_id, "status": status, "completion_tokens": completion,
        "reasoning_tokens": reasoning, "finish_reason": finish, "set_name": set_name, "kind": kind, "group_id": group,
    }


def test_pair_stats_identical_and_early_divergence():
    same = pt.pair_stats(row("a"), row("a__r2"), {"content": "x", "reasoning": "think"}, {"content": "x", "reasoning": "think"})
    assert same["exact_match_full"] and same["exact_match_content"] and same["completion_tokens_abs_diff"] == 0
    diverge = pt.pair_stats(
        row("a", completion=100), row("a__r2", completion=300),
        {"content": "```\nprint(1)\n```", "reasoning": "alpha beta"},
        {"content": "```\nprint(2)\n```", "reasoning": "alpha gamma"},
    )
    assert not diverge["exact_match_full"] and diverge["common_prefix_chars_full"] == len("alpha ")
    assert diverge["completion_tokens_abs_diff"] == 200
    assert diverge["completion_tokens_rel_diff"] == pytest.approx(0.6667, abs=1e-4)
    assert diverge["extracted_code_identical"] is False


def test_pair_stats_prefix_equal_different_length_and_empty():
    a = {"content": "", "reasoning": "same start"}
    b = {"content": "answer", "reasoning": "same start and more"}
    stats = pt.pair_stats(row("a"), row("a__r2"), a, b)
    assert stats["common_prefix_chars_content"] == 0
    assert stats["common_prefix_chars_full"] == len("same start")
    assert not stats["exact_match_content"]
    empty = pt.pair_stats(row("a"), row("a__r2"), {"content": "", "reasoning": ""}, {"content": "", "reasoning": ""})
    assert empty["exact_match_full"] and empty["extracted_code_identical"]


def test_pair_stats_not_comparable_when_either_failed():
    stats = pt.pair_stats(
        row("a"), row("a__r2", status=524, completion=None),
        {"content": "", "reasoning": ""}, {"content": "", "reasoning": ""},
    )
    assert stats == {"comparable": False, "request_ids": ["a", "a__r2"], "statuses": [200, 524]}


def test_build_variance_pairs_planned_with_repeat_only():
    state = pt.RunState()
    for rid, set_name in (("smoke", "planned"), ("smoke__r2", "repeat"), ("lonely", "planned")):
        kind = "smoke" if rid.startswith("smoke") else "problem"
        state.rows.append(row(rid, set_name=set_name, kind=kind, group=rid.split("__")[0]))
        state.texts[("Qwen 3.8-27B", rid)] = {"content": "c", "reasoning": "r"}
    pairs = pt.build_variance(state)
    assert [p["group_id"] for p in pairs] == ["smoke"] and pairs[0]["comparable"]


def diversity_state(outputs_by_condition, status=200):
    state = pt.RunState()
    for condition, outputs in outputs_by_condition.items():
        for i, text in enumerate(outputs, 1):
            rid = f"div_{condition}_{i}"
            state.rows.append(row(rid, status=status, set_name="diversity", kind="diversity", group=f"diversity:{condition}"))
            state.texts[("Qwen 3.8-27B", rid)] = {"content": text, "reasoning": ""}
    return state


def test_diversity_counts_normalise_case_and_whitespace():
    state = diversity_state(
        {
            "omitted": ["The Fox runs.", "the  fox runs.", "A cat sleeps. Really."],
            "t0": ["Fox.", "Fox.", "Fox."],
            "t1": ["a.", "b.", "c.", "d."],
        }
    )
    d = pt.build_diversity(state)["Qwen 3.8-27B"]
    assert d["omitted"]["distinct_outputs"] == 2 and d["omitted"]["distinct_first_sentence"] == 2
    assert d["t0"]["distinct_outputs"] == 1 and d["t1"]["distinct_outputs"] == 4
    assert d["assessment"] == "greedy_like"
    assert all(len(o) <= 200 for o in d["omitted"]["outputs"])


@pytest.mark.parametrize(
    "n_default, n_t0, n_t1, ok, expected",
    [
        (1, 1, 6, [6, 6, 6], "greedy_like"),
        (2, 1, 6, [6, 6, 6], "greedy_like"),
        (6, 1, 6, [6, 6, 6], "sampled_like"),
        (5, 1, 6, [6, 6, 6], "sampled_like"),
        (3, 1, 6, [6, 6, 6], "inconclusive"),
        (4, 3, 4, [6, 6, 6], "not_informative"),
        (1, 1, 6, [2, 6, 6], "insufficient_data"),
    ],
)
def test_classify_default_rule(n_default, n_t0, n_t1, ok, expected):
    assert pt.classify_default(n_default, n_t0, n_t1, ok) == expected


def test_missing_condition_is_insufficient():
    d = pt.build_diversity(diversity_state({"omitted": ["a", "b", "c"]}))["Qwen 3.8-27B"]
    assert d["assessment"] == "insufficient_data"


# ---------------------------------------- thinking-off gating, guard, headers


def thinking_aware_responder(off_honoured=True):
    def respond(payload):
        off = payload.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
        if off and off_honoured:
            return 200, ok_body("Lion. A big cat.", completion=8, reasoning=0)
        return 200, ok_body("Lion.", completion=8, reasoning=5, reasoning_content="hmm")

    return respond


def test_diversity_runs_when_thinking_off_is_honoured(tmp_path):
    fake = FakeBackend(thinking_aware_responder(True))
    state, _ = run(tmp_path, fake, sets=("planned", "diversity"), diversity_k=2, manifest=manifest(problems=()))
    assert sum(1 for r in state.rows if r["set_name"] == "diversity") == 6
    assert state.skipped == []
    div = [p for p in fake.payloads if p["max_tokens"] == 64]
    assert len(div) == 6 and sum("temperature" not in p for p in div) == 2


def test_diversity_dropped_when_smoke_off_shows_thinking(tmp_path):
    fake = FakeBackend(thinking_aware_responder(False))
    state, _ = run(tmp_path, fake, sets=("planned", "diversity"), diversity_k=2, manifest=manifest(problems=()))
    assert not any(r["set_name"] == "diversity" for r in state.rows)
    assert "diversity dropped" in state.skipped[0]["reason"] and state.skipped[0]["model"] == "Mango 1.5-9B"


def test_diversity_only_drops_after_first_request_shows_reasoning(tmp_path):
    fake = FakeBackend(thinking_aware_responder(False))
    state, _ = run(tmp_path, fake, sets=("diversity",), diversity_k=3, manifest=manifest(problems=()))
    assert len(fake.payloads) == 1
    assert "not honoured" in state.skipped[0]["reason"]


def test_guard_uses_diversity_cap_not_run_cap(tmp_path):
    fake = FakeBackend(thinking_aware_responder(True))
    # 18 requests at a 64-token cap fit under 0.02 THB; at the run cap of 32768 the first one would not.
    state, guard = run(
        tmp_path, fake, sets=("diversity",), diversity_k=6, manifest=manifest(problems=()), max_tokens=32768, max_thb=0.02
    )
    assert len(fake.payloads) == 18 and state.stopped_reason is None


def test_guard_stops_mid_set(tmp_path):
    fake = FakeBackend(lambda p: (200, ok_body(completion=1000, prompt=500)))
    state, _ = run(
        tmp_path, fake, sets=("planned", "repeat"), max_thb=0.0165, thinking_off_check=False, long_hold=False
    )
    assert state.stopped_reason.startswith("spend guard") and 1 <= len(fake.payloads) < 6


def test_plan_worst_case_total_matches_sum_of_requests():
    planned = pt.plan_requests(
        manifest(), thinking_off_check=True, long_hold=True, sets=("planned", "repeat", "diversity"), diversity_k=2
    )
    total = pt.print_plan(["Qwen 3.8-27B"], planned, manifest(), 32768, 50, out=lambda text: None)
    expected = sum(
        pt.worst_case_thb("Qwen 3.8-27B", pt.estimate_input_tokens(p.prompt), pt.effective_max_tokens(p, 32768))
        for p in planned
    )
    assert total == pytest.approx(expected)


def test_all_headers_saved_and_scrubbed(tmp_path):
    def send(payload, timeout_s):
        headers = {"x-whatever": "v", "x-echo": f"leak {FAKE_KEY}", "date": "d"}
        return pt.SendResult(200, headers, json.dumps(ok_body()), None, 1.0)

    common = dict(thinking_off_check=False, long_hold=False, manifest=manifest(problems=()))
    backend = pt.Backend(send, lambda: None)
    run(tmp_path, FakeBackend(lambda p: (200, {})), backend=backend, save_all_headers=True, **common)
    raw = json.loads(next((tmp_path / "raw").glob("*.json")).read_text(encoding="utf-8"))
    assert set(raw["response_headers_all"]) == {"x-whatever", "x-echo", "date"}
    assert FAKE_KEY not in json.dumps(raw)
    second = tmp_path / "second"
    run(second, FakeBackend(lambda p: (200, {})), backend=backend, save_all_headers=False, **common)
    raw2 = json.loads(next((second / "raw").glob("*.json")).read_text(encoding="utf-8"))
    assert "response_headers_all" not in raw2


def run_cmd(tmp_path, monkeypatch, backend, extra):
    monkeypatch.setattr(pt, "LOG_DIR", tmp_path / "logs")
    out_root = tmp_path / "out"
    args = pt.build_parser().parse_args(
        [
            "run", "--manifest", str(write_manifest(tmp_path)), "--out-root", str(out_root),
            "--models", "Mango 1.5-9B", "--sets", "diversity", "--diversity-k", "1", *extra,
        ]
    )
    code = pt.cmd_run(args, backend_factory=lambda key, timeout: (backend, lambda: None), key_reader=lambda: FAKE_KEY)
    return code, out_root


def test_model_info_records_any_status_scrubs_key_and_stays_on_api_host(tmp_path, monkeypatch):
    calls = []

    def get(url):
        calls.append(url)
        if url.endswith("/v1/model/info"):
            return 403, f"forbidden for {FAKE_KEY}"
        return 404, "x" * (pt.MODEL_INFO_MAX_CHARS + 10)

    def send(payload, timeout_s):
        return pt.SendResult(200, {}, json.dumps(ok_body()), None, 1.0)

    code, out_root = run_cmd(tmp_path, monkeypatch, pt.Backend(send, lambda: None, get), [])
    assert code == 0
    assert calls == list(pt.MODEL_INFO_URLS)
    assert all(url.startswith("https://api.orion.cmkl.ai/") for url in calls)
    info = json.loads(next(out_root.rglob("model_info.json")).read_text(encoding="utf-8"))
    assert sorted(v["status"] for v in info.values()) == [403, 404]
    assert any(v["truncated"] for v in info.values())
    assert FAKE_KEY not in json.dumps(info)
    assert next(out_root.rglob("diversity.json")).exists()


def test_model_info_off_makes_no_lookup(tmp_path, monkeypatch):
    calls = []

    def get(url):
        calls.append(url)
        return 200, ""

    def send(payload, timeout_s):
        return pt.SendResult(200, {}, json.dumps(ok_body()), None, 1.0)

    code, _ = run_cmd(tmp_path, monkeypatch, pt.Backend(send, lambda: None, get), ["--no-model-info", "--no-save-all-headers"])
    assert code == 0 and calls == []


# --------------------------------------------- vendor_t1 set and --problems filter


def test_vendor_t1_set_sends_each_selected_problem_twice_at_explicit_t1():
    planned = pt.plan_requests(
        manifest(), thinking_off_check=True, long_hold=True, sets=("vendor_t1",), problem_ids=("3333",)
    )
    assert ids(planned) == ["3333__t1a", "3333__t1b"]
    original = pt.build_prompt(FUNCTIONAL["question_content"], FUNCTIONAL["starter_code"])
    for p in planned:
        assert p.set_name == "vendor_t1" and p.kind == "problem_t1" and p.sampling_mode == "explicit_t1"
        assert p.prompt == original and p.max_tokens is None and p.extra_body is None
        payload = pt.build_payload("Qwen 3.8-Flash-Next", p, 32768, stream=True)
        assert (payload["temperature"], payload["top_p"]) == (1.0, 0.95)
        assert payload["max_tokens"] == 32768 and payload["stream"] is True and "seed" not in payload


def test_t1_repeats_and_default_both_problems():
    planned = pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=("vendor_t1",), t1_repeats=3)
    assert ids(planned) == [
        "abc999_a__t1a", "abc999_a__t1b", "abc999_a__t1c", "3333__t1a", "3333__t1b", "3333__t1c",
    ]
    default = pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=("vendor_t1",))
    assert len(default) == 4


def test_problems_filter_applies_to_planned_and_repeat_and_rejects_unknown_ids():
    planned = pt.plan_requests(
        manifest(), thinking_off_check=False, long_hold=False, sets=("planned", "repeat"), problem_ids=("abc999_a",)
    )
    assert ids(planned) == ["smoke", "abc999_a", "smoke__r2", "abc999_a__r2"]
    with pytest.raises(ValueError):
        pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=("vendor_t1",), problem_ids=("nope",))
    with pytest.raises(ValueError):
        pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True, sets=("vendor_t1",), t1_repeats=0)


def test_vendor_t1_sits_between_repeat_and_diversity():
    planned = pt.plan_requests(
        manifest(), thinking_off_check=True, long_hold=True, sets=("diversity", "vendor_t1", "repeat", "planned"), diversity_k=1
    )
    order = [p.set_name for p in planned]
    assert order == sorted(order, key={"planned": 0, "repeat": 1, "vendor_t1": 2, "diversity": 3}.get)


def test_vendor_t1_run_records_rows_without_variance_or_diversity(tmp_path):
    fake = FakeBackend(lambda p: (200, ok_body("```python\nprint(1)\n```", completion=300, reasoning=200)))
    state, guard = run(
        tmp_path, fake, models=("Qwen 3.8-Flash-Next",), sets=("vendor_t1",), problem_ids=("abc999_a", "3333"),
        max_tokens=32768,
    )
    assert [r["request_id"] for r in state.rows] == ["abc999_a__t1a", "abc999_a__t1b", "3333__t1a", "3333__t1b"]
    assert all(r["sampling_mode"] == "explicit_t1" and r["set_name"] == "vendor_t1" and r["extracted_nonempty"] for r in state.rows)
    assert all(p["temperature"] == 1.0 and p["top_p"] == 0.95 for p in fake.payloads)
    assert pt.build_variance(state) == [] and pt.build_diversity(state) == {}


def test_cmd_run_dry_run_with_vendor_t1_and_problems_lists_only_selected(tmp_path, capsys):
    args = pt.build_parser().parse_args(
        ["run", "--dry-run", "--stream", "--models", "Qwen 3.8-Flash-Next", "--sets", "vendor_t1",
         "--problems", "3333", "--manifest", str(write_manifest(tmp_path))]
    )
    assert pt.cmd_run(args) == 0
    printed = capsys.readouterr().out
    assert "3333__t1a" in printed and "3333__t1b" in printed and "abc999_a" not in printed
    bad = pt.build_parser().parse_args(
        ["run", "--dry-run", "--sets", "vendor_t1", "--problems", "zzz", "--manifest", str(write_manifest(tmp_path))]
    )
    assert pt.cmd_run(bad) == 2
