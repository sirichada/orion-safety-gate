import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import probe_thinking as pt  # noqa: E402

FAKE_KEY = "sk-FAKEKEYFORTESTS0123456789"

STDIN_PROBLEM = {
    "question_id": "abc999_a",
    "question_title": "Sum",
    "platform": "atcoder",
    "contest_date": "2025-03-01T00:00:00",
    "difficulty": "easy",
    "problem_format": "stdin",
    "question_content": "Print A+B.",
    "starter_code": "",
    "public_test_cases": "[]",
}
FUNCTIONAL_PROBLEM = {
    "question_id": "3333",
    "question_title": "Two Sum",
    "platform": "leetcode",
    "contest_date": "2025-03-02T00:00:00",
    "difficulty": "medium",
    "problem_format": "functional",
    "question_content": "Return the sum.",
    "starter_code": "class Solution:\n    def total(self, a: int, b: int) -> int:\n        ",
    "public_test_cases": "[]",
}


def manifest(problems=(STDIN_PROBLEM, FUNCTIONAL_PROBLEM)):
    return {"dataset_revision": pt.DATASET_REVISION, "problems": list(problems)}


def ok_body(content="```python\nprint(1)\n```", completion=50, reasoning=10, prompt=20, **message):
    return {
        "model": "m",
        "choices": [{"finish_reason": "stop", "message": {"content": content, **message}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class FakeBackend:
    def __init__(self, responder, served=None):
        self.responder = responder
        self.served = served
        self.payloads = []

    def backend(self):
        def send(payload, timeout_s):
            self.payloads.append(payload)
            status, body = self.responder(payload)
            text = body if isinstance(body, str) else json.dumps(body)
            return pt.SendResult(status, {"x-litellm-response-cost": "0.0001"}, text, None if status else "ReadTimeout", 1.5)

        return pt.Backend(send, lambda: self.served)


def run(tmp_path, fake, models=("Mango 1.5-9B",), **kw):
    options = dict(
        manifest=manifest(),
        models=list(models),
        refused=[],
        backend=fake.backend(),
        scrub=pt.Scrubber(FAKE_KEY),
        out_dir=tmp_path,
        max_tokens=1000,
        read_timeout_s=10,
        max_total_minutes=60,
        max_thb=50,
        stop_after_first_smoke=False,
        thinking_off_check=True,
        long_hold=True,
        out=lambda text: None,
    )
    options.update(kw)
    return pt.run_probe(**options)


# ------------------------------------------------------------------ prompts


def test_prompt_stdin_golden():
    expected = (
        "### Question:\nPrint A+B.\n\n"
        "### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
        "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
        "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
        "output to STDOUT.\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    assert pt.build_prompt("Print A+B.", "") == expected


def test_prompt_functional_golden():
    starter = "class Solution:\n    def f(self):\n        "
    expected = (
        "### Question:\nQ\n\n"
        "### Format: You will use the following starter code to write the solution to the problem and "
        "enclose your code within delimiters.\n"
        f"```python\n{starter}\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    assert pt.build_prompt("Q", starter) == expected


def test_planned_requests_have_no_system_message(tmp_path):
    planned = pt.plan_requests(manifest(), thinking_off_check=True, long_hold=True)
    assert [p.request_id for p in planned] == ["smoke", "smoke_off", "long_hold", "abc999_a", "3333"]
    for request in planned:
        payload = pt.build_payload("Mango 1.5-9B", request, 32768)
        assert [m["role"] for m in payload["messages"]] == ["user"]
        assert payload["temperature"] == 0.0 and payload["top_p"] == 1.0 and "seed" not in payload
    off = pt.build_payload("Mango 1.5-9B", planned[1], 32768)
    assert off["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "chat_template_kwargs" not in off


# --------------------------------------------------------------- extraction


@pytest.mark.parametrize(
    "text, expected",
    [
        ("no code here", ""),
        ("```python\nx = 1", ""),
        ("intro\n```python\nx = 1\ny = 2\n```\ndone", "x = 1\ny = 2"),
        ("```\nfirst\n```\ntext\n```python\nsecond\n```", "second"),
        ("```\na\n```\nprose between\n```\nb", "prose between"),
        ("see ```inline``` fences\ncode\n```", "code"),
    ],
)
def test_lcb_extraction_rule(text, expected):
    assert pt.lcb_extract_code(text) == expected


def test_think_close_helper():
    assert pt.after_last_think_close("a</think>b</think>c") == "c"
    assert pt.after_last_think_close("no tags") is None


def test_row_reports_extraction_before_and_after_think_strip():
    content = "<think>\n```\nnot code\n```\n</think>\n```python\nprint(1)\n```"
    result = pt.SendResult(200, {}, "", None, 2.0)
    row = pt.derive_row(
        model="Mango 1.5-9B", request_id="x", kind="problem", problem=STDIN_PROBLEM, max_tokens=100,
        result=result, body=ok_body(content), cost_thb=0.0, cost_basis="usage",
    )
    assert row["extracted_nonempty"] and row["extracted_parses"]
    assert row["extracted_after_think_nonempty"] and row["extracted_after_think_parses"]
    assert row["think_close_in_content"] == 1 and row["n_fence_lines"] == 4


def test_row_separate_reasoning_field_and_evidence():
    body = ok_body("answer", reasoning_content="r" * 500, completion=900, reasoning=800)
    body["reasoning_effort"] = "xhigh"
    result = pt.SendResult(200, {"x-litellm-model-id": "abc", "date": "x"}, "", None, 3.0)
    row = pt.derive_row(
        model="Qwen 3.8-27B", request_id="smoke", kind="smoke", problem=None, max_tokens=1000,
        result=result, body=body, cost_thb=0.0, cost_basis="usage",
    )
    assert row["reasoning_field"] == "reasoning_content" and row["reasoning_chars"] == 500
    assert row["settings_evidence"]["reasoning_effort"] == "xhigh"
    assert row["settings_evidence"]["choices[0].message.reasoning_content"] == "<str 500 chars>"
    assert row["header_evidence"] == {"x-litellm-model-id": "abc"}


# -------------------------------------------------------------- spend guard


def test_missing_price_refuses_model():
    runnable, refused = pt.resolve_models(["Mango 1.5-9B", "Qwen 3.8-27B-NVFP4"])
    assert runnable == ["Mango 1.5-9B"]
    assert refused[0]["model"] == "Qwen 3.8-27B-NVFP4"


def test_guard_stops_before_sending(tmp_path):
    fake = FakeBackend(lambda p: (200, ok_body()))
    # worst case for one Mango request at 1000 max_tokens is a small THB amount
    state, guard = run(tmp_path, fake, max_thb=0.001)
    assert fake.payloads == []
    assert state.stopped_reason.startswith("spend guard")


def test_guard_stops_once_total_would_pass_limit(tmp_path):
    fake = FakeBackend(lambda p: (200, ok_body(completion=1000, prompt=500)))
    state, guard = run(tmp_path, fake, max_thb=0.0165)
    assert 1 <= len(fake.payloads) < 5
    assert guard.spent_thb <= 0.0165
    assert state.stopped_reason.startswith("spend guard")


def test_failed_request_counts_full_worst_case(tmp_path):
    fake = FakeBackend(lambda p: (None, ""))
    state, guard = run(tmp_path, fake, thinking_off_check=False, long_hold=False)
    first = state.rows[0]
    assert first["cost_basis"] == "worst_case"
    est_in = pt.estimate_input_tokens(pt.SMOKE_PROMPT)
    assert first["cost_thb"] == pytest.approx(pt.worst_case_thb("Mango 1.5-9B", est_in, 1000), abs=1e-6)


def test_no_usage_counts_worst_case_and_usage_counts_actual(tmp_path):
    bodies = iter([{"choices": [{"finish_reason": "stop", "message": {"content": "1"}}]}, ok_body(completion=100, prompt=10)])
    fake = FakeBackend(lambda p: (200, next(bodies)))
    state, guard = run(tmp_path, fake, thinking_off_check=True, long_hold=False, manifest=manifest(problems=()))
    assert [r["cost_basis"] for r in state.rows] == ["worst_case", "usage"]
    assert state.rows[1]["cost_thb"] == pytest.approx(pt.usage_thb("Mango 1.5-9B", 10, 100), abs=1e-6)
    assert guard.spent_thb == pytest.approx(sum(r["cost_thb"] for r in state.rows), abs=1e-5)


def test_prices_cover_the_three_models():
    assert set(pt.MODEL_ORDER) == set(pt.PRICES_THB_PER_M)
    assert pt.PRICES_THB_PER_M["Mango 1.5-9B"] == (1.00, 1.00)
    assert pt.PRICES_THB_PER_M["Qwen 3.8-27B"] == (1.00, 1.00)
    assert pt.PRICES_THB_PER_M["Qwen 3.8-Flash-Next"] == (1.00, 1.00)


# ---------------------------------------------------------------- behaviour


def test_stop_after_first_smoke_sends_exactly_one(tmp_path):
    fake = FakeBackend(lambda p: (200, ok_body()))
    state, _ = run(tmp_path, fake, models=("Mango 1.5-9B", "Qwen 3.8-27B"), stop_after_first_smoke=True)
    assert len(fake.payloads) == 1
    assert state.stopped_reason == "--stop-after-first-smoke"


def test_no_retries_and_empty_length_is_recorded(tmp_path):
    body = ok_body("", completion=1000, reasoning=1000)
    body["choices"][0]["finish_reason"] = "length"
    fake = FakeBackend(lambda p: (200, body))
    state, _ = run(tmp_path, fake, thinking_off_check=False, long_hold=False)
    assert len(fake.payloads) == 1 + 2  # smoke + two problems, one request each
    assert all(r["cap_hit"] and r["content_empty"] and r["finish_reason"] == "length" for r in state.rows)


def test_smoke_rejection_skips_the_rest_of_that_model(tmp_path):
    fake = FakeBackend(lambda p: (400, {"error": {"message": "max_tokens too large"}}))
    state, _ = run(tmp_path, fake, models=("Mango 1.5-9B", "Qwen 3.8-27B"))
    assert len(fake.payloads) == 2  # one smoke per model
    assert len(state.skipped) == 2


def test_unserved_model_is_skipped(tmp_path):
    fake = FakeBackend(lambda p: (200, ok_body()), served=["Mango 1.5-9B"])
    state, _ = run(tmp_path, fake, models=("Qwen 3.8-27B",))
    assert fake.payloads == [] and state.skipped[0]["model"] == "Qwen 3.8-27B"


def test_consecutive_transport_failures_abort_model(tmp_path):
    fake = FakeBackend(lambda p: (None, ""))
    state, _ = run(tmp_path, fake)
    assert len(fake.payloads) == pt.MAX_CONSECUTIVE_TRANSPORT_FAILURES


# ------------------------------------------------------- key and dry run


def write_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    return path


def test_dry_run_makes_no_http_call_and_reads_no_key(tmp_path, monkeypatch, capsys):
    import httpx

    def boom(*args, **kwargs):
        raise AssertionError("httpx.Client constructed during a dry run")

    monkeypatch.setattr(httpx, "Client", boom)
    monkeypatch.delenv("ORION_API_KEY", raising=False)
    out_root = tmp_path / "out"
    args = pt.build_parser().parse_args(["run", "--dry-run", "--manifest", str(write_manifest(tmp_path)), "--out-root", str(out_root)])
    assert pt.cmd_run(args) == 0
    printed = capsys.readouterr().out
    assert "dry run" in printed and "worst case total" in printed
    assert not out_root.exists()


def test_key_never_appears_in_output_or_files(tmp_path, monkeypatch, capsys):
    echoing = ok_body("final", completion=20, reasoning=5)
    echoing["note"] = f"server echoed {FAKE_KEY} by mistake"
    fake = FakeBackend(lambda p: (200, echoing), served=list(pt.MODEL_ORDER))
    monkeypatch.setattr(pt, "LOG_DIR", tmp_path / "logs")
    out_root = tmp_path / "out"
    args = pt.build_parser().parse_args(
        ["run", "--manifest", str(write_manifest(tmp_path)), "--out-root", str(out_root), "--no-stop-after-first-smoke", "--models", "Mango 1.5-9B"]
    )
    code = pt.cmd_run(args, backend_factory=lambda key, timeout: (fake.backend(), lambda: None), key_reader=lambda: FAKE_KEY)
    assert code == 0
    assert FAKE_KEY not in capsys.readouterr().out
    files = [p for p in list((out_root).rglob("*")) + list((tmp_path / "logs").rglob("*")) if p.is_file()]
    assert files
    for path in files:
        assert FAKE_KEY not in path.read_text(encoding="utf-8"), path
    for payload in fake.payloads:
        assert FAKE_KEY not in json.dumps(payload)
    meta = json.loads(next(out_root.rglob("run_meta.json")).read_text(encoding="utf-8"))
    assert meta["spent_thb"] > 0 and meta["max_thb"] == 50.0


# ------------------------------------------------------------------ prepare


def test_select_problems_cells_window_and_no_private_fields():
    rows = []
    for i in range(6):
        for fmt in ("stdin", "functional"):
            for difficulty in ("easy", "medium", "hard"):
                rows.append(
                    {
                        "question_id": f"{fmt}-{difficulty}-{i}",
                        "question_title": "t",
                        "question_content": "c",
                        "platform": "p",
                        "contest_date": f"2025-0{2 + i % 3}-10T00:00:00",
                        "starter_code": "" if fmt == "stdin" else "def f(): pass",
                        "difficulty": difficulty,
                        "public_test_cases": "[]",
                    }
                )
    rows.append({**rows[0], "question_id": "old", "contest_date": "2025-01-04T00:00:00"})
    chosen = pt.select_problems(rows, seed=0, max_chars=4000, window_start="2025-02-01", window_end="2025-04-30")
    assert chosen["max_contest_date"] == "2025-04-10"
    assert chosen["n_in_window"] == 36
    assert len(chosen["problems"]) == 5
    assert chosen["window_counts"]["stdin/easy"] == 6
    assert "old" not in {p["question_id"] for p in chosen["problems"]}
    assert all("private_test_cases" not in p for p in chosen["problems"])
    again = pt.select_problems(rows, seed=0, max_chars=4000, window_start="2025-02-01", window_end="2025-04-30")
    assert [p["question_id"] for p in again["problems"]] == [p["question_id"] for p in chosen["problems"]]


# ---------------------------------------------------------------- streaming


def sse(*chunks, done=True, comments=True):
    lines = [": keep-alive"] if comments else []
    for chunk in chunks:
        lines.append("data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk)))
        lines.append("")
    if done:
        lines.append("data: [DONE]")
    return lines


def delta(**fields):
    return {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def stream_lines():
    return sse(
        delta(role="assistant", content=""),
        delta(reasoning_content="think "),
        delta(reasoning_content="more"),
        delta(content="391"),
        {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c1", "model": "m", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16, "completion_tokens_details": {"reasoning_tokens": 5}}},
    )


def test_assemble_stream_folds_reasoning_content_and_usage():
    body, facts = pt.assemble_stream(stream_lines())
    message = body["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "391", "reasoning_content": "think more"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5
    assert facts == {"n_chunks": 6, "complete": True}


def test_assemble_stream_incomplete_without_done_or_finish():
    body, facts = pt.assemble_stream(sse(delta(content="par"), done=False))
    assert body["choices"][0]["message"]["content"] == "par"
    assert body["choices"][0]["finish_reason"] is None
    assert "usage" not in body and facts["complete"] is False


def test_assemble_stream_ignores_malformed_lines_and_records_error_chunk():
    body, facts = pt.assemble_stream(sse("{not json", {"error": {"message": "boom"}}, delta(content="x")))
    assert body["error"] == {"message": "boom"}
    assert body["choices"][0]["message"]["content"] == "x"
    assert facts["complete"] is False


def test_assemble_stream_accepts_reasoning_field_name():
    body, _ = pt.assemble_stream(sse(delta(reasoning="why")))
    assert body["choices"][0]["message"]["reasoning_content"] == "why"


def test_stream_payload_asks_for_usage_and_only_when_requested():
    request = pt.plan_requests(manifest(), thinking_off_check=False, long_hold=False)[0]
    plain = pt.build_payload("Mango 1.5-9B", request, 100)
    streamed = pt.build_payload("Mango 1.5-9B", request, 100, stream=True)
    assert "stream" not in plain and "stream_options" not in plain
    assert streamed["stream"] is True and streamed["stream_options"] == {"include_usage": True}


def test_streamed_run_saves_sse_file_scrubs_key_and_records_stream_facts(tmp_path):
    lines = stream_lines()
    lines.insert(1, f"data: {json.dumps(delta(content=FAKE_KEY))}")
    body, facts = pt.assemble_stream(lines)

    def send(payload, timeout_s):
        assert payload["stream"] is True
        return pt.SendResult(
            200, {}, json.dumps(body), None, 2.0,
            stream_lines=lines, n_chunks=facts["n_chunks"], ttft_s=0.4, stream_complete=facts["complete"],
        )

    backend = pt.Backend(send, lambda: None)
    state, guard = run(
        tmp_path, FakeBackend(lambda p: (200, {})), backend=backend, stream=True,
        thinking_off_check=False, long_hold=False, manifest=manifest(problems=()),
    )
    row = state.rows[0]
    assert row["streamed"] and row["stream_complete"] and row["ttft_s"] == 0.4 and row["n_chunks"] == facts["n_chunks"]
    assert row["cost_basis"] == "usage"
    sse_file = next((tmp_path / "raw").glob("*.sse.txt"))
    assert FAKE_KEY not in sse_file.read_text(encoding="utf-8")
    assert FAKE_KEY not in next((tmp_path / "raw").glob("*.json")).read_text(encoding="utf-8")


def test_incomplete_stream_without_usage_is_charged_worst_case(tmp_path):
    lines = sse(delta(content="par"), done=False)
    body, facts = pt.assemble_stream(lines)

    def send(payload, timeout_s):
        return pt.SendResult(200, {}, json.dumps(body), "StreamInterrupted", 3.0, stream_lines=lines, n_chunks=1, ttft_s=0.5, stream_complete=False)

    state, _ = run(
        tmp_path, FakeBackend(lambda p: (200, {})), backend=pt.Backend(send, lambda: None), stream=True,
        thinking_off_check=False, long_hold=False, manifest=manifest(problems=()),
    )
    assert state.rows[0]["cost_basis"] == "worst_case" and state.rows[0]["stream_complete"] is False


def test_dry_run_reports_streaming_mode(tmp_path, capsys):
    args = pt.build_parser().parse_args(["run", "--dry-run", "--stream", "--manifest", str(write_manifest(tmp_path))])
    assert pt.cmd_run(args) == 0
    assert "streaming" in capsys.readouterr().out
