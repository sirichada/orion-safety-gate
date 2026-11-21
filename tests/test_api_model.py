import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

import api_model
import run_config
from conftest import TEST_PRICES, write_prices

MODEL = "Mango 1.5-9B"
BASE = "https://api.example/v1"
# Built at run time so no key-shaped literal sits in the file.
FAKE_KEY = "sk-" + "FAKEKEYFORTESTS0123456789"
SECRET_IN_BODY = "sk-" + "SECRETSECRET1234"


def sse(content="", reasoning="", finish="stop", usage=(10, 20), done=True, pieces=2):
    """A streamed chat completion: reasoning chunks, content chunks, a finish chunk, a usage chunk."""
    lines = []

    def chunk(delta=None, finish_reason=None, usage_obj=None):
        body = {"id": "c1", "model": MODEL, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]}
        if usage_obj is not None:
            body = {"id": "c1", "model": MODEL, "choices": [], "usage": usage_obj}
        lines.append("data: " + json.dumps(body))

    chunk({"role": "assistant"})
    for text in (reasoning[: len(reasoning) // pieces], reasoning[len(reasoning) // pieces :]):
        if text:
            chunk({"reasoning_content": text})
    for text in (content[: len(content) // pieces], content[len(content) // pieces :]):
        if text:
            chunk({"content": text})
    if finish:
        chunk({}, finish_reason=finish)
    if usage:
        chunk(usage_obj={"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)})
    if done:
        lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def stream_response(text):
    return httpx.Response(200, content=text.encode("utf-8"), headers={"content-type": "text/event-stream"})


class Server:
    """A fake gateway: a queue of canned responses per call, and a log of what was sent."""

    def __init__(self, *responses, default=None, delay=0.0):
        self.responses = list(responses)
        self.default = default
        self.delay = delay
        self.requests = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        with self.lock:
            self.requests.append(json.loads(request.content))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            response = self.responses.pop(0) if self.responses else self.default
        try:
            if self.delay:
                time.sleep(self.delay)
            return response() if callable(response) else response
        finally:
            with self.lock:
                self.in_flight -= 1

    def transport(self):
        return httpx.MockTransport(self)


@pytest.fixture(autouse=True)
def fast_and_keyed(monkeypatch):
    monkeypatch.setenv(api_model.API_KEY_ENV, FAKE_KEY)
    monkeypatch.setattr(api_model.time, "sleep", lambda s: None)


def make(server, **kwargs):
    return api_model.make_api_generate(MODEL, base_url=BASE, transport=server.transport(), **kwargs)


# ------------------------------------------------------------------ stream


def test_assemble_stream_folds_content_reasoning_usage_and_finish():
    body, facts = api_model.assemble_stream(sse("hello world", "let me think", finish="stop", usage=(5, 9)).splitlines())
    assert body["content"] == "hello world" and body["reasoning_content"] == "let me think"
    assert body["finish_reason"] == "stop" and body["usage"]["completion_tokens"] == 9
    assert facts["complete"] is True


@pytest.mark.parametrize("text", [sse("a", done=False), sse("a", finish=None)])
def test_a_stream_without_done_or_finish_is_incomplete(text):
    _, facts = api_model.assemble_stream(text.splitlines())
    assert facts["complete"] is False


def test_assemble_stream_ignores_comments_and_garbage():
    lines = [": keep-alive", "data: not json", "data: [1,2]"] + sse("ok").splitlines()
    body, facts = api_model.assemble_stream(lines)
    assert body["content"] == "ok" and facts["complete"]


# ---------------------------------------------------------------- generate


def test_generate_streams_one_user_message_and_returns_the_content_only():
    server = Server(default=stream_response(sse("42", reasoning="17 * 23 = 391?", usage=(8, 30))))
    generate = make(server)
    texts = generate(["What is 17 * 23?"], 256, 0.0, 1.0, seed=7)
    assert texts == ["42"]
    sent = server.requests[0]
    assert sent["messages"] == [{"role": "user", "content": "What is 17 * 23?"}]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert (sent["max_tokens"], sent["temperature"], sent["top_p"], sent["seed"]) == (256, 0.0, 1.0, 7)
    meta = generate.meta[0]
    assert meta["finish_reason"] == "stop" and meta["attempts"] == 1 and meta["failed"] is False
    assert (meta["prompt_tokens"], meta["completion_tokens"]) == (8, 30)
    assert meta["reasoning_content"] == "17 * 23 = 391?" and meta["cost_thb"] is None
    assert generate.finish_reasons == ["stop"] and generate.attempts == [1]


def test_results_keep_input_order_under_concurrency():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        prompt = json.loads(request.content)["messages"][0]["content"]
        time.sleep(0.05 if prompt == "p0" else 0.0)  # the first prompt finishes last
        return stream_response(sse("reply to " + prompt))

    generate = api_model.make_api_generate(MODEL, base_url=BASE, transport=httpx.MockTransport(handler), concurrency=3)
    prompts = [f"p{i}" for i in range(6)]
    assert generate(prompts, 64, 0.0, 1.0) == [f"reply to {p}" for p in prompts]


def test_in_flight_requests_never_exceed_the_concurrency_setting():
    server = Server(default=stream_response(sse("ok")), delay=0.03)
    generate = make(server, concurrency=2)
    assert generate([f"p{i}" for i in range(8)], 64, 0.0, 1.0) == ["ok"] * 8
    assert server.max_in_flight <= 2


@pytest.mark.parametrize("bad", [0, 4, 8])
def test_concurrency_above_the_ceiling_is_refused_before_any_request(bad):
    server = Server(default=stream_response(sse("ok")))
    with pytest.raises(ValueError, match="between 1 and 3"):
        make(server, concurrency=bad)
    assert server.requests == []


def test_default_concurrency_is_the_ceiling():
    assert run_config.DEFAULTS["api_concurrency"] == run_config.API_CONCURRENCY_CEILING == 3


def test_chat_template_is_rejected():
    generate = make(Server(default=stream_response(sse("ok"))))
    with pytest.raises(ValueError, match="chat_template"):
        generate(["x"], 8, 0.0, 1.0, chat_template="T")


# ------------------------------------------------------------------ retries


def test_429_is_retried_and_free_and_5xx_is_retried_and_charged_worst_case(tmp_path):
    guard = api_model.SpendGuard(MODEL, 10.0, tmp_path / "ledger.json")
    server = Server(
        httpx.Response(429, headers={"retry-after": "1"}, text="slow down"),
        httpx.Response(502, text="bad gateway"),
        stream_response(sse("done", usage=(10, 100))),
    )
    generate = make(server, spend=guard)
    assert generate(["hello"], 1000, 0.0, 1.0) == ["done"]
    assert len(server.requests) == 3 and generate.meta[0]["failed"] is False
    worst = api_model.worst_case_thb(MODEL, api_model.estimate_input_tokens("hello"), 1000)
    expected = worst + api_model.usage_thb(MODEL, 10, 100)
    assert guard.spent_thb == pytest.approx(expected)
    assert generate.meta[0]["cost_thb"] == pytest.approx(expected, abs=1e-6)


def test_a_non_retryable_status_marks_the_prompt_failed_without_retrying():
    server = Server(default=httpx.Response(400, text=f"{SECRET_IN_BODY} bad request"))
    generate = make(server)
    texts = generate(["x"], 64, 0.0, 1.0)
    assert texts == [""] and len(server.requests) == 1
    meta = generate.meta[0]
    assert meta["failed"] is True and meta["finish_reason"] is None
    assert "HTTP 400" in meta["error"] and "SECRET" not in meta["error"]


def test_an_incomplete_stream_is_retried_up_to_three_times_then_marked_failed():
    server = Server(default=stream_response(sse("partial", done=False)))
    generate = make(server)
    assert generate(["x"], 64, 0.0, 1.0) == [""]
    assert len(server.requests) == api_model.MAX_STREAM_ATTEMPTS
    assert generate.meta[0]["failed"] is True


def test_an_incomplete_stream_that_recovers_returns_the_good_reply():
    server = Server(stream_response(sse("cut", done=False)), stream_response(sse("whole")))
    generate = make(server)
    assert generate(["x"], 64, 0.0, 1.0) == ["whole"]
    assert generate.meta[0]["attempts"] == 1  # attempts counts completed replies, not transport retries


def test_transport_errors_are_retried():
    calls = {"n": 0}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return stream_response(sse("fine"))

    generate = api_model.make_api_generate(MODEL, base_url=BASE, transport=httpx.MockTransport(handler))
    assert generate(["x"], 64, 0.0, 1.0) == ["fine"] and calls["n"] == 2


# ------------------------------------------------------------- no-answer


def test_an_empty_reply_at_the_cap_is_not_re_requested_by_default():
    server = Server(default=stream_response(sse("", reasoning="thinking...", finish="length", usage=(5, 64))))
    generate = make(server)
    assert generate(["x"], 64, 0.0, 1.0) == [""]
    assert len(server.requests) == 1 and generate.meta[0]["finish_reason"] == "length"


def test_no_answer_retries_re_request_until_an_answer_or_the_limit():
    empty = stream_response(sse("", reasoning="t", finish="length"))
    server = Server(empty, empty, stream_response(sse("finally")))
    generate = make(server, no_answer_retries=2)
    assert generate(["x"], 64, 0.0, 1.0) == ["finally"]
    assert len(server.requests) == 3 and generate.meta[0]["attempts"] == 3

    server = Server(default=empty)
    generate = make(server, no_answer_retries=1)
    assert generate(["x"], 64, 0.0, 1.0) == [""]
    assert len(server.requests) == 2


# ---------------------------------------------------------------- spend


def test_worst_case_and_usage_cost_follow_the_prices_file():
    price_in, price_out = TEST_PRICES["Qwen 3.8-Flash-Next"]
    assert api_model.worst_case_thb("Qwen 3.8-Flash-Next", 100, 1000) == pytest.approx((100 * price_in + 1000 * price_out) / 1e6)
    price_in, price_out = TEST_PRICES["Mango 1.5-9B"]
    assert api_model.usage_thb("Mango 1.5-9B", 1000, 2000) == pytest.approx((1000 * price_in + 2000 * price_out) / 1e6)


def test_prices_are_read_when_needed_and_follow_the_file(tmp_path, monkeypatch):
    path = tmp_path / "own_prices.json"
    write_prices(path, {"M": (1.0, 2.0)})
    monkeypatch.setattr(api_model, "PRICES_PATH", path)
    assert api_model.price_of("M") == (1.0, 2.0)
    write_prices(path, {"M": (3.0, 4.0)})
    os.utime(path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    assert api_model.price_of("M") == (3.0, 4.0)


def test_a_missing_prices_file_is_a_reported_problem_not_an_import_error(tmp_path, monkeypatch):
    monkeypatch.setattr(api_model, "PRICES_PATH", tmp_path / "absent.json")
    assert "config/prices.json not found" in api_model.price_problem(MODEL)
    with pytest.raises(ValueError, match="not found"):
        api_model.SpendGuard(MODEL, 5.0)


def test_a_malformed_prices_file_is_a_reported_problem(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text('{"models": {"M": {"input": "x"}}}', encoding="utf-8")
    monkeypatch.setattr(api_model, "PRICES_PATH", path)
    assert "config/prices.json is malformed" in api_model.price_problem("M")


def test_importing_api_model_needs_no_prices_file(tmp_path):
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    code = "import sys; sys.path.insert(0, sys.argv[1]); import api_model; api_model.PRICES_PATH = api_model.Path(sys.argv[2]); print(api_model.price_problem('x'))"
    done = subprocess.run([sys.executable, "-I", "-c", code, str(scripts), str(tmp_path / "absent.json")], capture_output=True, text=True)
    assert done.returncode == 0 and "not found" in done.stdout


def test_spend_guard_refuses_an_unpriced_model_and_a_bad_limit():
    with pytest.raises(ValueError, match="no price"):
        api_model.SpendGuard("Unpriced", 5.0)
    with pytest.raises(ValueError, match="positive"):
        api_model.SpendGuard(MODEL, 0)


def test_guard_counts_requests_in_flight_at_their_worst_case():
    guard = api_model.SpendGuard(MODEL, 1.0)
    guard.reserve(0.6)
    with pytest.raises(run_config.BudgetExceeded):
        guard.reserve(0.5)
    guard.settle(0.6, 0.1)
    guard.reserve(0.5)  # fits now that the first settled at its real cost
    assert guard.spent_thb == pytest.approx(0.1)


def test_ledger_persists_the_total_across_guards(tmp_path):
    ledger = tmp_path / "ledger.json"
    first = api_model.SpendGuard(MODEL, 5.0, ledger)
    first.reserve(1.0)
    first.settle(1.0, 0.75)
    assert json.loads(ledger.read_text(encoding="utf-8"))["spent_thb"] == 0.75
    second = api_model.SpendGuard(MODEL, 5.0, ledger)
    assert second.spent_thb == 0.75


def test_generate_stops_before_sending_when_the_worst_case_does_not_fit():
    guard = api_model.SpendGuard(MODEL, 0.001)  # a 4096-token reply alone is a small THB amount
    server = Server(default=stream_response(sse("ok")))
    generate = make(server, spend=guard)
    with pytest.raises(run_config.BudgetExceeded):
        generate(["x", "y", "z"], 4096, 0.0, 1.0)
    assert server.requests == []


def test_generate_charges_actual_usage_after_each_reply(tmp_path):
    guard = api_model.SpendGuard(MODEL, 5.0, tmp_path / "ledger.json")
    server = Server(default=stream_response(sse("ok", usage=(100, 1000))))
    generate = make(server, spend=guard)
    generate(["a", "b"], 2048, 0.0, 1.0)
    assert guard.spent_thb == pytest.approx(2 * api_model.usage_thb(MODEL, 100, 1000))
    assert guard.reserved_thb == pytest.approx(0.0)
    assert [m["cost_thb"] for m in generate.meta] == pytest.approx([api_model.usage_thb(MODEL, 100, 1000)] * 2, abs=1e-6)


def test_a_reply_without_usage_is_charged_its_worst_case():
    guard = api_model.SpendGuard(MODEL, 5.0)
    server = Server(default=stream_response(sse("ok", usage=None)))
    generate = make(server, spend=guard)
    generate(["x"], 1000, 0.0, 1.0)
    assert guard.spent_thb == pytest.approx(api_model.worst_case_thb(MODEL, api_model.estimate_input_tokens("x"), 1000))


def test_the_key_is_only_sent_as_a_header():
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        assert "sk-FAKEKEY" not in request.content.decode()
        return stream_response(sse("ok"))

    api_model.make_api_generate(MODEL, base_url=BASE, transport=httpx.MockTransport(handler))(["x"], 8, 0.0, 1.0)
    assert set(seen) == {f"Bearer {FAKE_KEY}"}


def test_an_unserved_model_id_exits_before_generating():
    server = Server(default=stream_response(sse("ok")))
    with pytest.raises(SystemExit, match="not served"):
        api_model.make_api_generate("Other", base_url=BASE, transport=server.transport())
    assert server.requests == []


def test_ledger_write_retries_a_briefly_locked_file(tmp_path, monkeypatch):
    guard = api_model.SpendGuard(MODEL, 5.0, tmp_path / "ledger.json")
    real = api_model.os.replace
    failures = {"n": 2}

    def flaky(src, dst):
        if failures["n"]:
            failures["n"] -= 1
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(api_model.os, "replace", flaky)
    guard.reserve(0.1)
    guard.settle(0.1, 0.05)
    assert json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))["spent_thb"] == 0.05


def test_ledger_write_gives_up_after_a_few_attempts(tmp_path, monkeypatch):
    guard = api_model.SpendGuard(MODEL, 5.0, tmp_path / "ledger.json")
    calls = {"n": 0}

    def locked(src, dst):
        calls["n"] += 1
        raise PermissionError("locked")

    monkeypatch.setattr(api_model.os, "replace", locked)
    guard.reserve(0.1)
    with pytest.raises(PermissionError):
        guard.settle(0.1, 0.05)
    assert calls["n"] == api_model.LEDGER_REPLACE_ATTEMPTS


def test_retry_message_shows_the_redacted_error_body(capsys):
    server = Server(
        httpx.Response(500, text="model backend down sk-LEAKEDKEY12345 please retry"),
        stream_response(sse("ok")),
    )
    generate = make(server)
    assert generate(["x"], 64, 0.0, 1.0) == ["ok"]
    out = capsys.readouterr().out
    assert "API retry: HTTP 500: model backend down sk-*** please retry" in out
    assert "LEAKEDKEY" not in out


def test_extra_params_are_sent_at_the_top_level_of_the_request():
    server = Server(default=stream_response(sse("ok")))
    generate = make(server, extra_params={"top_k": 20})
    generate(["x"], 64, 1.0, 0.95)
    assert server.requests[0]["top_k"] == 20 and server.requests[0]["temperature"] == 1.0
    plain = Server(default=stream_response(sse("ok")))
    make(plain)(["x"], 64, 0.0, 1.0)
    assert "top_k" not in plain.requests[0]


def test_presence_penalty_and_top_k_extra_body_travel_together():
    server = Server(default=stream_response(sse("ok")))
    make(server, extra_params={"extra_body": {"top_k": 20}, "presence_penalty": 1.5})(["x"], 64, 1.0, 0.95)
    sent = server.requests[0]
    assert sent["presence_penalty"] == 1.5 and sent["extra_body"] == {"top_k": 20}


def test_max_attempts_one_sends_each_request_once():
    server = Server(default=httpx.Response(500, text="boom"))
    generate = make(server, max_attempts=1)
    assert generate(["x"], 64, 0.0, 1.0) == [""]
    assert len(server.requests) == 1 and generate.meta[0]["failed"] is True
    with pytest.raises(ValueError, match="max_attempts"):
        make(Server(), max_attempts=0)


def test_extra_body_param_is_nested_in_the_request():
    server = Server(default=stream_response(sse("ok")))
    make(server, extra_params={"extra_body": {"top_k": 20}})(["x"], 64, 1.0, 0.95)
    assert server.requests[0]["extra_body"] == {"top_k": 20} and "top_k" not in server.requests[0]
