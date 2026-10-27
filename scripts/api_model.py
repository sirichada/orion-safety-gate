#!/usr/bin/env python3
"""
HTTP backend for model_wrapper.py's generation seam: a second implementation of
generate(prompts, max_new_tokens, temperature, top_p, seed=None, chat_template=None) -> list[str]
against an OpenAI-compatible /chat/completions endpoint.

Every prompt is sent as exactly one user message; no system message is ever added. The server
applies its own chat template and input-length limit, so chat_template is rejected rather than
ignored. Every request is streamed (a reply that thinks can take many minutes, far past the
gateway's non-streaming limit) and the stream is folded into one reply. One request per
prompt, at most `concurrency` in flight at once (never more than CONCURRENCY_CEILING), and
results come back in input order. After a call, generate.meta holds one dict per returned
reply (finish_reason, attempts, usage, latency_s, cost_thb, reasoning_content), aligned with
the returned list; generate.finish_reasons and generate.attempts are views of it.

A reply that is empty and cut off at the token cap is re-requested up to `no_answer_retries`
more times with identical parameters (0 by default: every re-request re-spends the reply's
tokens), and the first non-empty result wins.

Spend is bounded by an optional SpendGuard: before every request its worst case (estimated
input plus the full token cap at the output price) must fit under the limit, and a request that
fails after it may have run is charged that worst case. The guard's running total is kept in a
ledger file shared by every case of a run.

Requires ORION_API_KEY in the environment. The key is only ever placed in the Authorization
header of the HTTP client and is never printed or written anywhere.

Usage:
    python api_model.py    # prints the model ids the endpoint serves
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

from run_config import API_CONCURRENCY_CEILING, DEFAULTS, BudgetExceeded

BASE_URL = "https://api.orion.cmkl.ai/v1"
API_KEY_ENV = "ORION_API_KEY"

MAX_ATTEMPTS = 6  # 429 / 5xx responses, with backoff
MAX_STREAM_ATTEMPTS = 3  # transport errors and streams that ended without a finish_reason
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
CONNECT_TIMEOUT_S = 30.0
# Longest silence between two chunks of one reply, and longest one reply may take in total.
READ_TIMEOUT_S = 900.0
MAX_REQUEST_S = 3600.0

# Transient: retried in the backend and never counted against the failure budget. Every other
# non-200 status, a malformed 200 body, and a retryable error that outlasts its attempts raise
# GenerationFailed, which model_wrapper.py counts as a failed generation.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# A rejection before generation starts is free; any other failure may have generated tokens.
UNCHARGED_STATUS = frozenset({429})

LEDGER_REPLACE_ATTEMPTS = 5
LEDGER_REPLACE_SLEEP_S = 0.05

# THB per 1M tokens (input, output), as of 2026-10-13.
PRICES_THB_PER_M = {
    "Mango 1.5-9B": (1.00, 1.00),
    "Qwen 3.8-27B": (1.00, 1.00),
    "Qwen 3.8-Flash-Next": (1.00, 1.00),
}

_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]+")


class GenerationFailed(Exception):
    """A request that exhausted its retries or hit a non-retryable error."""


def _redact(text: str) -> str:
    return _KEY_PATTERN.sub("sk-***", text)


def _read_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        sys.exit(f"{API_KEY_ENV} is not set. Set it in the environment before using --backend api.")
    return key


def _backoff_s(attempt: int) -> float:
    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt) * random.uniform(0.5, 1.0)


# ---------------------------------------------------------------- spend


def estimate_input_tokens(text: str) -> int:
    """A deliberate over-estimate for the guard; billing uses the reported usage."""
    return math.ceil(len(text) / 2.0) + 16


def worst_case_thb(model: str, est_input_tokens: int, max_tokens: int) -> float:
    price_in, price_out = PRICES_THB_PER_M[model]
    return (est_input_tokens * price_in + max_tokens * price_out) / 1_000_000


def usage_thb(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES_THB_PER_M[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


class SpendGuard:
    """Computed spend in THB against a limit, with the running total kept in a ledger file.

    reserve() is called before a request and settle() after it. Requests in flight count at
    their worst case, so the limit holds even with several outstanding.
    """

    def __init__(self, model: str, limit_thb: float, ledger_path=None):
        if model not in PRICES_THB_PER_M:
            raise ValueError(f"no price for model {model!r}; refusing to run without a spend bound")
        if not limit_thb > 0:
            raise ValueError("the spend limit must be positive")
        self.model = model
        self.limit_thb = limit_thb
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.reserved_thb = 0.0
        self._lock = threading.Lock()
        self.spent_thb = 0.0
        if self.ledger_path is not None and self.ledger_path.exists():
            self.spent_thb = float(json.loads(self.ledger_path.read_text(encoding="utf-8"))["spent_thb"])

    def reserve(self, worst_case: float) -> None:
        with self._lock:
            if self.spent_thb + self.reserved_thb + worst_case > self.limit_thb + 1e-9:
                raise BudgetExceeded(
                    f"spend limit {self.limit_thb:.4f} THB: {self.spent_thb:.4f} spent, "
                    f"{self.reserved_thb:.4f} reserved in flight, next request worst case {worst_case:.4f}"
                )
            self.reserved_thb += worst_case

    def settle(self, worst_case: float, charge: float) -> None:
        with self._lock:
            self.reserved_thb -= worst_case
            self.spent_thb += charge
            self._write_ledger()

    def _write_ledger(self) -> None:
        if self.ledger_path is None:
            return
        payload = {
            "spent_thb": round(self.spent_thb, 6),
            "limit_thb": self.limit_thb,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Another process (an indexer, a virus scanner) can briefly hold the file on Windows.
        for attempt in range(LEDGER_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, self.ledger_path)
                return
            except OSError:
                if attempt == LEDGER_REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(LEDGER_REPLACE_SLEEP_S)


# ---------------------------------------------------------------- stream


def assemble_stream(lines: list[str]) -> tuple[dict, dict]:
    """Folds SSE `data:` lines into a chat.completion-shaped body plus stream facts.

    A usage-only final chunk (empty choices) is expected when stream_options.include_usage is set.
    """
    content: list[str] = []
    reasoning: list[str] = []
    finish = usage = error = None
    done = False
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            error = chunk["error"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for name in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(name), str):
                    reasoning.append(delta[name])
                    break
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    body = {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "finish_reason": finish,
        "usage": usage,
        "error": error,
    }
    return body, {"complete": done and finish is not None and error is None}


def _stream_once(client: httpx.Client, payload: dict) -> tuple[int | None, dict | None, str, str | None]:
    """One streamed request: (status, folded body or None, error text, retry-after header)."""
    started = time.monotonic()
    lines: list[str] = []
    try:
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                text = resp.read().decode("utf-8", errors="replace")
                return resp.status_code, None, _redact(text[:300]), resp.headers.get("retry-after")
            for line in resp.iter_lines():
                lines.append(line)
                if time.monotonic() - started > MAX_REQUEST_S:
                    return 200, None, "WallClockTimeout", None
    except (httpx.TimeoutException, httpx.TransportError) as e:
        return None, None, type(e).__name__, None
    body, facts = assemble_stream(lines)
    if not facts["complete"]:
        return 200, None, "StreamIncomplete" if body["error"] is None else f"StreamError: {_redact(str(body['error'])[:200])}", None
    return 200, body, "", None


def _complete(client: httpx.Client, payload: dict, reserve=None, settle=None, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Sends one streamed request with retries; returns the folded body.

    reserve() is called before every attempt and, for an attempt that fails after it may have
    generated tokens, settle(True) is called (settle(False) for a rejection that is free), so the
    spend guard sees what a retry re-spends. After a successful attempt the reserve is left
    outstanding for the caller to settle with the reported usage.
    """
    last_error = "unknown"
    stream_failures = 0
    for attempt in range(max_attempts):
        if reserve is not None:
            reserve()
        try:
            status, body, error, retry_after = _stream_once(client, payload)
        except BaseException:
            if settle is not None:
                settle(True)
            raise
        if body is not None:
            return body
        if settle is not None:
            settle(status not in UNCHARGED_STATUS)
        last_error = f"HTTP {status}: {error}" if status not in (None, 200) and error else (
            f"HTTP {status}" if status not in (None, 200) else error
        )
        if status is not None and status != 200 and status not in RETRYABLE_STATUS:
            raise GenerationFailed(f"HTTP {status}: {error}")
        if status in (None, 200):
            stream_failures += 1
            if stream_failures >= MAX_STREAM_ATTEMPTS:
                break
        delay = _backoff_s(attempt)
        if retry_after is not None:
            try:
                delay = min(BACKOFF_CAP_S, float(retry_after))
            except ValueError:
                pass
        if attempt < max_attempts - 1:
            print(f"API retry: {last_error}, attempt {attempt + 1}/{max_attempts}, sleeping {delay:.1f}s")
            time.sleep(delay)
    raise GenerationFailed(f"still failing after retries (last: {last_error})")


def list_models(client: httpx.Client) -> list[str] | None:
    resp = client.get("/models")
    if resp.status_code != 200:
        return None
    return [m["id"] for m in resp.json().get("data", [])]


def make_api_generate(
    model_id: str,
    concurrency: int = DEFAULTS["api_concurrency"],
    base_url: str = BASE_URL,
    no_answer_retries: int = DEFAULTS["no_answer_retries"],
    spend: SpendGuard | None = None,
    transport: httpx.BaseTransport | None = None,
    extra_params: dict | None = None,
    max_attempts: int = MAX_ATTEMPTS,
):
    if not 1 <= concurrency <= API_CONCURRENCY_CEILING:
        raise ValueError(f"concurrency must be between 1 and {API_CONCURRENCY_CEILING}, got {concurrency}")
    if no_answer_retries < 0:
        raise ValueError("no_answer_retries must be >= 0")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {_read_api_key()}"},
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        transport=transport,
    )

    available = list_models(client)
    if available is None:
        print("WARNING: GET /models unavailable; cannot verify the model id before generating")
    elif model_id not in available:
        sys.exit(f"Model id {model_id!r} not served. Available: {available}")

    stop = threading.Event()

    def generate(
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        chat_template: str | None = None,
    ) -> list[str]:
        if chat_template is not None:
            raise ValueError("chat_template is not supported by the API backend: the server applies its own template")

        def one(prompt: str) -> tuple[str, dict]:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if seed is not None:
                payload["seed"] = seed
            # Extra sampling fields: top-level ones such as presence_penalty, or an extra_body object holding top_k.
            payload.update(extra_params or {})
            worst = worst_case_thb(model_id, estimate_input_tokens(prompt), max_new_tokens) if spend is not None else 0.0
            spent = [0.0]
            attempts = 0
            started = time.monotonic()

            def reserve():
                if spend is not None:
                    spend.reserve(worst)

            def settle(charged: bool):
                if spend is not None:
                    charge = worst if charged else 0.0
                    spend.settle(worst, charge)
                    spent[0] += charge

            def meta_for(finish, usage, reasoning, failed=False, error=None):
                return {
                    "finish_reason": finish,
                    "attempts": attempts,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "latency_s": round(time.monotonic() - started, 3),
                    "cost_thb": round(spent[0], 6) if spend is not None else None,
                    "reasoning_content": reasoning,
                    "failed": failed,
                    "error": error,
                }

            while True:
                if stop.is_set():
                    raise BudgetExceeded("stopped: another request in this batch exceeded the spend limit")
                try:
                    body = _complete(client, payload, reserve, settle, max_attempts)
                except GenerationFailed as e:
                    return "", meta_for(None, {}, "", failed=True, error=str(e))
                attempts += 1
                usage = body["usage"] or {}
                if spend is not None:
                    known = isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int)
                    charge = usage_thb(model_id, usage["prompt_tokens"], usage["completion_tokens"]) if known else worst
                    spend.settle(worst, charge)
                    spent[0] += charge
                text, finish = body["content"], body["finish_reason"]
                if text.strip() or finish != "length" or attempts > no_answer_retries:
                    return text, meta_for(finish, usage, body["reasoning_content"])
                print(f"API no-answer retry: empty content at the token cap, attempt {attempts}/{no_answer_retries + 1}")

        pool = ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(prompts))))
        try:
            futures = [pool.submit(one, p) for p in prompts]
            results = []
            for f in futures:
                try:
                    results.append(f.result())  # submission order == input order; raises on first failure
                except BudgetExceeded:
                    stop.set()
                    raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            stop.clear()

        generate.meta = [meta for _, meta in results]
        generate.finish_reasons = [meta["finish_reason"] for meta in generate.meta]
        generate.attempts = [meta["attempts"] for meta in generate.meta]
        return [text for text, _ in results]

    generate.meta = []
    generate.finish_reasons = []
    generate.attempts = []
    return generate


if __name__ == "__main__":
    with httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {_read_api_key()}"}) as c:
        print(list_models(c))
