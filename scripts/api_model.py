#!/usr/bin/env python3
"""
HTTP backend for model_wrapper.py's generation seam: a second implementation of
generate(prompts, max_new_tokens, temperature, top_p, seed=None, chat_template=None) -> list[str]
against an OpenAI-compatible /chat/completions endpoint.

Every prompt is sent as exactly one user message; no system message is ever added. The server
applies its own chat template and input-length limit, so chat_template is rejected rather than
ignored. One request per prompt: len(prompts) requests are in flight at once and results come
back in input order. finish_reason and the number of attempts for the most recent
successful call are exposed as generate.finish_reasons and generate.attempts (aligned with
the returned list).

A response that is empty and cut off at the token cap is re-requested up to NO_ANSWER_RETRIES
more times with identical parameters, and the first non-empty result wins: the endpoint's
output varies between identical requests, so a re-request can succeed where the first attempt
spent its whole budget before answering. Such retries are ordinary successful requests and
never count toward the failure budget.

Requires ORION_API_KEY in the environment. The key is only ever placed in the Authorization
header of the HTTP client and is never printed or written anywhere.

Usage:
    python api_model.py    # prints the model ids the endpoint serves
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE_URL = "https://api.orion.cmkl.ai/v1"
API_KEY_ENV = "ORION_API_KEY"

MAX_ATTEMPTS = 6
NO_ANSWER_RETRIES = 3
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
REQUEST_TIMEOUT_S = 300.0

# Transient: retried in the backend and never counted against the failure budget. Every other
# non-200 status, a malformed 200 body, and a retryable error that outlasts MAX_ATTEMPTS raise
# GenerationFailed, which model_wrapper.py counts as a failed generation.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]+")
_LIMIT_HEADER_PATTERN = re.compile(r"ratelimit|rate-limit|quota|retry-after", re.IGNORECASE)


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


def _retry_after_s(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return min(BACKOFF_CAP_S, float(value)) if value is not None else None
    except ValueError:
        return None


def _parse(resp: httpx.Response) -> tuple[str, str | None]:
    try:
        choice = resp.json()["choices"][0]
        content = choice["message"].get("content")
        return (content if content is not None else ""), choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise GenerationFailed(f"malformed 200 response ({type(e).__name__})") from e


def _complete(client: httpx.Client, payload: dict, limits_seen: list) -> tuple[str, str | None]:
    last_error = "unknown"
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_error = type(e).__name__
            delay = _backoff_s(attempt)
        else:
            if resp.status_code == 200:
                if not limits_seen:
                    limits_seen.append(True)
                    seen = {k: v for k, v in resp.headers.items() if _LIMIT_HEADER_PATTERN.search(k)}
                    print(f"API limit/quota headers: {seen or 'none advertised'}")
                    try:
                        print(f"API message fields: {sorted(resp.json()['choices'][0]['message'])}")
                    except (KeyError, IndexError, TypeError, ValueError):
                        pass
                return _parse(resp)
            if resp.status_code not in RETRYABLE_STATUS:
                raise GenerationFailed(f"HTTP {resp.status_code}: {_redact(resp.text[:300])}")
            last_error = f"HTTP {resp.status_code}"
            delay = _retry_after_s(resp) or _backoff_s(attempt)
        if attempt < MAX_ATTEMPTS - 1:
            print(f"API retry: {last_error}, attempt {attempt + 1}/{MAX_ATTEMPTS}, sleeping {delay:.1f}s")
            time.sleep(delay)
    raise GenerationFailed(f"still failing after {MAX_ATTEMPTS} attempts (last: {last_error})")


def list_models(client: httpx.Client) -> list[str] | None:
    resp = client.get("/models")
    if resp.status_code != 200:
        return None
    return [m["id"] for m in resp.json().get("data", [])]


def make_api_generate(model_id: str, max_workers: int = 8, base_url: str = BASE_URL):
    client = httpx.Client(
        base_url=base_url, headers={"Authorization": f"Bearer {_read_api_key()}"}, timeout=REQUEST_TIMEOUT_S
    )

    available = list_models(client)
    if available is None:
        print("WARNING: GET /models unavailable; cannot verify the model id before generating")
    elif model_id not in available:
        sys.exit(f"Model id {model_id!r} not served. Available: {available}")

    limits_seen: list = []

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

        def one(prompt: str) -> tuple[str, str | None, int]:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if seed is not None:
                payload["seed"] = seed
            attempts = 0
            while True:
                text, finish = _complete(client, payload, limits_seen)
                attempts += 1
                if text.strip() or finish != "length" or attempts > NO_ANSWER_RETRIES:
                    return text, finish, attempts
                print(f"API no-answer retry: empty content at the token cap, attempt {attempts}/{NO_ANSWER_RETRIES + 1}")

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(prompts)))) as pool:
            futures = [pool.submit(one, p) for p in prompts]
            results = [f.result() for f in futures]  # submission order == input order; raises on first failure

        generate.finish_reasons = [finish for _, finish, _ in results]
        generate.attempts = [n for _, _, n in results]
        return [text for text, _, _ in results]

    generate.finish_reasons = []
    generate.attempts = []
    return generate


if __name__ == "__main__":
    with httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {_read_api_key()}"}) as c:
        print(list_models(c))
