#!/usr/bin/env python3
"""
Measures how Orion's API models return thinking on coding prompts: whether they think by
default, where the trace comes back, how many tokens and seconds a reply costs at a large
cap, and whether the final content holds a code block under LiveCodeBench's extraction rule.

Standalone: imports only the key reading and redaction helpers from scripts/api_model.py and
never goes through model_wrapper or the orchestrator. No retries are made, and an empty
reply cut off at the cap is recorded rather than re-requested. Nothing here executes model
output; ast.parse is a syntax check only.

Subcommands:
    prepare    select problems from a local copy of the dataset's test6.jsonl and write a
               manifest of public fields (no API access, no key)
    run        send the probe requests (needs ORION_API_KEY); --dry-run prints the plan
               and makes no network call and reads no key
    reconcile  compare a finished run's computed spend with Orion's dashboard figure

Usage:
    python tools/probe_thinking.py prepare --test6 data/livecodebench/test6.jsonl
    python tools/probe_thinking.py run --dry-run
    python tools/probe_thinking.py run
    python tools/probe_thinking.py reconcile --run-dir runs/results/thinking_probe/<ts> --dashboard-thb 1.23
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import api_model  # noqa: E402
import run_config  # noqa: E402

DATASET_REPO = "livecodebench/code_generation_lite"
DATASET_FILE = "test6.jsonl"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

WINDOW_START = "2025-02-01"
WINDOW_END = "2025-04-30"

DEFAULT_MANIFEST = REPO_ROOT / "data" / "probe" / "thinking_probe_manifest.json"
DEFAULT_TEST6 = REPO_ROOT / "data" / "livecodebench" / DATASET_FILE
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "results" / "thinking_probe"
LOG_DIR = REPO_ROOT / "runs" / "logs"

MODEL_ORDER = ("Mango 1.5-9B", "Qwen 3.8-Flash-Next", "Qwen 3.8-27B")

# THB per 1M tokens (input, output), as of 2026-10-13.
PRICES_THB_PER_M = {
    "Mango 1.5-9B": (1.00, 1.00),
    "Qwen 3.8-27B": (1.00, 1.00),
    "Qwen 3.8-Flash-Next": (1.00, 1.00),
}

TARGET_CELLS = (
    ("stdin", "easy"),
    ("stdin", "medium"),
    ("stdin", "hard"),
    ("functional", "medium"),
    ("functional", "hard"),
)
FALLBACK_CELLS = (("functional", "easy"), ("stdin", "medium"), ("functional", "medium"))

TEMPERATURE = 0.0
TOP_P = 1.0
DEFAULT_MAX_TOKENS = 32768
CONNECT_TIMEOUT_S = 30.0
DEFAULT_READ_TIMEOUT_S = 1800.0
DEFAULT_MAX_TOTAL_MINUTES = 240.0
DEFAULT_MAX_THB = 50.0
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 2

SMOKE_PROMPT = "What is 17 * 23? Reply with just the number."
LONG_HOLD_PROMPT = "Write the integers from 1 to 4000 in order, separated by single spaces. Output only the numbers."
THINKING_OFF_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

SETS = ("planned", "repeat", "vendor_t1", "diversity")
DEFAULT_DIVERSITY_K = 6
DEFAULT_T1_REPEATS = 2
DIVERSITY_MAX_TOKENS = 64
DIVERSITY_PROMPT = "Name one random animal and describe it in one sentence."
# (condition label, sampling mode); "omitted" sends neither temperature nor top_p.
DIVERSITY_CONDITIONS = (("omitted", "omitted"), ("t0", "explicit_t0"), ("t1", "explicit_t1"))
SAMPLING_PARAMS = {"explicit_t0": (TEMPERATURE, TOP_P), "explicit_t1": (1.0, 0.95), "omitted": None}
MODEL_INFO_MAX_CHARS = 65536

INTERPRETATION_RULE = (
    "Diversity set, per model, on distinct normalised outputs out of k sends per condition "
    "(n_default, n_t0, n_t1): if fewer than 3 sends succeeded in any condition -> insufficient_data; "
    "if n_t1 < n_t0 + 2 the prompt cannot separate sampled from greedy -> not_informative; "
    "else if n_default <= n_t0 + 1 -> greedy_like; else if n_default >= n_t1 - 1 -> sampled_like; "
    "else inconclusive. Planned-vs-repeat variance is descriptive only (common prefix, token differences) "
    "and carries no verdict."
)

FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters."
)
FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
    "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)

_EVIDENCE_KEY = re.compile(r"effort|think|reason", re.IGNORECASE)
_EVIDENCE_HEADER = re.compile(r"ratelimit|rate-limit|quota|retry-after|litellm|cost|model|effort|think", re.IGNORECASE)
_EVIDENCE_SCALAR_MAX_CHARS = 64


# ------------------------------------------------------------------ prompts


def build_prompt(question_content: str, starter_code: str) -> str:
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def problem_format(starter_code: str) -> str:
    return "functional" if starter_code else "stdin"


def lcb_extract_code(text: str) -> str:
    """LiveCodeBench's rule: the lines between the last two lines containing a fence marker."""
    lines = text.split("\n")
    fence_lines = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_lines) < 2:
        return ""
    return "\n".join(lines[fence_lines[-2] + 1 : fence_lines[-1]])


def n_fence_lines(text: str) -> int:
    return sum(1 for line in text.split("\n") if "```" in line)


def after_last_think_close(text: str) -> str | None:
    marker = "</think>"
    return text.rsplit(marker, 1)[1] if marker in text else None


# -------------------------------------------------------------- spend guard


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
    """Tracks computed spend in THB and refuses a request whose worst case would pass the limit."""

    def __init__(self, limit_thb: float):
        self.limit_thb = limit_thb
        self.spent_thb = 0.0

    def can_send(self, model: str, est_input_tokens: int, max_tokens: int) -> bool:
        return self.spent_thb + worst_case_thb(model, est_input_tokens, max_tokens) <= self.limit_thb

    def charge_worst_case(self, model: str, est_input_tokens: int, max_tokens: int) -> float:
        amount = worst_case_thb(model, est_input_tokens, max_tokens)
        self.spent_thb += amount
        return amount

    def charge_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        amount = usage_thb(model, prompt_tokens, completion_tokens)
        self.spent_thb += amount
        return amount


# ------------------------------------------------------------- transport


@dataclass
class SendResult:
    status: int | None
    headers: dict
    body_text: str
    error: str | None
    elapsed_s: float
    # Set only for streamed requests: the raw SSE lines as received, the chunk count, time to the
    # first data line, and whether the stream ended with [DONE] after a finish_reason.
    stream_lines: list[str] | None = None
    n_chunks: int | None = None
    ttft_s: float | None = None
    stream_complete: bool | None = None


@dataclass
class Backend:
    send: Callable[[dict, float], SendResult]
    served_models: Callable[[], list[str] | None]
    # (url) -> (status or None, body text or error name); absent on backends that cannot look things up.
    get: Callable[[str], tuple[int | None, str]] | None = None


class Scrubber:
    """Removes the API key, and anything shaped like one, from text before it is written or printed."""

    def __init__(self, key: str | None):
        self.key = key

    def __call__(self, text: str) -> str:
        if self.key:
            text = text.replace(self.key, "sk-***")
        return api_model._redact(text)


def assemble_stream(lines: list[str]) -> tuple[dict, dict]:
    """Folds SSE `data:` lines into a chat.completion-shaped body plus stream facts.

    A usage-only final chunk (empty choices) is expected when stream_options.include_usage is set.
    """
    content: list[str] = []
    reasoning: list[str] = []
    finish = usage = model = response_id = role = error = None
    done = False
    n_chunks = 0
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
        n_chunks += 1
        if chunk.get("error"):
            error = chunk["error"]
        model = chunk.get("model", model)
        response_id = chunk.get("id", response_id)
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta.get("role"), str):
                role = delta["role"]
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for name in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(name), str):
                    reasoning.append(delta[name])
                    break
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    message = {"role": role or "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    body: dict = {
        "id": response_id,
        "model": model,
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
    }
    if usage is not None:
        body["usage"] = usage
    if error is not None:
        body["error"] = error
    return body, {"n_chunks": n_chunks, "complete": done and finish is not None and error is None}


def make_http_backend(key: str, read_timeout_s: float) -> tuple[Backend, Callable[[], None]]:
    import httpx

    base_url = api_model.read_base_url()
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(read_timeout_s, connect=CONNECT_TIMEOUT_S),
    )

    def send_stream(payload: dict, timeout_s: float) -> SendResult:
        started = time.monotonic()
        lines: list[str] = []
        status = None
        headers: dict = {}
        error = None
        ttft = None
        try:
            with client.stream("POST", "/chat/completions", json=payload) as resp:
                status, headers = resp.status_code, dict(resp.headers)
                if status != 200:
                    text = resp.read().decode("utf-8", errors="replace")
                    return SendResult(status, headers, text, None, time.monotonic() - started)
                for line in resp.iter_lines():
                    if ttft is None and line.startswith("data:"):
                        ttft = time.monotonic() - started
                    lines.append(line)
                    if time.monotonic() - started > timeout_s:
                        error = "WallClockTimeout"
                        break
        except (httpx.TimeoutException, httpx.TransportError) as e:
            error = type(e).__name__
        elapsed = time.monotonic() - started
        if status is None:
            return SendResult(None, {}, "", error, elapsed)
        body, facts = assemble_stream(lines)
        return SendResult(
            status, headers, json.dumps(body, ensure_ascii=False), error, elapsed,
            stream_lines=lines, n_chunks=facts["n_chunks"], ttft_s=ttft, stream_complete=facts["complete"] and error is None,
        )

    def send(payload: dict, timeout_s: float) -> SendResult:
        if payload.get("stream"):
            return send_stream(payload, timeout_s)
        started = time.monotonic()
        try:
            resp = client.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            return SendResult(None, {}, "", type(e).__name__, time.monotonic() - started)
        return SendResult(resp.status_code, dict(resp.headers), resp.text, None, time.monotonic() - started)

    def served_models() -> list[str] | None:
        try:
            return api_model.list_models(client)
        except (httpx.TimeoutException, httpx.TransportError):
            return None

    api_host = httpx.URL(base_url).host

    def get(url: str) -> tuple[int | None, str]:
        # Lookups are limited to the API host the key belongs to.
        if httpx.URL(url).host != api_host:
            return None, "refused: not the API host"
        try:
            resp = client.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            return None, type(e).__name__
        return resp.status_code, resp.text

    return Backend(send, served_models, get), client.close


# ------------------------------------------------------------ response rows


def find_setting_evidence(body, path: str = "") -> dict:
    """Paths in a response whose key names mention effort, thinking or reasoning.

    Short scalars are kept as-is; long strings and containers are described by size only, so a
    reasoning trace is never copied into the summary.
    """
    found: dict = {}
    if isinstance(body, dict):
        for key, value in body.items():
            where = f"{path}.{key}" if path else str(key)
            if _EVIDENCE_KEY.search(str(key)):
                if isinstance(value, str) and len(value) > _EVIDENCE_SCALAR_MAX_CHARS:
                    found[where] = f"<str {len(value)} chars>"
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    found[where] = value
                elif isinstance(value, dict):
                    found[where] = {k: v for k, v in value.items() if isinstance(v, (int, float, bool)) or v is None}
                else:
                    found[where] = f"<{type(value).__name__} len {len(value)}>"
            if isinstance(value, (dict, list)):
                found.update(find_setting_evidence(value, where))
    elif isinstance(body, list):
        for i, item in enumerate(body):
            found.update(find_setting_evidence(item, f"{path}[{i}]"))
    return found


def _int_or_none(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parses(code: str) -> bool | None:
    if not code.strip():
        return None
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def derive_row(
    *,
    model: str,
    request_id: str,
    kind: str,
    problem: dict | None,
    max_tokens: int,
    result: SendResult,
    body,
    cost_thb: float,
    cost_basis: str,
) -> dict:
    row: dict = {
        "model": model,
        "request_id": request_id,
        "kind": kind,
        "problem_format": problem["problem_format"] if problem else None,
        "difficulty": problem["difficulty"] if problem else None,
        "status": result.status,
        "error": result.error,
        "elapsed_s": round(result.elapsed_s, 2),
        "streamed": result.n_chunks is not None,
        "n_chunks": result.n_chunks,
        "ttft_s": round(result.ttft_s, 2) if result.ttft_s is not None else None,
        "stream_complete": result.stream_complete,
        "max_tokens": max_tokens,
        "cost_thb": round(cost_thb, 6),
        "cost_basis": cost_basis,
    }
    choice = None
    if isinstance(body, dict) and isinstance(body.get("choices"), list) and body["choices"]:
        first = body["choices"][0]
        choice = first if isinstance(first, dict) else None
    message = choice.get("message") if choice and isinstance(choice.get("message"), dict) else {}
    usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}

    content = message.get("content")
    content = content if isinstance(content, str) else ""
    reasoning_field = next(
        (name for name in ("reasoning_content", "reasoning") if isinstance(message.get(name), str) and message[name]),
        None,
    )
    reasoning_chars = len(message[reasoning_field]) if reasoning_field else 0
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    reasoning_tokens = _int_or_none(details.get("reasoning_tokens"))

    after_think = after_last_think_close(content)
    extracted = lcb_extract_code(content)
    extracted_stripped = lcb_extract_code(after_think) if after_think is not None else None
    fences = n_fence_lines(content)

    row.update(
        finish_reason=choice.get("finish_reason") if choice else None,
        message_fields=sorted(message) if message else [],
        response_model=body.get("model") if isinstance(body, dict) else None,
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=completion_tokens,
        total_tokens=_int_or_none(usage.get("total_tokens")),
        reasoning_tokens=reasoning_tokens,
        content_chars=len(content),
        content_empty=not content.strip(),
        reasoning_field=reasoning_field,
        reasoning_chars=reasoning_chars,
        think_open_in_content=content.count("<think>"),
        think_close_in_content=content.count("</think>"),
        cap_hit=completion_tokens is not None and completion_tokens >= max_tokens,
        tokens_per_s=round(completion_tokens / result.elapsed_s, 1)
        if completion_tokens and result.elapsed_s > 0
        else None,
        chars_per_completion_token=round((len(content) + reasoning_chars) / completion_tokens, 2)
        if completion_tokens
        else None,
        n_fence_lines=fences,
        odd_fence_count=fences % 2 == 1,
        extracted_nonempty=bool(extracted.strip()),
        extracted_chars=len(extracted),
        extracted_parses=_parses(extracted),
        extracted_after_think_nonempty=bool(extracted_stripped.strip()) if extracted_stripped is not None else None,
        extracted_after_think_parses=_parses(extracted_stripped) if extracted_stripped is not None else None,
        settings_evidence=find_setting_evidence(body) if isinstance(body, (dict, list)) else {},
        header_evidence={k: v for k, v in result.headers.items() if _EVIDENCE_HEADER.search(k)},
    )
    if problem and problem["problem_format"] == "functional":
        match = re.search(r"def\s+(\w+)\s*\(", problem.get("starter_code", ""))
        method = match.group(1) if match else None
        row["starter_method_present"] = bool(method) and method in extracted
        row["class_solution_present"] = "class Solution" in extracted
    return row


# --------------------------------------------------------------- run plan


@dataclass
class PlannedRequest:
    request_id: str
    kind: str
    prompt: str
    problem: dict | None
    extra_body: dict | None = None
    set_name: str = "planned"
    sampling_mode: str = "explicit_t0"
    group_id: str | None = None
    # None means the run-wide --max-tokens applies.
    max_tokens: int | None = None


def effective_max_tokens(request: PlannedRequest, default: int) -> int:
    return request.max_tokens if request.max_tokens is not None else default


def plan_requests(
    manifest: dict,
    *,
    thinking_off_check: bool,
    long_hold: bool,
    sets: tuple[str, ...] = ("planned",),
    diversity_k: int = DEFAULT_DIVERSITY_K,
    problem_ids: tuple[str, ...] | None = None,
    t1_repeats: int = DEFAULT_T1_REPEATS,
) -> list[PlannedRequest]:
    """Requests in send order: planned, repeat, vendor_t1, then diversity.

    problem_ids restricts every set that uses the manifest's problems to those ids.
    """
    unknown = [name for name in sets if name not in SETS]
    if unknown:
        raise ValueError(f"unknown request set(s) {unknown}; choose from {', '.join(SETS)}")
    if diversity_k < 1:
        raise ValueError("diversity_k must be >= 1")
    if t1_repeats < 1:
        raise ValueError("t1_repeats must be >= 1")
    problems = manifest["problems"]
    if problem_ids is not None:
        known = {problem["question_id"] for problem in problems}
        missing = [pid for pid in problem_ids if pid not in known]
        if missing:
            raise ValueError(f"unknown problem id(s) {missing}; manifest has {sorted(known)}")
        problems = [problem for problem in problems if problem["question_id"] in problem_ids]
    planned: list[PlannedRequest] = []
    problem_prompts = [(problem, build_prompt(problem["question_content"], problem["starter_code"])) for problem in problems]
    if "planned" in sets:
        planned.append(PlannedRequest("smoke", "smoke", SMOKE_PROMPT, None, group_id="smoke"))
        if thinking_off_check:
            planned.append(PlannedRequest("smoke_off", "smoke_off", SMOKE_PROMPT, None, THINKING_OFF_EXTRA_BODY))
        if long_hold:
            planned.append(PlannedRequest("long_hold", "long_hold", LONG_HOLD_PROMPT, None))
        for problem, prompt in problem_prompts:
            planned.append(PlannedRequest(problem["question_id"], "problem", prompt, problem, group_id=problem["question_id"]))
    if "repeat" in sets:
        planned.append(PlannedRequest("smoke__r2", "smoke_repeat", SMOKE_PROMPT, None, set_name="repeat", group_id="smoke"))
        for problem, prompt in problem_prompts:
            planned.append(
                PlannedRequest(
                    f"{problem['question_id']}__r2", "problem_repeat", prompt, problem,
                    set_name="repeat", group_id=problem["question_id"],
                )
            )
    if "vendor_t1" in sets:
        # The vendor-recommended thinking-mode sampling (temperature 1.0, top_p 0.95), problems only.
        for problem, prompt in problem_prompts:
            for i in range(t1_repeats):
                suffix = "t1" + "abcdefghijklmnopqrstuvwxyz"[i % 26]
                planned.append(
                    PlannedRequest(
                        f"{problem['question_id']}__{suffix}", "problem_t1", prompt, problem,
                        set_name="vendor_t1", sampling_mode="explicit_t1", group_id=problem["question_id"],
                    )
                )
    if "diversity" in sets:
        for condition, mode in DIVERSITY_CONDITIONS:
            for i in range(1, diversity_k + 1):
                planned.append(
                    PlannedRequest(
                        f"div_{condition}_{i}", "diversity", DIVERSITY_PROMPT, None, THINKING_OFF_EXTRA_BODY,
                        set_name="diversity", sampling_mode=mode, group_id=f"diversity:{condition}",
                        max_tokens=DIVERSITY_MAX_TOKENS,
                    )
                )
    return planned


def build_payload(model: str, request: PlannedRequest, max_tokens: int, stream: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": effective_max_tokens(request, max_tokens),
    }
    sampling = SAMPLING_PARAMS[request.sampling_mode]
    if sampling is not None:
        payload["temperature"], payload["top_p"] = sampling
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if request.extra_body:
        payload["extra_body"] = request.extra_body
    return payload


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def resolve_models(requested: list[str] | None) -> tuple[list[str], list[dict]]:
    """Models that can run, in order, and the ones refused for want of a price."""
    wanted = list(requested) if requested else list(MODEL_ORDER)
    runnable, refused = [], []
    for model in wanted:
        if model in PRICES_THB_PER_M:
            runnable.append(model)
        else:
            refused.append({"model": model, "reason": "no price in PRICES_THB_PER_M; refusing to run without a spend bound"})
    return runnable, refused


def print_plan(models, planned, manifest, max_tokens, guard_limit, out=print, stream: bool = False) -> float:
    total = 0.0
    mode = "streaming (stream_options.include_usage)" if stream else "non-streaming"
    out(f"plan: {len(models)} models x {len(planned)} requests, max_tokens {max_tokens} (diversity {DIVERSITY_MAX_TOKENS}), no seed, {mode}")
    out(f"dataset revision {manifest.get('dataset_revision')}, {len(manifest['problems'])} problems")
    for model in models:
        for request in planned:
            cap = effective_max_tokens(request, max_tokens)
            worst = worst_case_thb(model, estimate_input_tokens(request.prompt), cap)
            total += worst
            extra = " +extra_body enable_thinking=false" if request.extra_body else ""
            out(
                f"  {model!r:22} {request.request_id:18} {request.set_name:9} {request.kind:13} {request.sampling_mode:11} "
                f"cap {cap:6} prompt {len(request.prompt):5} chars  worst case {worst:.4f} THB{extra}"
            )
    out(f"worst case total {total:.4f} THB against --max-thb {guard_limit}")
    return total


# ------------------------------------------------------------------- run


@dataclass
class RunState:
    rows: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    stopped_reason: str | None = None
    # (model, request_id) -> {"content": str, "reasoning": str}, kept for the variance and diversity summaries.
    texts: dict = field(default_factory=dict)


def message_texts(body) -> tuple[str, str]:
    """(content, reasoning) of a response body; empty strings when absent."""
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "", ""
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    reasoning = next((message[n] for n in ("reasoning_content", "reasoning") if isinstance(message.get(n), str)), "")
    return (content if isinstance(content, str) else ""), reasoning


def write_json(path: Path, value, scrub: Scrubber) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scrub(json.dumps(value, indent=2, ensure_ascii=False)), encoding="utf-8")


def run_probe(
    *,
    manifest: dict,
    models: list[str],
    refused: list[dict],
    backend: Backend,
    scrub: Scrubber,
    out_dir: Path,
    max_tokens: int,
    read_timeout_s: float,
    max_total_minutes: float,
    max_thb: float,
    stop_after_first_smoke: bool,
    thinking_off_check: bool,
    long_hold: bool,
    stream: bool = False,
    sets: tuple[str, ...] = ("planned",),
    diversity_k: int = DEFAULT_DIVERSITY_K,
    problem_ids: tuple[str, ...] | None = None,
    t1_repeats: int = DEFAULT_T1_REPEATS,
    save_all_headers: bool = False,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], None] = print,
) -> tuple[RunState, SpendGuard]:
    state = RunState(skipped=list(refused))
    guard = SpendGuard(max_thb)
    planned = plan_requests(
        manifest, thinking_off_check=thinking_off_check, long_hold=long_hold, sets=sets,
        diversity_k=diversity_k, problem_ids=problem_ids, t1_repeats=t1_repeats,
    )
    started = clock()
    raw_dir = out_dir / "raw"

    served = backend.served_models()
    if served is None:
        out("WARNING: GET /models unavailable; served model ids not verified")

    for model in models:
        if served is not None and model not in served:
            state.skipped.append({"model": model, "reason": f"not served for this key (served: {served})"})
            out(f"skip {model!r}: not served")
            continue
        consecutive_failures = 0
        # Diversity needs thinking off to be honoured; the first evidence either way ends the doubt.
        thinking_off_ok, thinking_off_reason = True, ""
        diversity_skip_logged = False
        for request in planned:
            if request.set_name == "diversity" and not thinking_off_ok:
                if not diversity_skip_logged:
                    state.skipped.append({"model": model, "reason": f"diversity dropped: {thinking_off_reason}"})
                    out(f"skip diversity for {model!r}: {thinking_off_reason}")
                    diversity_skip_logged = True
                continue
            cap = effective_max_tokens(request, max_tokens)
            if (clock() - started) / 60.0 > max_total_minutes:
                state.stopped_reason = f"--max-total-minutes {max_total_minutes} exceeded"
                out(f"STOP: {state.stopped_reason}")
                return state, guard
            est_in = estimate_input_tokens(request.prompt)
            if not guard.can_send(model, est_in, cap):
                state.stopped_reason = (
                    f"spend guard: spent {guard.spent_thb:.4f} THB + worst case "
                    f"{worst_case_thb(model, est_in, cap):.4f} THB would pass --max-thb {max_thb}"
                )
                out(f"STOP: {state.stopped_reason}")
                return state, guard

            payload = build_payload(model, request, max_tokens, stream)
            started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result = backend.send(payload, read_timeout_s)
            body = None
            if result.status is not None and result.body_text:
                try:
                    body = json.loads(result.body_text)
                except ValueError:
                    body = None

            usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
            prompt_tokens, completion_tokens = _int_or_none(usage.get("prompt_tokens")), _int_or_none(usage.get("completion_tokens"))
            if result.status == 200 and prompt_tokens is not None and completion_tokens is not None:
                cost = guard.charge_usage(model, prompt_tokens, completion_tokens)
                basis = "usage"
            else:
                cost = guard.charge_worst_case(model, est_in, cap)
                basis = "worst_case"

            row = derive_row(
                model=model,
                request_id=request.request_id,
                kind=request.kind,
                problem=request.problem,
                max_tokens=cap,
                result=result,
                body=body,
                cost_thb=cost,
                cost_basis=basis,
            )
            row["started_utc"] = started_utc
            row["set_name"] = request.set_name
            row["sampling_mode"] = request.sampling_mode
            row["group_id"] = request.group_id
            state.rows.append(row)
            content_text, reasoning_text = message_texts(body)
            state.texts[(model, request.request_id)] = {"content": content_text, "reasoning": reasoning_text}
            record = {
                "request": payload,
                "sampling_mode": request.sampling_mode,
                "http_status": result.status,
                "transport_error": result.error,
                "elapsed_s": result.elapsed_s,
                "started_utc": started_utc,
                "response_headers": row["header_evidence"],
                "stream": {"n_chunks": result.n_chunks, "ttft_s": result.ttft_s, "complete": result.stream_complete}
                if result.stream_lines is not None
                else None,
                "response_body": body if body is not None else result.body_text,
            }
            if save_all_headers:
                record["response_headers_all"] = dict(result.headers)
            write_json(raw_dir / f"{slug(model)}__{slug(request.request_id)}.json", record, scrub)

            if request.kind == "smoke_off":
                thinking_off_ok = (
                    result.status == 200
                    and not (row["reasoning_tokens"] or 0)
                    and row["reasoning_chars"] == 0
                    and row["think_close_in_content"] == 0
                )
                thinking_off_reason = "" if thinking_off_ok else (
                    f"thinking-off check (smoke_off) not honoured or failed: HTTP {result.status}, "
                    f"reasoning_tokens {row['reasoning_tokens']}, reasoning_chars {row['reasoning_chars']}"
                )
            elif request.set_name == "diversity" and result.status == 200 and (
                (row["reasoning_tokens"] or 0) > 0 or row["reasoning_chars"] > 0
            ):
                thinking_off_ok = False
                thinking_off_reason = (
                    f"thinking-off not honoured on {request.request_id}: reasoning_tokens {row['reasoning_tokens']}, "
                    f"reasoning_chars {row['reasoning_chars']}"
                )
            if result.stream_lines is not None:
                sse_path = raw_dir / f"{slug(model)}__{slug(request.request_id)}.sse.txt"
                sse_path.write_text(scrub("\n".join(result.stream_lines)), encoding="utf-8")
            stream_note = (
                f"chunks {result.n_chunks} ttft {result.ttft_s if result.ttft_s is None else round(result.ttft_s, 1)}s "
                f"complete {result.stream_complete}  "
                if result.stream_lines is not None
                else ""
            )
            out(
                f"{model} {request.request_id}: status {result.status} {result.error or ''} "
                f"finish {row['finish_reason']} completion {row['completion_tokens']} "
                f"reasoning {row['reasoning_tokens']} {result.elapsed_s:.1f}s  {stream_note}"
                f"cost {cost:.4f} THB ({basis})  running total {guard.spent_thb:.4f} THB"
            )

            if result.status is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_TRANSPORT_FAILURES:
                    state.skipped.append({"model": model, "reason": "aborted after consecutive transport failures"})
                    out(f"abort {model!r}: {consecutive_failures} consecutive transport failures")
                    break
            else:
                consecutive_failures = 0
            if request.request_id == "smoke" and result.status is not None and 400 <= result.status < 500:
                state.skipped.append({"model": model, "reason": f"smoke request at max_tokens {max_tokens} rejected: HTTP {result.status}"})
                out(f"abort {model!r}: smoke rejected (HTTP {result.status}); see raw file for the server's message")
                break
            if request.request_id == "smoke" and stop_after_first_smoke:
                state.stopped_reason = "--stop-after-first-smoke"
                out(
                    f"STOP (--stop-after-first-smoke): script total {guard.spent_thb:.6f} THB for 1 request. "
                    "Compare with Orion's dashboard, then rerun with --no-stop-after-first-smoke."
                )
                return state, guard
    return state, guard


# --------------------------------------------------------------- summaries


def model_answers(rows: list[dict]) -> dict:
    on = [r for r in rows if r["kind"] in ("smoke", "long_hold", "problem") and r["status"] == 200]
    off = [r for r in rows if r["kind"] == "smoke_off" and r["status"] == 200]
    problems = [r for r in on if r["kind"] == "problem"]
    with_reasoning_tokens = [r for r in on if r["reasoning_tokens"] is not None]
    return {
        "q1_thinks_by_default": any(
            (r["reasoning_tokens"] or 0) > 0 or r["reasoning_chars"] > 0 or r["think_close_in_content"] > 0 for r in on
        )
        if on
        else None,
        "q2_trace_fields": sorted({r["reasoning_field"] for r in on if r["reasoning_field"]}),
        "q2_think_tags_in_content": any(r["think_open_in_content"] or r["think_close_in_content"] for r in on),
        "q2_reasoning_tokens_reported": bool(with_reasoning_tokens),
        "q2_reasoning_within_completion_tokens": all(
            r["completion_tokens"] is not None and r["reasoning_tokens"] <= r["completion_tokens"]
            for r in with_reasoning_tokens
        )
        if with_reasoning_tokens
        else None,
        "q2_capped_with_empty_content": sum(1 for r in on if r["cap_hit"] and r["content_empty"]),
        "q3_requests": len(on),
        "q3_cap_hits": sum(1 for r in on if r["cap_hit"]),
        "q3_completion_tokens_min_max": [
            min((r["completion_tokens"] for r in on if r["completion_tokens"] is not None), default=None),
            max((r["completion_tokens"] for r in on if r["completion_tokens"] is not None), default=None),
        ],
        "q3_elapsed_s_min_max": [min((r["elapsed_s"] for r in on), default=None), max((r["elapsed_s"] for r in on), default=None)],
        "q3_finish_reasons": sorted({str(r["finish_reason"]) for r in on}),
        "q4_problems": len(problems),
        "q4_extracted_nonempty": sum(1 for r in problems if r["extracted_nonempty"]),
        "q4_extracted_parses": sum(1 for r in problems if r["extracted_parses"]),
        "q4_odd_fence_count": sum(1 for r in problems if r["odd_fence_count"]),
        "thinking_off_check": [
            {"reasoning_tokens": r["reasoning_tokens"], "reasoning_chars": r["reasoning_chars"], "finish_reason": r["finish_reason"]}
            for r in off
        ],
    }


def common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def pair_stats(row_a: dict, row_b: dict, texts_a: dict, texts_b: dict) -> dict:
    """Descriptive comparison of two replies to the same request; carries no verdict."""
    ok = row_a["status"] == 200 and row_b["status"] == 200
    stats: dict = {"comparable": ok, "request_ids": [row_a["request_id"], row_b["request_id"]]}
    if not ok:
        stats["statuses"] = [row_a["status"], row_b["status"]]
        return stats
    full_a = texts_a["reasoning"] + "\n<<<CONTENT>>>\n" + texts_a["content"]
    full_b = texts_b["reasoning"] + "\n<<<CONTENT>>>\n" + texts_b["content"]
    tokens = [row_a["completion_tokens"], row_b["completion_tokens"]]
    known = all(t is not None for t in tokens)
    stats.update(
        exact_match_content=texts_a["content"] == texts_b["content"],
        exact_match_full=full_a == full_b,
        common_prefix_chars_full=common_prefix_len(full_a, full_b),
        common_prefix_chars_content=common_prefix_len(texts_a["content"], texts_b["content"]),
        len_chars_full=[len(full_a), len(full_b)],
        completion_tokens=tokens,
        completion_tokens_abs_diff=abs(tokens[0] - tokens[1]) if known else None,
        completion_tokens_rel_diff=round(abs(tokens[0] - tokens[1]) / max(tokens), 4) if known and max(tokens) else None,
        reasoning_tokens=[row_a["reasoning_tokens"], row_b["reasoning_tokens"]],
        finish_reasons=[row_a["finish_reason"], row_b["finish_reason"]],
        extracted_code_identical=lcb_extract_code(texts_a["content"]) == lcb_extract_code(texts_b["content"]),
    )
    return stats


def build_variance(state: RunState) -> list[dict]:
    """Planned-vs-repeat pairs: request X against X__r2, per model."""
    index = {(r["model"], r["request_id"]): r for r in state.rows}
    pairs = []
    for (model, request_id), row in index.items():
        if row["set_name"] != "planned" or row["kind"] not in ("smoke", "problem"):
            continue
        other = index.get((model, f"{request_id}__r2"))
        if other is None:
            continue
        stats = pair_stats(row, other, state.texts[(model, request_id)], state.texts[(model, f"{request_id}__r2")])
        pairs.append({"model": model, "group_id": row["group_id"], "kind": row["kind"], **stats})
    return pairs


def normalise_output(text: str) -> str:
    return " ".join(text.split()).casefold()


def first_sentence(text: str) -> str:
    return normalise_output(re.split(r"(?<=[.!?])\s|\n", text.strip(), maxsplit=1)[0])


def classify_default(n_default: int, n_t0: int, n_t1: int, ok_counts: list[int]) -> str:
    """The pre-registered rule from INTERPRETATION_RULE, applied to distinct-output counts."""
    if min(ok_counts) < 3:
        return "insufficient_data"
    if n_t1 < n_t0 + 2:
        return "not_informative"
    if n_default <= n_t0 + 1:
        return "greedy_like"
    if n_default >= n_t1 - 1:
        return "sampled_like"
    return "inconclusive"


def build_diversity(state: RunState) -> dict:
    out: dict = {}
    for row in state.rows:
        if row["set_name"] != "diversity":
            continue
        model = row["model"]
        condition = row["group_id"].split(":", 1)[1]
        entry = out.setdefault(model, {}).setdefault(condition, {"sends": 0, "ok": 0, "outputs": []})
        entry["sends"] += 1
        if row["status"] == 200:
            entry["ok"] += 1
            entry["outputs"].append(state.texts[(model, row["request_id"])]["content"])
    for model, conditions in out.items():
        for entry in conditions.values():
            entry["distinct_outputs"] = len({normalise_output(t) for t in entry["outputs"]})
            entry["distinct_first_sentence"] = len({first_sentence(t) for t in entry["outputs"]})
            entry["outputs"] = [t[:200] for t in entry["outputs"]]
        needed = ("omitted", "t0", "t1")
        if all(name in conditions for name in needed):
            conditions["assessment"] = classify_default(
                conditions["omitted"]["distinct_outputs"],
                conditions["t0"]["distinct_outputs"],
                conditions["t1"]["distinct_outputs"],
                [conditions[name]["ok"] for name in needed],
            )
        else:
            conditions["assessment"] = "insufficient_data"
    return out


def write_summary(out_dir: Path, state: RunState, guard: SpendGuard, scrub: Scrubber) -> None:
    by_model: dict[str, list[dict]] = {}
    for row in state.rows:
        by_model.setdefault(row["model"], []).append(row)
    variance = build_variance(state)
    diversity = build_diversity(state)
    write_json(
        out_dir / "summary.json",
        {
            "spent_thb": round(guard.spent_thb, 6),
            "max_thb": guard.limit_thb,
            "stopped_reason": state.stopped_reason,
            "skipped": state.skipped,
            "interpretation_rule": INTERPRETATION_RULE,
            "answers": {model: model_answers(rows) for model, rows in by_model.items()},
            "rows": state.rows,
        },
        scrub,
    )
    if variance:
        write_json(out_dir / "variance.json", variance, scrub)
        variance_columns = [
            "model", "group_id", "kind", "comparable", "exact_match_content", "exact_match_full",
            "common_prefix_chars_full", "common_prefix_chars_content", "completion_tokens_abs_diff",
            "completion_tokens_rel_diff", "extracted_code_identical",
        ]
        with open(out_dir / "variance.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=variance_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(variance)
    if diversity:
        write_json(out_dir / "diversity.json", {"rule": INTERPRETATION_RULE, "models": diversity}, scrub)
    columns = [
        "model", "request_id", "kind", "problem_format", "difficulty", "status", "finish_reason", "elapsed_s",
        "prompt_tokens", "completion_tokens", "reasoning_tokens", "cap_hit", "content_empty", "reasoning_field",
        "reasoning_chars", "think_close_in_content", "n_fence_lines", "extracted_nonempty", "extracted_parses",
        "cost_thb", "cost_basis",
    ]
    path = out_dir / "summary.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state.rows)


# ------------------------------------------------------------------ prepare


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_problems(rows: list[dict], *, seed: int, max_chars: int, window_start: str, window_end: str) -> dict:
    """Pure selection over already-parsed rows (public fields only are read)."""
    max_date = max((r["contest_date"][:10] for r in rows), default=None)
    in_window = [r for r in rows if window_start <= r["contest_date"][:10] <= window_end]
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in in_window:
        cells.setdefault((problem_format(r["starter_code"]), r["difficulty"]), []).append(r)
    counts = {f"{fmt}/{difficulty}": len(v) for (fmt, difficulty), v in sorted(cells.items())}

    rng = random.Random(seed)
    chosen: list[dict] = []
    notes: list[str] = []
    taken: set[str] = set()
    fallback = list(FALLBACK_CELLS)
    for cell in TARGET_CELLS:
        pool = sorted((r for r in cells.get(cell, []) if len(r["question_content"]) <= max_chars and r["question_id"] not in taken), key=lambda r: r["question_id"])
        used = cell
        while not pool and fallback:
            used = fallback.pop(0)
            pool = sorted((r for r in cells.get(used, []) if len(r["question_content"]) <= max_chars and r["question_id"] not in taken), key=lambda r: r["question_id"])
        if not pool:
            notes.append(f"cell {cell[0]}/{cell[1]} empty and no fallback available")
            continue
        if used != cell:
            notes.append(f"cell {cell[0]}/{cell[1]} empty; substituted {used[0]}/{used[1]}")
        pick = rng.choice(pool)
        taken.add(pick["question_id"])
        chosen.append(
            {
                "question_id": pick["question_id"],
                "question_title": pick["question_title"],
                "platform": pick["platform"],
                "contest_date": pick["contest_date"],
                "difficulty": pick["difficulty"],
                "problem_format": problem_format(pick["starter_code"]),
                "question_content": pick["question_content"],
                "starter_code": pick["starter_code"],
                "public_test_cases": pick["public_test_cases"],
            }
        )
    return {"max_contest_date": max_date, "n_rows": len(rows), "n_in_window": len(in_window), "window_counts": counts, "problems": chosen, "notes": notes}


def read_public_rows(path: Path) -> list[dict]:
    keep = ("question_id", "question_title", "question_content", "platform", "contest_date", "starter_code", "difficulty", "public_test_cases")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                full = json.loads(line)
                rows.append({k: full[k] for k in keep})
    return rows


def cmd_prepare(args) -> int:
    path = Path(args.test6) if args.test6 else None
    if path is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=DATASET_FILE,
                revision=args.revision,
                local_dir=str(DEFAULT_TEST6.parent),
            )
        )
    rows = read_public_rows(path)
    selected = select_problems(rows, seed=args.seed, max_chars=args.max_chars, window_start=args.window_start, window_end=args.window_end)
    print(f"rows: {selected['n_rows']}  max contest_date: {selected['max_contest_date']}")
    print(f"problems in window {args.window_start}..{args.window_end}: {selected['n_in_window']}")
    for cell, count in selected["window_counts"].items():
        print(f"  {cell}: {count}")
    for note in selected["notes"]:
        print(f"NOTE: {note}")
    for problem in selected["problems"]:
        print(f"selected {problem['question_id']} {problem['platform']} {problem['contest_date'][:10]} {problem['difficulty']} {problem['problem_format']}")
    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_file": DATASET_FILE,
        "dataset_revision": args.revision,
        "dataset_file_sha256": file_sha256(path),
        "seed": args.seed,
        "max_chars": args.max_chars,
        "window": [args.window_start, args.window_end],
        **{k: selected[k] for k in ("n_rows", "max_contest_date", "n_in_window", "window_counts", "notes", "problems")},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------- commands


class Tee:
    def __init__(self, path: Path, scrub: Scrubber):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.scrub = scrub

    def __call__(self, text: str) -> None:
        text = self.scrub(text)
        print(text)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def model_info_urls() -> tuple[str, str]:
    base_url = api_model.read_base_url()
    return base_url.removesuffix("/v1") + "/model/info", base_url + "/model/info"


def fetch_model_info(get: Callable[[str], tuple[int | None, str]]) -> dict:
    """Read-only lookups of the gateway's model info; whatever comes back is recorded, truncated."""
    found = {}
    for url in model_info_urls():
        status, body = get(url)
        found[url] = {
            "status": status,
            "truncated": len(body) > MODEL_INFO_MAX_CHARS,
            "body": body[:MODEL_INFO_MAX_CHARS],
        }
    return found


def cmd_run(args, backend_factory=make_http_backend, key_reader=api_model._read_api_key) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    models, refused = resolve_models(args.models)
    for item in refused:
        print(f"REFUSED {item['model']!r}: {item['reason']}", file=sys.stderr)
    if not models:
        print("no runnable model", file=sys.stderr)
        return 2
    sets = tuple(part.strip() for part in args.sets.split(",") if part.strip())
    problem_ids = (
        tuple(part.strip() for part in args.problems.split(",") if part.strip()) if args.problems else None
    )
    try:
        planned = plan_requests(
            manifest, thinking_off_check=args.thinking_off_check, long_hold=args.long_hold,
            sets=sets, diversity_k=args.diversity_k, problem_ids=problem_ids, t1_repeats=args.t1_repeats,
        )
    except ValueError as e:
        print(f"--sets/--diversity-k/--problems/--t1-repeats: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_plan(models, planned, manifest, args.max_tokens, args.max_thb, stream=args.stream)
        if args.model_info:
            print(f"model info: would GET /model/info under the {api_model.BASE_URL_ENV} address, with and without /v1 (not performed in a dry run)")
        print("dry run: no key read, no network call, nothing written")
        return 0

    key = key_reader()
    scrub = Scrubber(key)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"probe_thinking_{stamp}.log"
    out = Tee(log_path, scrub)
    backend, close = backend_factory(key, args.read_timeout)
    try:
        worst = print_plan(models, planned, manifest, args.max_tokens, args.max_thb, out=out, stream=args.stream)
        with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as f:
            for request in planned:
                f.write(
                    scrub(
                        json.dumps(
                            {
                                "request_id": request.request_id, "kind": request.kind, "set": request.set_name,
                                "sampling_mode": request.sampling_mode, "group_id": request.group_id,
                                "max_tokens": effective_max_tokens(request, args.max_tokens),
                                "prompt": request.prompt, "extra_body": request.extra_body,
                            },
                            ensure_ascii=False,
                        )
                    )
                    + "\n"
                )
        if args.model_info and backend.get is not None:
            write_json(out_dir / "model_info.json", fetch_model_info(backend.get), scrub)
            out(f"model info lookups written to {out_dir / 'model_info.json'}")
        state, guard = RunState(), SpendGuard(args.max_thb)
        try:
            state, guard = run_probe(
                manifest=manifest,
                models=models,
                refused=refused,
                backend=backend,
                scrub=scrub,
                out_dir=out_dir,
                max_tokens=args.max_tokens,
                read_timeout_s=args.read_timeout,
                max_total_minutes=args.max_total_minutes,
                max_thb=args.max_thb,
                stop_after_first_smoke=args.stop_after_first_smoke,
                thinking_off_check=args.thinking_off_check,
                long_hold=args.long_hold,
                stream=args.stream,
                sets=sets,
                diversity_k=args.diversity_k,
                problem_ids=problem_ids,
                t1_repeats=args.t1_repeats,
                save_all_headers=args.save_all_headers,
                out=out,
            )
        finally:
            write_summary(out_dir, state, guard, scrub)
            write_json(
                out_dir / "run_meta.json",
                {
                    "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "spent_thb": round(guard.spent_thb, 6),
                    "max_thb": args.max_thb,
                    "worst_case_plan_thb": round(worst, 6),
                    "stopped_reason": state.stopped_reason,
                    "skipped": state.skipped,
                    "models_requested": models,
                    "args": {k: v for k, v in vars(args).items() if k not in ("func",)},
                    "dataset_revision": manifest.get("dataset_revision"),
                    "manifest_problems": [p["question_id"] for p in manifest["problems"]],
                    "prices_thb_per_m": PRICES_THB_PER_M,
                    "script_sha256": file_sha256(Path(__file__)),
                    "suite_git": run_config.git_state(REPO_ROOT),
                    "base_url": api_model.BASE_URL_RECORDED,
                },
                scrub,
            )
        out(f"spent (computed) {guard.spent_thb:.6f} THB of {args.max_thb}; {len(state.rows)} requests; outputs {out_dir}")
        out(
            f"suggested runs/_index.md row: | {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]} | "
            f"tools/probe_thinking.py | run {len(models)} models, max_tokens {args.max_tokens} | "
            f"{len(state.rows)} requests, {guard.spent_thb:.4f} THB computed, stopped: {state.stopped_reason or 'completed'} | "
            f"probe_thinking_{stamp}.log | runs/results/thinking_probe/{stamp}/ |"
        )
    finally:
        out.close()
        close()
    return 0


def cmd_reconcile(args) -> int:
    meta = json.loads((Path(args.run_dir) / "run_meta.json").read_text(encoding="utf-8"))
    computed = meta["spent_thb"]
    print(f"computed {computed:.6f} THB, dashboard {args.dashboard_thb:.6f} THB")
    if computed > 0:
        print(f"dashboard / computed = {args.dashboard_thb / computed:.3f}")
    print(f"difference (dashboard - computed) = {args.dashboard_thb - computed:+.6f} THB")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="select problems from a local test6.jsonl and write a manifest")
    prep.add_argument("--test6", default=None, help=f"local path to {DATASET_FILE}; omitted -> download at --revision")
    prep.add_argument("--revision", default=DATASET_REVISION)
    prep.add_argument("--out", default=str(DEFAULT_MANIFEST))
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--max-chars", type=int, default=4000)
    prep.add_argument("--window-start", default=WINDOW_START)
    prep.add_argument("--window-end", default=WINDOW_END)
    prep.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", help="send the probe requests")
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    run.add_argument("--models", nargs="+", default=None, help=f"default: {', '.join(MODEL_ORDER)}")
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S)
    run.add_argument("--max-total-minutes", type=float, default=DEFAULT_MAX_TOTAL_MINUTES)
    run.add_argument("--max-thb", type=float, default=DEFAULT_MAX_THB)
    run.add_argument("--stop-after-first-smoke", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--thinking-off-check", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--long-hold", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--stream", action="store_true", help="send stream=true with include_usage and reassemble the chunks")
    run.add_argument("--sets", default="planned", help=f"comma list of request sets: {', '.join(SETS)}")
    run.add_argument("--diversity-k", type=int, default=DEFAULT_DIVERSITY_K, help="sends per diversity condition")
    run.add_argument("--problems", default=None, help="comma list of manifest question ids; restricts planned, repeat and vendor_t1")
    run.add_argument("--t1-repeats", type=int, default=DEFAULT_T1_REPEATS, help="sends per problem in the vendor_t1 set")
    run.add_argument("--save-all-headers", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--model-info", action=argparse.BooleanOptionalAction, default=True,
                     help="two read-only GETs to /model/info and /v1/model/info on the API host")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    run.set_defaults(func=cmd_run)

    rec = sub.add_parser("reconcile", help="compare a run's computed spend with the dashboard")
    rec.add_argument("--run-dir", required=True)
    rec.add_argument("--dashboard-thb", type=float, required=True)
    rec.set_defaults(func=cmd_reconcile)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
