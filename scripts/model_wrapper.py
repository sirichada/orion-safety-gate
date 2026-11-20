#!/usr/bin/env python3
"""
Generic model-calling wrapper. Implements a fixed generation interface every
benchmark adapter/scorer calls against, backed by any HF `transformers`
AutoModelForCausalLM-compatible checkpoint, with an optional PEFT adapter
attached on top (covers a base-model + LoRA-adapter shape, e.g. BackdoorLLM's
released adapters).

Usage:
    python3 model_wrapper.py --model meta-llama/Llama-2-7b-chat-hf \
        --adapter path/to/backdoorllm/adapter --prompts prompts.jsonl

Requires: torch, transformers, peft (if --adapter given), bitsandbytes
(if --precision 4bit, the default).

Before generating, writes <out>.run_config.json: the values this process actually
ran with, what the chat template wraps around a prompt, and where the weights came
from (see run_config.py).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import run_config
from hf_model import load_causal_lm, unload

# Exit code for a run stopped by --max-consecutive-failures. Always paired with a
# <out>.stopped.json sidecar, so a bare 3 from anywhere else isn't mistaken for it.
STOPPED_NO_ANSWER_EXIT_CODE = 3

# Exit code for a run that could not start because the requested precision or revision
# cannot be applied. Always paired with a <out>.error.json sidecar holding the message.
CONFIG_ERROR_EXIT_CODE = 4

# Exit code for a run stopped because the next request would pass --max-thb. Always paired with
# a <out>.budget.json sidecar; the responses written so far are kept and --resume continues them.
BUDGET_EXIT_CODE = 5

# Exit code for an api run stopped by an unexpected error in generate(). Always paired with an
# <out>.aborted.json sidecar; the batches already flushed are kept and --resume continues them.
ABORTED_EXIT_CODE = 6


class StoppedNoAnswer(RuntimeError):
    """Raised when consecutive batches produce no answer for any prompt."""


# Re-exported for backwards compatibility -- callers/scripts that imported
# load_model from this module directly still work.
load_model = load_causal_lm


def make_generate(model, tokenizer, max_input_tokens: int = 2048, apply_chat_template: bool = True, template_kwargs: dict | None = None):
    """Returns generate(prompts, max_new_tokens, temperature, top_p, seed=None,
    chat_template=None) -> List[str].

    max_input_tokens is an explicit cap -- tokenizer.model_max_length defaults to
    a huge sentinel on some tokenizers, silently disabling truncation otherwise.

    apply_chat_template (set at construction) toggles templating at all; the
    call-time chat_template overrides which template is used instead of the
    tokenizer's own default.
    """
    import torch

    def render(prompts: list[str], chat_template: str | None) -> list[str]:
        if apply_chat_template and (chat_template or tokenizer.chat_template):
            template = chat_template if chat_template is not None else tokenizer.chat_template
            return [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    chat_template=template,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(template_kwargs or {}),
                )
                for p in prompts
            ]
        return prompts

    def count_truncated(prompts: list[str], chat_template: str | None = None) -> int:
        """Prompts whose tokenized length exceeds max_input_tokens, i.e. that generate() cuts short."""
        encoded = tokenizer(render(prompts, chat_template), truncation=False)["input_ids"]
        return sum(1 for ids in encoded if len(ids) > max_input_tokens)

    def generate(
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None = None,
        chat_template: str | None = None,
    ) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)

        prompts = render(prompts, chat_template)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        completions = output_ids[:, input_len:]
        return tokenizer.batch_decode(completions, skip_special_tokens=True)

    generate.count_truncated = count_truncated
    return generate


def template_kwargs(args) -> dict:
    """Extra chat-template variables for the hf backend: --no-thinking sets enable_thinking to false."""
    return {"enable_thinking": False} if getattr(args, "no_thinking", False) else {}


def build_report(args, model, tokenizer, n_truncated_inputs) -> dict:
    """The run_config.json content: effective values, template record and provenance."""
    api = args.backend == "api"
    args_used = {
        "backend": args.backend,
        "model": args.model,
        "adapter": args.adapter,
        "batch_size": args.batch_size,
        "max_consecutive_failures": args.max_consecutive_failures,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "model_revision": args.model_revision,
        "adapter_revision": args.adapter_revision,
    }
    if api:
        import api_model

        args_used.update(
            api_concurrency=args.api_concurrency,
            no_answer_retries=args.no_answer_retries,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            stream=True,
        )
        server_side = run_config.unavailable("server-side chat template")
        return run_config.build_stage_report(
            args=args_used,
            precision=run_config.unavailable("api backend: weights are server-side"),
            model_input={"chat_template": server_side},
            per_model={
                "max_new_tokens": args.max_new_tokens,
                "max_input_tokens": run_config.unavailable("server-side limit"),
                "thinking": {"template_parameters": server_side, "passed": {}},
            },
            provenance={
                "model_revision": run_config.unavailable("api backend"),
                "adapter_revision": None,
                "model_files": None,
                "adapter_files": None,
                "libraries": run_config.library_versions(("httpx",)),
                "gpu": run_config.unavailable("api backend"),
                "api": {
                    "base_url": api_model.BASE_URL,
                    "reported_model": run_config.unavailable("deferred"),
                    "system_fingerprint": run_config.unavailable("deferred"),
                },
            },
            n_truncated_inputs=run_config.unavailable("server-side limit"),
        )

    args_used.update(precision=args.precision, max_input_tokens=args.max_input_tokens)
    if args.no_thinking:
        args_used["no_thinking"] = True
    template = args.chat_template if args.chat_template is not None else tokenizer.chat_template
    source = "override" if args.chat_template is not None else "tokenizer"

    def identify(path: str, revision: str | None):
        if Path(path).exists():
            return run_config.unavailable("local path"), run_config.local_checkpoint_identity(path)
        resolved = revision or run_config.HUB.cached_commit(path)
        return (resolved or run_config.unavailable("revision not resolved")), None

    model_revision, model_files = identify(args.model, args.model_revision)
    adapter_revision, adapter_files = (None, None)
    if args.adapter:
        adapter_revision, adapter_files = identify(args.adapter, args.adapter_revision)

    return run_config.build_stage_report(
        args=args_used,
        precision=run_config.describe_precision(args.precision, model),
        model_input={
            "chat_template": run_config.render_template_record(
                tokenizer, template, not args.no_chat_template, source, template_kwargs(args)
            )
        },
        per_model={
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "thinking": {"template_parameters": run_config.template_parameters(template), "passed": template_kwargs(args)},
        },
        provenance={
            "model_revision": model_revision,
            "adapter_revision": adapter_revision,
            "model_files": model_files,
            "adapter_files": adapter_files,
            "libraries": run_config.library_versions(),
            "gpu": run_config.gpu_info(),
            "api": None,
        },
        n_truncated_inputs=n_truncated_inputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or HF repo id")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path/repo id")
    parser.add_argument("--prompts", required=True, help="JSONL file with a 'model_input' field per line")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="api backend: top_k, sent inside the request's extra_body object (the gateway rejects it as a top-level field)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="api backend: presence_penalty in [-2, 2], sent as a top-level request field",
    )
    parser.add_argument(
        "--precision",
        choices=run_config.PRECISIONS,
        default=run_config.DEFAULTS["precision"],
        help="4bit = bitsandbytes NF4; full = bfloat16; as_published = the checkpoint's own dtype/quantization "
        "(required for pre-quantized checkpoints)",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Alias for --precision full")
    parser.add_argument("--model-revision", default=None, help="Hub commit to load the model and tokenizer at")
    parser.add_argument("--adapter-revision", default=None, help="Hub commit to load the adapter at")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort after this many batches in a row that produced no answer for any prompt, even after per-prompt retry",
    )
    parser.add_argument("--no-chat-template", action="store_true", help="Skip the tokenizer's chat template, send raw prompts")
    parser.add_argument("--batch-size", type=int, default=run_config.DEFAULTS["batch_size"], help="Prompts per generate() call -- the full HarmBench/ThaiSafetyBench prompt sets (400/1,889) would OOM a 12GB GPU in one call")
    parser.add_argument("--no-thinking", action="store_true", help="hf only: render the chat template with enable_thinking=false")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling")
    parser.add_argument(
        "--chat-template", default=None, help="Override chat template string (defaults to the tokenizer's own)"
    )
    parser.add_argument("--out", default=None, help="Optional path to write responses JSONL to")
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=run_config.DEFAULTS["api_concurrency"],
        help=f"api backend: requests in flight at once, 1 to {run_config.API_CONCURRENCY_CEILING}",
    )
    parser.add_argument(
        "--no-answer-retries",
        type=int,
        default=run_config.DEFAULTS["no_answer_retries"],
        help="api backend: re-requests of a reply that came back empty at the token cap",
    )
    parser.add_argument("--max-thb", type=float, default=None, help="api backend: spend limit in THB for this run (required)")
    parser.add_argument("--spend-ledger", default=None, help="api backend: JSON file holding the run's spend so far")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the responses already in --out (same settings only) instead of starting over",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "api"],
        default="hf",
        help="hf = local transformers weights; api = OpenAI-compatible HTTP endpoint (--model is then the API model id)",
    )
    args = parser.parse_args()
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be >= 1")
    if args.no_4bit:
        if args.precision not in (run_config.DEFAULTS["precision"], "full"):
            parser.error("--no-4bit conflicts with --precision")
        args.precision = "full"
    if args.adapter_revision and not args.adapter:
        parser.error("--adapter-revision needs --adapter")
    if args.resume and not args.out:
        parser.error("--resume needs --out")
    if args.backend == "api":
        if not 1 <= args.api_concurrency <= run_config.API_CONCURRENCY_CEILING:
            parser.error(f"--api-concurrency must be between 1 and {run_config.API_CONCURRENCY_CEILING}")
        if args.no_answer_retries < 0:
            parser.error("--no-answer-retries must be >= 0")
        if args.max_thb is None or not args.max_thb > 0:
            parser.error("--max-thb is required for --backend api and must be positive")
    elif args.max_thb is not None or args.spend_ledger is not None:
        parser.error("--max-thb/--spend-ledger apply only to the api backend")
    if args.top_k is not None:
        if args.backend != "api":
            parser.error("--top-k applies only to the api backend")
        if args.top_k < 1:
            parser.error("--top-k must be >= 1")
    if args.presence_penalty is not None:
        if args.backend != "api":
            parser.error("--presence-penalty applies only to the api backend")
        if not -2 <= args.presence_penalty <= 2:
            parser.error("--presence-penalty must be in [-2, 2]")

    if args.backend == "api":
        if (
            args.adapter
            or args.precision != run_config.DEFAULTS["precision"]
            or args.model_revision
            or args.adapter_revision
            or args.no_chat_template
            or args.no_thinking
            or args.chat_template is not None
            or args.max_input_tokens != run_config.DEFAULTS["max_input_tokens"]
        ):
            parser.error(
                "--adapter/--precision/--no-4bit/--model-revision/--adapter-revision/--no-chat-template/"
                "--chat-template/--max-input-tokens/--no-thinking are hf-only"
            )
        from api_model import SpendGuard, make_api_generate

        model = tokenizer = None
        try:
            spend = SpendGuard(args.model, args.max_thb, args.spend_ledger)
        except ValueError as e:
            parser.error(str(e))
        extra_params = {}
        if args.top_k is not None:
            extra_params["extra_body"] = {"top_k": args.top_k}
        if args.presence_penalty is not None:
            extra_params["presence_penalty"] = args.presence_penalty
        sampling_extras = {"extra_params": extra_params} if extra_params else {}
        generate = make_api_generate(
            args.model,
            concurrency=args.api_concurrency,
            no_answer_retries=args.no_answer_retries,
            spend=spend,
            **sampling_extras,
        )
    else:
        try:
            model, tokenizer = load_model(
                args.model,
                args.adapter,
                precision=args.precision,
                revision=args.model_revision,
                adapter_revision=args.adapter_revision,
            )
        except (run_config.PrecisionError, run_config.RevisionError) as e:
            print(f"Cannot load the model: {e}", file=sys.stderr)
            if args.out:
                error = {"kind": type(e).__name__, "message": str(e)}
                Path(args.out + ".error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
            sys.exit(CONFIG_ERROR_EXIT_CODE)
        generate = make_generate(
            model,
            tokenizer,
            max_input_tokens=args.max_input_tokens,
            apply_chat_template=not args.no_chat_template,
            **({"template_kwargs": template_kwargs(args)} if args.no_thinking else {}),
        )

    records = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line]
    prompts = [r["model_input"] for r in records]
    api = args.backend == "api"

    def config_error(kind: str, message: str) -> None:
        print(f"Cannot start: {message}", file=sys.stderr)
        if args.out:
            error = {"kind": kind, "message": message}
            Path(args.out + ".error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        sys.exit(CONFIG_ERROR_EXIT_CODE)

    n_truncated = (
        run_config.unavailable("server-side limit") if api else generate.count_truncated(prompts, args.chat_template)
    )
    report = build_report(args, model, tokenizer, n_truncated)
    reasoning_path = Path(args.out + ".reasoning.jsonl") if args.out and api else None

    # A resumed run continues the records already on disk, and only under the settings that wrote them.
    start = 0
    if args.resume and Path(args.out).exists():
        try:
            previous = run_config.read_stage_report(args.out)
        except ValueError as e:
            config_error("ResumeMismatch", f"cannot resume: {e}")
        mismatched = run_config.tested_mismatches(previous, report)
        if mismatched:
            config_error(
                "ResumeMismatch",
                "cannot resume: settings differ from the run that wrote the file: " + ", ".join(mismatched),
            )
        kept = []
        for line in Path(args.out).read_text(encoding="utf-8").splitlines():
            try:
                kept.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a torn final line from an interrupted write
        if len(kept) > len(records) or any(k.get("id") != r.get("id") for k, r in zip(kept, records)):
            config_error("ResumeMismatch", "cannot resume: the responses on disk are not a prefix of these prompts")
        start = len(kept)
        records[:start] = kept
        with open(args.out, "w", encoding="utf-8") as f:
            for record in kept:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if reasoning_path is not None and reasoning_path.exists():
            lines = [ln for ln in reasoning_path.read_text(encoding="utf-8").splitlines() if ln]
            reasoning_path.write_text(
                "".join(ln + "\n" for ln in lines if json.loads(ln)["index"] < start), encoding="utf-8"
            )
        print(f"Resuming: {start}/{len(records)} responses already written")
    elif args.out:
        Path(args.out).write_text("", encoding="utf-8")
        if reasoning_path is not None:
            reasoning_path.write_text("", encoding="utf-8")

    # Written before generating, so a stopped or killed run still states what it was run with.
    # A .stopped.json that exists therefore always describes the latest attempt.
    if args.out:
        Path(args.out + ".stopped.json").unlink(missing_ok=True)
        run_config.write_stage_report(args.out, report)

    def write_out(records_to_write: list[dict]) -> None:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for record in records_to_write:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Wrote {len(records_to_write)} responses -> {args.out}")
        else:
            for record in records_to_write[:5]:
                print(json.dumps(record, ensure_ascii=False))

    def clear_cuda_cache() -> None:
        if api:
            return
        # Unguarded empty_cache() raises on CPU-only torch builds.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Populated only by the api backend; stays aligned with `responses`. Resumed records carry
    # their own fields, so their entries are placeholders and are never re-filled.
    metas: list[dict] = [{}] * start
    finish_reasons: list[str | None] = [r.get("finish_reason") for r in records[:start]] if api else []
    attempts: list[int | None] = [r.get("attempts") for r in records[:start]] if api else []
    responses: list[str | None] = [
        None if r.get("generation_failed") and not r.get("no_answer") else r.get("response", "")
        for r in records[:start]
    ]
    failed = sum(1 for r in responses if r is None)
    flushed = start

    def collect_api_metadata(batch_metas: list[dict]) -> None:
        if api:
            metas.extend(batch_metas)
            finish_reasons.extend(m.get("finish_reason") for m in batch_metas)
            attempts.extend(m.get("attempts") for m in batch_metas)

    def failed_meta() -> dict:
        return {"finish_reason": None, "attempts": None, "failed": True}

    def is_no_answer(response: str | None) -> bool:
        return response is not None and not response.strip()

    def answered(k: int) -> bool:
        if responses[k] is None:
            return False
        return not api or not is_no_answer(responses[k])

    def fill_record(k: int, record: dict, response: str | None) -> None:
        record["response"] = response if response is not None else ""
        record["generation_failed"] = response is None
        if api:
            meta = metas[k]
            finish = finish_reasons[k]
            # A reply with empty content never answered, whatever its finish_reason (cut off at the cap, or
            # stopped with the text only in the reasoning channel): it must not be scored as a refusal.
            # generation_failed is the flag every scorer already excludes; no_answer keeps the cause distinguishable.
            no_answer = is_no_answer(response)
            record["finish_reason"] = finish
            record["attempts"] = attempts[k]
            record["no_answer"] = no_answer
            record["cap_used"] = args.max_new_tokens
            record["usage"] = {
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
            }
            record["latency_s"] = meta.get("latency_s")
            record["cost_thb"] = meta.get("cost_thb")
            if meta.get("failed"):
                record["generation_error"] = meta.get("error")
            if no_answer:
                record["generation_failed"] = True

    def fill_new(upto: int) -> None:
        for k in range(start, upto):
            fill_record(k, records[k], responses[k])

    def flush() -> None:
        """Appends the records finished since the last flush, so an interrupted run keeps them."""
        nonlocal flushed
        if not args.out or flushed >= len(responses):
            return
        fill_new(len(responses))
        if reasoning_path is not None:
            with open(reasoning_path, "a", encoding="utf-8") as f:
                for k in range(flushed, len(responses)):
                    trace = metas[k].get("reasoning_content")
                    if trace:
                        entry = {"index": k, "id": records[k].get("id"), "reasoning_content": trace}
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(args.out, "a", encoding="utf-8") as f:
            for k in range(flushed, len(responses)):
                f.write(json.dumps(records[k], ensure_ascii=False) + "\n")
        flushed = len(responses)

    # A failed generation is None, not "" -- an empty string is a real, scoreable
    # output (e.g. immediate EOS), not an infra failure.
    def generate_one_prompt(prompt: str, record_id) -> str | None:
        try:
            return generate(
                [prompt], args.max_new_tokens, args.temperature, args.top_p, args.seed, args.chat_template
            )[0]
        except run_config.BudgetExceeded:
            raise
        except Exception as e:
            print(f"WARNING: generation failed for record {record_id!r}: {e}")
            return None

    # Only unbroken runs of fully-failed batches count; scattered failures never abort.
    consecutive_failed_batches = 0
    try:
        for i in range(start, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            # Only generate() sits inside the try: anything that raised after responses grew would
            # re-append the batch on retry and misalign every later record.
            try:
                batch_responses = generate(
                    batch, args.max_new_tokens, args.temperature, args.top_p, args.seed, args.chat_template
                )
            except run_config.BudgetExceeded:
                raise
            except Exception as e:
                if api:
                    # Every reply in this batch may already have been paid for; re-sending
                    # them one by one would pay again. Stop and let --resume continue.
                    raise run_config.GenerationAborted(f"{type(e).__name__}: {e}") from e
                # Likely OOM -- clear cache and retry per-prompt so only the
                # actually oversized prompt(s) get marked failed.
                print(f"WARNING: batch {i}-{i + len(batch)} failed ({e}); clearing cache and retrying per-prompt")
                clear_cuda_cache()
                for j, prompt in enumerate(batch):
                    record_id = records[i + j].get("id", i + j)
                    response = generate_one_prompt(prompt, record_id)
                    responses.append(response)
                    if response is None:
                        failed += 1
                    if api:
                        single = generate.meta if response is not None and generate.meta else [failed_meta()]
                        collect_api_metadata(list(single))
            else:
                if api:
                    # The api backend reports a request that exhausted its retries per prompt, so one
                    # failure never re-sends the batch's other (already paid for) replies.
                    batch_responses = [None if m.get("failed") else r for r, m in zip(batch_responses, generate.meta)]
                    failed += sum(1 for r in batch_responses if r is None)
                    collect_api_metadata(list(generate.meta))
                responses.extend(batch_responses)
            flush()

            batch_answered = any(answered(k) for k in range(i, len(responses)))
            consecutive_failed_batches = 0 if batch_answered else consecutive_failed_batches + 1

            processed = len(responses)
            print(f"Generated {processed}/{len(prompts)} ({failed} failed)")
            if consecutive_failed_batches >= args.max_consecutive_failures:
                raise StoppedNoAnswer(
                    f"Aborting: {consecutive_failed_batches} consecutive batches produced no answer for any prompt "
                    f"({failed}/{processed} generations raised overall); stopping generation for this case."
                )
    except RuntimeError as e:
        # Preserve partial output before propagating the abort.
        fill_new(len(responses))
        write_out(records[: len(responses)])
        if isinstance(e, StoppedNoAnswer) and args.out:
            sidecar = {
                "reason": str(e),
                "n_prompts": len(prompts),
                "n_attempted": len(responses),
                "n_answered": sum(1 for k in range(len(responses)) if answered(k)),
                "max_consecutive_failures": args.max_consecutive_failures,
            }
            Path(args.out + ".stopped.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        if isinstance(e, run_config.BudgetExceeded) and args.out:
            sidecar = {"reason": str(e), "n_prompts": len(prompts), "n_completed": len(responses)}
            Path(args.out + ".budget.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        if isinstance(e, run_config.GenerationAborted) and args.out:
            sidecar = {"reason": str(e), "n_prompts": len(prompts), "n_completed": len(responses)}
            Path(args.out + ".aborted.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        raise

    fill_new(len(responses))

    if failed:
        print(f"WARNING: {failed}/{len(prompts)} prompts had failed generations")
    if api:
        n_no_answer = sum(r.get("no_answer", False) for r in records)
        retried = [r for r in records if (r.get("attempts") or 1) > 1]
        print(f"finish_reason counts: {dict(Counter(finish_reasons))}")
        print(f"attempts per record: {dict(sorted(Counter(attempts).items(), key=lambda kv: (kv[0] is None, kv[0])))}")
        print(
            f"records that needed a no-answer retry: {len(retried)}/{len(records)}, "
            f"recovered: {sum(1 for r in retried if not r['no_answer'])}"
        )
        print(
            f"no_answer (empty content after all attempts): {n_no_answer}/{len(records)} "
            "-- flagged generation_failed so scorers exclude them from the rate"
        )
        total_cost = sum(r.get("cost_thb") or 0.0 for r in records)
        print(f"computed cost of this case: {total_cost:.4f} THB")

    # generate() closes over model/tokenizer -- drop it before unloading.
    del generate
    if model is not None:
        model = unload(model)
    del tokenizer

    write_out(records)


if __name__ == "__main__":
    try:
        main()
    except StoppedNoAnswer as e:
        print(str(e), file=sys.stderr)
        sys.exit(STOPPED_NO_ANSWER_EXIT_CODE)
    except run_config.BudgetExceeded as e:
        print(str(e), file=sys.stderr)
        sys.exit(BUDGET_EXIT_CODE)
    except run_config.GenerationAborted as e:
        print(f"Aborting: {e}", file=sys.stderr)
        sys.exit(ABORTED_EXIT_CODE)
