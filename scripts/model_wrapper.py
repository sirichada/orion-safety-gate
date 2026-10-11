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


class StoppedNoAnswer(RuntimeError):
    """Raised when consecutive batches produce no answer for any prompt."""


# Re-exported for backwards compatibility -- callers/scripts that imported
# load_model from this module directly still work.
load_model = load_causal_lm


def make_generate(model, tokenizer, max_input_tokens: int = 2048, apply_chat_template: bool = True):
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
                tokenizer, template, not args.no_chat_template, source
            )
        },
        per_model={
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "thinking": {"template_parameters": run_config.template_parameters(template), "passed": {}},
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
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling")
    parser.add_argument(
        "--chat-template", default=None, help="Override chat template string (defaults to the tokenizer's own)"
    )
    parser.add_argument("--out", default=None, help="Optional path to write responses JSONL to")
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

    if args.backend == "api":
        if (
            args.adapter
            or args.precision != run_config.DEFAULTS["precision"]
            or args.model_revision
            or args.adapter_revision
            or args.no_chat_template
            or args.chat_template is not None
            or args.max_input_tokens != run_config.DEFAULTS["max_input_tokens"]
        ):
            parser.error(
                "--adapter/--precision/--no-4bit/--model-revision/--adapter-revision/--no-chat-template/"
                "--chat-template/--max-input-tokens are hf-only"
            )
        from api_model import make_api_generate

        model = tokenizer = None
        generate = make_api_generate(args.model, max_workers=args.batch_size)
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
            model, tokenizer, max_input_tokens=args.max_input_tokens, apply_chat_template=not args.no_chat_template
        )

    records = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line]
    prompts = [r["model_input"] for r in records]

    # Written before generating, so a stopped or killed run still states what it was run with.
    if args.out:
        n_truncated = (
            run_config.unavailable("server-side limit")
            if args.backend == "api"
            else generate.count_truncated(prompts, args.chat_template)
        )
        run_config.write_stage_report(args.out, build_report(args, model, tokenizer, n_truncated))

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
        if args.backend == "api":
            return
        # Unguarded empty_cache() raises on CPU-only torch builds.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Populated only by backends that report per-generation metadata (api); both stay aligned with `responses`.
    finish_reasons: list[str | None] = []
    attempts: list[int | None] = []

    def collect_api_metadata(count: int, failed: bool = False) -> None:
        if args.backend == "api":
            finish_reasons.extend([None] * count if failed else generate.finish_reasons)
            attempts.extend([None] * count if failed else generate.attempts)

    def is_no_answer(response: str | None, finish: str | None) -> bool:
        return response is not None and not response.strip() and finish == "length"

    def answered(k: int) -> bool:
        if responses[k] is None:
            return False
        return args.backend != "api" or not is_no_answer(responses[k], finish_reasons[k])

    def fill_record(k: int, record: dict, response: str | None) -> None:
        record["response"] = response if response is not None else ""
        record["generation_failed"] = response is None
        if args.backend == "api":
            finish = finish_reasons[k]
            # An empty completion cut off at the token cap never answered: it must not be scored as a refusal.
            # generation_failed is the flag every scorer already excludes; no_answer keeps the cause distinguishable.
            no_answer = is_no_answer(response, finish)
            record["finish_reason"] = finish
            record["attempts"] = attempts[k]
            record["no_answer"] = no_answer
            if no_answer:
                record["generation_failed"] = True

    # A failed generation is None, not "" -- an empty string is a real, scoreable
    # output (e.g. immediate EOS), not an infra failure.
    def generate_one_prompt(prompt: str, record_id) -> str | None:
        try:
            return generate(
                [prompt], args.max_new_tokens, args.temperature, args.top_p, args.seed, args.chat_template
            )[0]
        except Exception as e:
            print(f"WARNING: generation failed for record {record_id!r}: {e}")
            return None

    responses: list[str | None] = []
    failed = 0
    # Only unbroken runs of fully-failed batches count; scattered failures never abort.
    consecutive_failed_batches = 0
    try:
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            # Only generate() sits inside the try: anything that raised after responses grew would
            # re-append the batch on retry and misalign every later record.
            try:
                batch_responses = generate(
                    batch, args.max_new_tokens, args.temperature, args.top_p, args.seed, args.chat_template
                )
            except Exception as e:
                # Likely OOM -- clear cache and retry per-prompt so only the
                # actually oversized prompt(s) get marked failed.
                print(f"WARNING: batch {i}-{i + len(batch)} failed ({e}); clearing cache and retrying per-prompt")
                clear_cuda_cache()
                for j, prompt in enumerate(batch):
                    record_id = records[i + j].get("id", i + j)
                    response = generate_one_prompt(prompt, record_id)
                    responses.append(response)
                    collect_api_metadata(1, failed=response is None)
                    if response is None:
                        failed += 1
            else:
                responses.extend(batch_responses)
                collect_api_metadata(len(batch))

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
        partial_records = records[: len(responses)]
        for k, (record, response) in enumerate(zip(partial_records, responses)):
            fill_record(k, record, response)
        write_out(partial_records)
        if isinstance(e, StoppedNoAnswer) and args.out:
            sidecar = {
                "reason": str(e),
                "n_prompts": len(prompts),
                "n_attempted": len(responses),
                "n_answered": sum(1 for k in range(len(responses)) if answered(k)),
                "max_consecutive_failures": args.max_consecutive_failures,
            }
            Path(args.out + ".stopped.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        raise

    for k, (record, response) in enumerate(zip(records, responses)):
        fill_record(k, record, response)

    if failed:
        print(f"WARNING: {failed}/{len(prompts)} prompts had failed generations")
    if args.backend == "api":
        n_no_answer = sum(r["no_answer"] for r in records)
        retried = [r for r in records if (r["attempts"] or 1) > 1]
        print(f"finish_reason counts: {dict(Counter(finish_reasons))}")
        print(f"attempts per record: {dict(sorted(Counter(attempts).items(), key=lambda kv: (kv[0] is None, kv[0])))}")
        print(
            f"records that needed a no-answer retry: {len(retried)}/{len(records)}, "
            f"recovered: {sum(1 for r in retried if not r['no_answer'])}"
        )
        print(
            f"no_answer (still empty at finish_reason=length after all attempts): {n_no_answer}/{len(records)} "
            "-- flagged generation_failed so scorers exclude them from the rate"
        )

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
