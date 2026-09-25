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
(if --4bit, the default).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from hf_model import load_causal_lm, unload

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

        if apply_chat_template and (chat_template or tokenizer.chat_template):
            template = chat_template if chat_template is not None else tokenizer.chat_template
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    chat_template=template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
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

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or HF repo id")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter path/repo id")
    parser.add_argument("--prompts", required=True, help="JSONL file with a 'model_input' field per line")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no-chat-template", action="store_true", help="Skip the tokenizer's chat template, send raw prompts")
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per generate() call -- the full HarmBench/ThaiSafetyBench prompt sets (400/1,889) would OOM a 12GB GPU in one call")
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

    if args.backend == "api":
        if args.adapter or args.no_4bit or args.no_chat_template or args.chat_template is not None or args.max_input_tokens != 2048:
            parser.error("--adapter/--no-4bit/--no-chat-template/--chat-template/--max-input-tokens are hf-only")
        from api_model import make_api_generate

        model = tokenizer = None
        generate = make_api_generate(args.model, max_workers=args.batch_size)
    else:
        model, tokenizer = load_model(args.model, args.adapter, use_4bit=not args.no_4bit)
        generate = make_generate(
            model, tokenizer, max_input_tokens=args.max_input_tokens, apply_chat_template=not args.no_chat_template
        )

    records = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line]
    prompts = [r["model_input"] for r in records]

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

    # Populated only by backends that report a per-generation finish_reason (api); stays aligned with `responses`.
    finish_reasons: list[str | None] = []

    def collect_finish_reasons(count: int, failed: bool = False) -> None:
        if args.backend == "api":
            finish_reasons.extend([None] * count if failed else generate.finish_reasons)

    def fill_record(k: int, record: dict, response: str | None) -> None:
        record["response"] = response if response is not None else ""
        record["generation_failed"] = response is None
        if args.backend == "api":
            finish = finish_reasons[k]
            # An empty completion cut off at the token cap never answered: it must not be scored as a refusal.
            # generation_failed is the flag every scorer already excludes; no_answer keeps the cause distinguishable.
            no_answer = response is not None and not response.strip() and finish == "length"
            record["finish_reason"] = finish
            record["no_answer"] = no_answer
            if no_answer:
                record["generation_failed"] = True

    # A failed generation is None, not "" -- an empty string is a real, scoreable
    # output (e.g. immediate EOS), not an infra failure.
    MAX_FAILURE_RATE = 0.2

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
    try:
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            try:
                responses.extend(
                    generate(
                        batch, args.max_new_tokens, args.temperature, args.top_p, args.seed, args.chat_template
                    )
                )
                collect_finish_reasons(len(batch))
            except Exception as e:
                # Likely OOM -- clear cache and retry per-prompt so only the
                # actually oversized prompt(s) get marked failed.
                print(f"WARNING: batch {i}-{i + len(batch)} failed ({e}); clearing cache and retrying per-prompt")
                clear_cuda_cache()
                for j, prompt in enumerate(batch):
                    record_id = records[i + j].get("id", i + j)
                    response = generate_one_prompt(prompt, record_id)
                    responses.append(response)
                    collect_finish_reasons(1, failed=response is None)
                    if response is None:
                        failed += 1

            processed = len(responses)
            print(f"Generated {processed}/{len(prompts)} ({failed} failed)")
            if processed >= 10 and failed / processed > MAX_FAILURE_RATE:
                raise RuntimeError(
                    f"Aborting: {failed}/{processed} generations failed (>{MAX_FAILURE_RATE:.0%}) "
                    "-- likely a real bug rather than isolated OOMs; not producing a verdict over this run."
                )
    except RuntimeError:
        # Preserve partial output before propagating the abort.
        partial_records = records[: len(responses)]
        for k, (record, response) in enumerate(zip(partial_records, responses)):
            fill_record(k, record, response)
        write_out(partial_records)
        raise

    for k, (record, response) in enumerate(zip(records, responses)):
        fill_record(k, record, response)

    if failed:
        print(f"WARNING: {failed}/{len(prompts)} prompts had failed generations")
    if args.backend == "api":
        n_no_answer = sum(r["no_answer"] for r in records)
        print(f"finish_reason counts: {dict(Counter(finish_reasons))}")
        print(
            f"no_answer (empty content, finish_reason=length): {n_no_answer}/{len(records)} "
            "-- flagged generation_failed so scorers exclude them from the rate"
        )

    # generate() closes over model/tokenizer -- drop it before unloading.
    del generate
    if model is not None:
        model = unload(model)
    del tokenizer

    write_out(records)


if __name__ == "__main__":
    main()
