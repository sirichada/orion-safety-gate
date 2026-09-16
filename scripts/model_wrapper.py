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
from pathlib import Path


def load_model(model_path: str, adapter_path: str | None = None, use_4bit: bool = True):
    """Loads a base HF causal-LM checkpoint, optionally with a PEFT adapter
    attached on top. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for batched causal-LM generation -- right padding (the HF default)
    # puts pad tokens before the real content ends, so slicing off the prompt by
    # a single input_len below would decode garbage for every shorter sequence.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def make_generate(model, tokenizer, max_input_tokens: int = 2048, apply_chat_template: bool = True):
    """Returns a generate() closure matching the fixed interface:
    generate(prompts, max_new_tokens, temperature, top_p) -> List[str].

    max_input_tokens sets an explicit truncation length -- tokenizer.model_max_length
    is a huge sentinel (e.g. ~1e30) on several tokenizers when no real limit is set,
    which makes truncation=True silently do nothing if relied on as the default.

    apply_chat_template wraps each raw prompt as a user turn via the tokenizer's own
    chat template when one is defined -- most instruct models (BackdoorLLM's
    Llama-2-Chat pair included) expect this, and skipping it materially shifts
    refusal rates versus how the benchmark's own pipeline calls the model.
    """
    import torch

    def generate(prompts: list[str], max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
        if apply_chat_template and tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
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
    parser.add_argument("--out", default=None, help="Optional path to write responses JSONL to")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter, use_4bit=not args.no_4bit)
    generate = make_generate(
        model, tokenizer, max_input_tokens=args.max_input_tokens, apply_chat_template=not args.no_chat_template
    )

    records = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line]
    prompts = [r["model_input"] for r in records]

    responses = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        responses.extend(generate(batch, args.max_new_tokens, args.temperature, args.top_p))
        print(f"Generated {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    for record, response in zip(records, responses):
        record["response"] = response

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} responses -> {args.out}")
    else:
        for record in records[:5]:
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
