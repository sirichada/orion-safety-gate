#!/usr/bin/env python3
"""
Shared HF-model loading/unloading helpers. Every model this suite loads --
the candidate model under test, HarmBench's classifier, and ThaiSafetyBench's
classifier -- goes through one of the two loaders here, and all three run
sequentially on a single 12GB GPU (one model resident in VRAM at a time), so
the 4-bit quantization config and the unload step live in exactly one place
rather than being copy-pasted per script.
"""

from __future__ import annotations


def quantization_config_4bit():
    """NF4 4-bit quantization config shared by every model load in this suite."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_causal_lm(model_path: str, adapter_path: str | None = None, use_4bit: bool = True):
    """Loads a base HF causal-LM checkpoint, optionally with a PEFT adapter
    attached on top. Returns (model, tokenizer).

    Used both for the candidate model under test (model_wrapper.py) and for
    HarmBench's classifier (score_en_text.py), which is a prompted, instruction-
    tuned Llama-2 -- not a plain sequence classifier -- so it loads the same way
    as any other causal LM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = quantization_config_4bit() if use_4bit else None

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


def load_sequence_classifier(model_path: str, use_4bit: bool = False):
    """Loads a plain HF sequence-classification checkpoint. Returns (model, tokenizer).

    Used for ThaiSafetyClassifier: its published HF repo root is a merged
    DebertaV2ForSequenceClassification checkpoint, not a bare PEFT adapter --
    load directly via AutoModelForSequenceClassification, no PeftModel wrap.

    Defaults to full precision -- this is a ~184M-parameter model with no VRAM
    pressure, unlike the 13B-scale candidate/classifier pair load_causal_lm
    handles, so 4-bit buys nothing here while adding accuracy risk.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    quantization_config = quantization_config_4bit() if use_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def unload(model):
    """Frees a model's VRAM. Explicit step for the sequential load-use-unload
    lifecycle this suite depends on (candidate model, then HarmBench-cls, then
    ThaiSafetyClassifier never coexist in the same 12GB card) -- previously
    relied on implicit process exit, which doesn't hold once a single process
    loads more than one model in sequence.

    Returns None -- callers MUST reassign their own reference (`model =
    unload(model)`), since `del model` inside this function only unbinds this
    function's local parameter, not the caller's variable. Without the
    reassignment the caller's reference keeps the weights alive and
    empty_cache() runs while VRAM is still occupied.
    """
    import gc

    import torch

    del model
    gc.collect()
    # Guarded: unconditional empty_cache() raises on a CPU-only torch build
    # with no CUDA support compiled in (as opposed to CUDA-available-but-
    # uninitialized, where it's a harmless no-op) -- this keeps a CPU-only
    # smoke test viable.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None
