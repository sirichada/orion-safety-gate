#!/usr/bin/env python3
"""
Shared HF-model loading/unloading helpers. Every model this suite loads --
the candidate model under test, HarmBench's classifier, and ThaiSafetyBench's
classifier -- goes through one of the two loaders here. They run one at a time
on a single 12GB GPU, so the 4-bit quantization config and the unload step
live in this module.
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


def load_causal_lm(
    model_path: str,
    adapter_path: str | None = None,
    use_4bit: bool = True,
    *,
    precision: str | None = None,
    revision: str | None = None,
    adapter_revision: str | None = None,
):
    """Loads a base HF causal-LM checkpoint, optionally with a PEFT adapter
    attached on top. Returns (model, tokenizer).

    Used both for the candidate model under test (model_wrapper.py) and for
    HarmBench's classifier (score_en_text.py), which is a prompted, instruction-
    tuned Llama-2 -- not a plain sequence classifier -- so it loads the same way
    as any other causal LM.

    precision selects how the weights are loaded: "4bit" (bitsandbytes NF4),
    "full" (bfloat16, set explicitly), or "as_published" (the checkpoint's own
    stored dtype and quantization, no bitsandbytes and no upcast). A checkpoint
    that ships quantized only loads as_published. precision=None keeps the
    use_4bit behaviour that the classifier loaders rely on. revision and
    adapter_revision pin the model/tokenizer and the adapter to a hub commit.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from run_config import check_precision

    load_kwargs: dict = {"device_map": "auto"}
    if revision:
        load_kwargs["revision"] = revision
    if precision is None:
        load_kwargs["quantization_config"] = quantization_config_4bit() if use_4bit else None
    else:
        config = AutoConfig.from_pretrained(model_path, revision=revision) if revision else AutoConfig.from_pretrained(model_path)
        check_precision(precision, getattr(config, "quantization_config", None))
        if precision == "4bit":
            load_kwargs["quantization_config"] = quantization_config_4bit()
        elif precision == "full":
            load_kwargs["dtype"] = torch.bfloat16
        else:
            load_kwargs["dtype"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_path, **({"revision": revision} if revision else {}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps every prompt flush against its completion, so slicing
    # the prompt off by the batch's input length decodes only generated tokens.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, adapter_path, **({"revision": adapter_revision} if adapter_revision else {})
        )

    model.eval()
    return model, tokenizer


def load_sequence_classifier(model_path: str, use_4bit: bool = False):
    """Loads a plain HF sequence-classification checkpoint. Returns (model, tokenizer).

    Used for ThaiSafetyClassifier: its published HF repo root is a merged
    DebertaV2ForSequenceClassification checkpoint, not a bare PEFT adapter --
    load directly via AutoModelForSequenceClassification, no PeftModel wrap.

    Defaults to full precision: a ~184M-parameter model needs no quantization.
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
    """Frees a model's VRAM, so the candidate model, HarmBench's classifier and
    ThaiSafetyClassifier never share the card.

    Returns None. Callers must reassign their own reference (`model =
    unload(model)`): `del model` here only unbinds this function's local
    parameter, so without the reassignment the caller's reference keeps the
    weights alive and empty_cache() runs while VRAM is still occupied.
    """
    import gc

    import torch

    del model
    gc.collect()
    # empty_cache() raises on a torch build with no CUDA support compiled in.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None
