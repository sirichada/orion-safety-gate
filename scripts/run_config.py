#!/usr/bin/env python3
"""
Run configuration: the settings a run is made with, what the model was shown,
and where everything came from, recorded as effective values.

Stdlib-only at import time. The suite process imports this module and must
never load the model stack, so anything that needs torch is imported inside
the function that uses it and is only called from the wrapper process.

A value that applies but could not be observed is {"available": False,
"reason": ...}; None means only "not present" (no adapter). Nothing is omitted
and no default is recorded as None.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

PRECISIONS = ("4bit", "full", "as_published")

# Single source for the values the suite forwards to every stage. The wrapper's own
# argparse defaults are not relied on.
DEFAULTS = {
    "batch_size": 8,
    "max_consecutive_failures": 3,
    "precision": "4bit",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
    "max_input_tokens": 2048,
    # API backend only. Retries re-spend a reply's tokens, so an empty cut-off reply is excluded, not re-requested.
    "api_batch_size": 12,
    "api_concurrency": 3,
    "no_answer_retries": 0,
}

# The key allows this many requests in flight; no setting may exceed it.
API_CONCURRENCY_CEILING = 3

# Caps that legitimately differ between cases; every other tested field must agree.
PER_CASE_FIELDS = ("max_new_tokens", "max_input_tokens")

PROTOCOL = {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "caller_system_prompt": "none"}

# Where the gateway accepts top_k: a top-level field is rejected with HTTP 500.
TOP_K_LOCATION = "extra_body"

# presence_penalty is accepted as a top-level field and range-checked to [-2, 2].
PRESENCE_PENALTY_LOCATION = "top_level"


def effective_sampling(temperature: float, top_p: float, top_k: int | None, presence_penalty: float | None = None) -> dict:
    """The sampling a run requested, and whether it is the protocol's (temperature 0, top_p 1, no top_k, no presence_penalty)."""
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "top_k_location": TOP_K_LOCATION if top_k is not None else None,
        "presence_penalty": presence_penalty,
        "presence_penalty_location": PRESENCE_PENALTY_LOCATION if presence_penalty is not None else None,
        "matches_protocol": (temperature, top_p, top_k, presence_penalty)
        == (PROTOCOL["temperature"], PROTOCOL["top_p"], None, None),
    }

TEMPLATE_PLACEHOLDER = "<<ORION_RUN_CONFIG_PLACEHOLDER>>"
TEMPLATE_PARAMETERS = ("enable_thinking", "reasoning_effort")

LIBRARIES = ("transformers", "torch", "bitsandbytes", "peft", "accelerate", "vllm")
SIDECAR_SUFFIX = ".run_config.json"
SCHEMA_VERSION = 1

# Fields that must agree between every case of one run: a difference means two different
# things were tested. Environment fields may differ and are flagged instead.
TESTED_FIELDS = (
    "args",
    "precision",
    "model_input",
    "per_model",
    "provenance.model_revision",
    "provenance.adapter_revision",
    "provenance.model_files",
    "provenance.adapter_files",
    "provenance.api",
)
ENVIRONMENT_FIELDS = ("provenance.libraries", "provenance.gpu")

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class RevisionError(ValueError):
    """A model or adapter revision could not be resolved to a commit."""


class PrecisionError(ValueError):
    """The requested precision cannot be applied to this checkpoint."""


class GenerationConfigError(ValueError):
    """A case has no usable generation cap for this model and backend."""


class GenerationAborted(RuntimeError):
    """The api backend raised something unexpected; the run stops rather than re-sending paid requests."""


class BudgetExceeded(RuntimeError):
    """The next request's worst case would take the run past its spend limit."""


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def sidecar_path(responses_path) -> Path:
    return Path(str(responses_path) + SIDECAR_SUFFIX)


# -------------------------------------------------------- generation caps


def load_generation_config(path) -> dict | None:
    """The per-case, per-model cap table, or None when the file does not exist.

    {"cases": {case_id: {"max_new_tokens": int, "max_input_tokens": int}},
     "models": {model_id: {case_id: {"max_new_tokens": int, ...}}}}
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GenerationConfigError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise GenerationConfigError(f"{path} must hold a JSON object")
    return config


def resolve_generation(config: dict | None, case_id: str, model_id: str, backend: str) -> dict:
    """max_new_tokens and max_input_tokens for one case.

    A model entry wins over the case default. The api backend has no fallback: a case with no
    model entry is refused, so a cap is always a recorded decision and never a leftover default.
    """
    config = config or {}
    model_entry = (config.get("models", {}).get(model_id) or {}).get(case_id) or {}
    case_entry = (config.get("cases") or {}).get(case_id) or {}
    if backend == "api":
        cap = model_entry.get("max_new_tokens")
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise GenerationConfigError(
                f"no max_new_tokens for case {case_id!r} and model {model_id!r} in the generation config; "
                "run the calibration tool and add models[model][case] before an api run"
            )
        return {"max_new_tokens": cap, "max_input_tokens": None}
    resolved = {}
    for name in PER_CASE_FIELDS:
        value = model_entry.get(name, case_entry.get(name, DEFAULTS[name]))
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise GenerationConfigError(f"{name} for case {case_id!r} must be a positive integer, got {value!r}")
        resolved[name] = value
    return resolved


def generation_sha256(resolved: dict) -> str:
    """Hash of the resolved caps of one run, independent of unrelated entries in the file."""
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- precision


def pre_quantized_method(quantization_config) -> str | None:
    """quant_method of a checkpoint's own quantization config, or None when it has none."""
    if quantization_config is None:
        return None
    if isinstance(quantization_config, dict):
        return str(quantization_config.get("quant_method", "unknown"))
    return str(getattr(quantization_config, "quant_method", "unknown"))


def check_precision(precision: str, quantization_config) -> None:
    """A checkpoint that ships quantized can only be loaded as published."""
    if precision not in PRECISIONS:
        raise PrecisionError(f"unknown precision {precision!r}; choose from {', '.join(PRECISIONS)}")
    method = pre_quantized_method(quantization_config)
    if method is not None and precision != "as_published":
        raise PrecisionError(
            f"the checkpoint is pre-quantized ({method}); --precision {precision} would re-quantize or upcast it. "
            "Use --precision as_published."
        )


def _dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def describe_precision(mode: str, model) -> dict:
    """The precision the model was actually loaded in; weights_dtype is read from the model."""
    weights = getattr(model, "dtype", None)
    if weights is None:
        try:
            weights = next(model.parameters()).dtype
        except Exception:
            weights = None
    weights_dtype = _dtype_name(weights) if weights is not None else "unknown"
    if mode == "4bit":
        return {"mode": mode, "quantization": "bitsandbytes-nf4", "compute_dtype": "float16", "weights_dtype": weights_dtype}
    config = getattr(model, "config", None)
    published = pre_quantized_method(getattr(config, "quantization_config", None))
    return {
        "mode": mode,
        "quantization": published or "none",
        "compute_dtype": weights_dtype,
        "weights_dtype": weights_dtype,
    }


# ----------------------------------------------------------------- template


def template_parameters(template: str | None) -> list[str]:
    if not template:
        return []
    return [name for name in TEMPLATE_PARAMETERS if name in template]


def render_template_record(tokenizer, template: str | None, apply: bool, source: str) -> dict:
    """What the model's chat template wraps around a user message.

    The placeholder splits the rendering into everything before and after the user text, so
    whatever the template inserts by default (a system block, a reasoning-effort line, a
    generation prompt) is recorded verbatim and not parsed.
    """
    if not apply:
        return {"applied": False, "reason": "chat template disabled"}
    if not template:
        return {"applied": False, "reason": "tokenizer has no chat template"}
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": TEMPLATE_PLACEHOLDER}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        return unavailable(f"template could not be rendered: {type(e).__name__}: {e}")
    if rendered.count(TEMPLATE_PLACEHOLDER) != 1:
        return unavailable("user text does not appear exactly once in the rendered template")
    prefix, suffix = rendered.split(TEMPLATE_PLACEHOLDER)
    return {
        "applied": True,
        "source": source,
        "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "prefix": prefix,
        "suffix": suffix,
    }


# --------------------------------------------------------------- provenance


def library_versions(names=LIBRARIES, version: Callable[[str], str] = importlib.metadata.version) -> dict:
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found


def gpu_info(torch_module=None, run: Callable = subprocess.run) -> dict:
    """GPU name(s), memory, CUDA/ROCm and driver. Wrapper-side only: imports torch."""
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        return {"device": "cpu"}
    devices = []
    for index in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(index)
        devices.append({"name": props.name, "total_memory_gb": round(props.total_memory / 1024**3, 1)})
    driver: object
    try:
        out = run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        driver = out.stdout.strip().splitlines()[0]
    except Exception as e:
        driver = unavailable(f"nvidia-smi: {type(e).__name__}")
    version = torch_module.version
    return {
        "devices": devices,
        "cuda": getattr(version, "cuda", None) or "none",
        "rocm": getattr(version, "hip", None) or "none",
        "driver": driver,
    }


def git_state(repo_root, run: Callable = subprocess.run) -> dict:
    """Suite commit and whether the code or thresholds differ from it (runs/ and data/ are ignored)."""

    def git(*args: str) -> str:
        return run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=30, check=True
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        changed = [line for line in git("status", "--porcelain", "--", "scripts", "config").splitlines() if line]
    except Exception as e:
        return unavailable(f"not a git checkout or git failed: {type(e).__name__}")
    return {"commit": commit, "dirty": bool(changed), "dirty_paths": [line[3:] for line in changed[:20]]}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_set_hashes(paths: dict) -> dict:
    return {case_id: file_sha256(path) for case_id, path in paths.items()}


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")


def local_checkpoint_identity(path) -> dict:
    """A local checkpoint has no hub commit: list its weight files and hash its config."""
    root = Path(path)
    if root.is_file():
        return {"files": [{"name": root.name, "size": root.stat().st_size}], "config_sha256": unavailable("single file")}
    files = sorted(
        ({"name": p.name, "size": p.stat().st_size} for p in root.iterdir() if p.suffix in _WEIGHT_SUFFIXES),
        key=lambda entry: entry["name"],
    )
    config = root / "config.json"
    adapter_config = root / "adapter_config.json"
    config_file = config if config.exists() else adapter_config if adapter_config.exists() else None
    return {
        "files": files,
        "config_sha256": file_sha256(config_file) if config_file else unavailable("no config.json"),
    }


# ----------------------------------------------------------------- revision


class HubClient:
    """Hugging Face Hub lookups; huggingface_hub is imported only when one is made."""

    def commit_for(self, repo_id: str, revision: str | None) -> str:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id, revision=revision or "main").sha

    def cached_commit(self, repo_id: str, revision: str | None = None) -> str | None:
        try:
            from huggingface_hub import constants
        except ImportError:
            return None

        folder = Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
        ref = folder / "refs" / (revision or "main")
        try:
            value = ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value if _COMMIT_RE.match(value) else None


HUB = HubClient()


def resolve_revision(repo_or_path: str, revision: str | None, hub=None) -> str | dict:
    """Commit for a hub repo (a branch, tag or commit given as `revision`, else main).

    A local path has no commit and returns an unavailable marker. An unresolvable repo raises
    RevisionError, so a run never starts without knowing what it is testing.
    """
    hub = hub or HUB
    if Path(repo_or_path).exists():
        if revision:
            raise RevisionError(f"{repo_or_path} is a local path; a revision does not apply to it")
        return unavailable("local path")
    if revision and _COMMIT_RE.match(revision):
        return revision
    try:
        return hub.commit_for(repo_or_path, revision)
    except Exception as e:
        cached = hub.cached_commit(repo_or_path, revision)
        if cached:
            return cached
        raise RevisionError(
            f"could not resolve {repo_or_path!r} (revision {revision or 'main'}) to a commit: "
            f"{type(e).__name__}: {e}. Pass a commit explicitly."
        ) from e


# ------------------------------------------------------------ stage reports


def build_stage_report(
    *,
    args: dict,
    precision: dict,
    model_input: dict,
    per_model: dict,
    provenance: dict,
    n_truncated_inputs,
) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "args": args,
        "precision": precision,
        "model_input": model_input,
        "per_model": per_model,
        "provenance": provenance,
        "n_truncated_inputs": n_truncated_inputs,
    }


def write_stage_report(responses_path, report: dict) -> Path:
    path = sidecar_path(responses_path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_stage_report(responses_path) -> dict:
    path = sidecar_path(responses_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ValueError(f"the wrapper wrote no run configuration ({path.name}); the run cannot state what it tested") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"run configuration {path.name} is not valid JSON: {e}") from e
    if report.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"run configuration {path.name} has schema {report.get('schema')!r}, expected {SCHEMA_VERSION}")
    return report


def _lookup(report: dict, dotted: str):
    value = report
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return {"available": False, "reason": f"field {dotted} missing from the report"}
        value = value[part]
    return value


def tested_mismatches(a: dict, b: dict) -> list[str]:
    """Names of the TESTED_FIELDS on which two stage reports differ, caps included."""
    return [field for field in TESTED_FIELDS if _lookup(a, field) != _lookup(b, field)]


def _comparable(value, field: str):
    if field in ("args", "per_model") and isinstance(value, dict):
        return {k: v for k, v in value.items() if k not in PER_CASE_FIELDS}
    return value


def check_forwarded(forwarded: dict, report: dict) -> list[str]:
    """Arguments the suite forwarded that the wrapper did not report using as given."""
    reported = report.get("args", {})
    return [
        f"{name}: forwarded {value!r}, wrapper used {reported.get(name, 'nothing')!r}"
        for name, value in forwarded.items()
        if reported.get(name, object()) != value
    ]


def merge_reports(reports: dict) -> tuple[dict, list[dict], list[dict]]:
    """Returns (baseline report, tested-field mismatches, environment-field mismatches).

    The first case is the baseline. n_truncated_inputs is per case and is not compared, and
    PER_CASE_FIELDS are left out of the args/per_model comparison (each is checked against the
    cap table by check_forwarded instead).
    """
    if not reports:
        raise ValueError("no stage reports to merge")
    ids = list(reports)
    baseline = reports[ids[0]]
    errors: list[dict] = []
    flags: list[dict] = []
    for fields, bucket in ((TESTED_FIELDS, errors), (ENVIRONMENT_FIELDS, flags)):
        for field in fields:
            values = {case_id: _comparable(_lookup(reports[case_id], field), field) for case_id in ids}
            first = values[ids[0]]
            if any(value != first for value in values.values()):
                bucket.append({"field": field, "values": values})
    return baseline, errors, flags
