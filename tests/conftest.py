import sys
from pathlib import Path

# Scripts import each other by bare module name, so tests need scripts/ on the path the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def stage_report_for(cmd: list[str], **overrides) -> dict:
    """A run_config report consistent with the flags a model_wrapper stage was given."""
    import run_config

    def flag(name, cast=str, default=None):
        return cast(cmd[cmd.index(name) + 1]) if name in cmd else default

    backend = flag("--backend", default="hf")
    d = run_config.DEFAULTS
    args = {
        "backend": backend,
        "model": flag("--model"),
        "adapter": flag("--adapter"),
        "batch_size": flag("--batch-size", int, d["batch_size"]),
        "max_consecutive_failures": flag("--max-consecutive-failures", int, d["max_consecutive_failures"]),
        "temperature": flag("--temperature", float, d["temperature"]),
        "top_p": flag("--top-p", float, d["top_p"]),
        "max_new_tokens": flag("--max-new-tokens", int, d["max_new_tokens"]),
        "model_revision": flag("--model-revision"),
        "adapter_revision": flag("--adapter-revision"),
    }
    if backend == "hf":
        args["precision"] = flag("--precision", default=d["precision"])
        args["max_input_tokens"] = flag("--max-input-tokens", int, d["max_input_tokens"])
        precision = {
            "mode": args["precision"],
            "quantization": "bitsandbytes-nf4",
            "compute_dtype": "float16",
            "weights_dtype": "float16",
        }
        template = {"applied": True, "source": "tokenizer", "sha256": "t" * 64, "prefix": "<s>[INST] ", "suffix": " [/INST]"}
        provenance = {
            "model_revision": args["model_revision"] or "a" * 40,
            "adapter_revision": args["adapter_revision"],
            "model_files": None,
            "adapter_files": None,
            "libraries": {"transformers": "4.0.0", "torch": "2.0.0"},
            "gpu": {"devices": [{"name": "Test GPU", "total_memory_gb": 12.0}], "cuda": "12.8", "rocm": "none", "driver": "1"},
            "api": None,
        }
        per_model = {
            "max_new_tokens": args["max_new_tokens"],
            "max_input_tokens": args["max_input_tokens"],
            "thinking": {"template_parameters": [], "passed": {}},
        }
        truncated = 0
    else:
        precision = run_config.unavailable("api backend: weights are server-side")
        template = run_config.unavailable("server-side chat template")
        provenance = {
            "model_revision": run_config.unavailable("api backend"),
            "adapter_revision": None,
            "model_files": None,
            "adapter_files": None,
            "libraries": {"httpx": "0.28.1"},
            "gpu": run_config.unavailable("api backend"),
            "api": {
                "base_url": "https://api.example/v1",
                "reported_model": run_config.unavailable("deferred"),
                "system_fingerprint": run_config.unavailable("deferred"),
            },
        }
        per_model = {
            "max_new_tokens": args["max_new_tokens"],
            "max_input_tokens": run_config.unavailable("server-side limit"),
            "thinking": {"template_parameters": template, "passed": {}},
        }
        truncated = run_config.unavailable("server-side limit")
    report = run_config.build_stage_report(
        args=args,
        precision=precision,
        model_input={"chat_template": template},
        per_model=per_model,
        provenance=provenance,
        n_truncated_inputs=truncated,
    )
    report.update(overrides)
    return report


class FakeHub:
    """Stands in for the Hugging Face hub: every repo resolves to one commit."""

    def __init__(self, commit="a" * 40, fail=False, cached=None):
        self.commit, self.fail, self.cached, self.asked = commit, fail, cached, []

    def commit_for(self, repo_id, revision):
        self.asked.append((repo_id, revision))
        if self.fail:
            raise OSError("offline")
        return self.commit

    def cached_commit(self, repo_id, revision=None):
        return self.cached
