import hashlib
import importlib.metadata
import json
import subprocess
from types import SimpleNamespace

import pytest

import run_config
from conftest import FakeHub, stage_report_for


class Tokenizer:
    """Renders a template by wrapping the user text with fixed text, like a real chat template would."""

    def __init__(self, prefix="<s>[INST] ", suffix=" [/INST]", chat_template="TEMPLATE", fail=False, drop_text=False):
        self.prefix, self.suffix, self.chat_template = prefix, suffix, chat_template
        self.fail, self.drop_text = fail, drop_text
        self.calls = []

    def apply_chat_template(self, messages, chat_template=None, tokenize=False, add_generation_prompt=False):
        self.calls.append((messages, chat_template, tokenize, add_generation_prompt))
        if self.fail:
            raise ValueError("template error")
        text = "" if self.drop_text else messages[0]["content"]
        return self.prefix + text + self.suffix


def test_template_record_splits_around_the_user_text():
    tok = Tokenizer(prefix="<|im_start|>system\nReasoning effort is set to xhigh.<|im_end|>\n<|im_start|>user\n")
    record = run_config.render_template_record(tok, "TEMPLATE", True, "tokenizer")
    assert record["applied"] is True and record["source"] == "tokenizer"
    assert record["prefix"].startswith("<|im_start|>system\nReasoning effort")
    assert record["suffix"] == " [/INST]"
    assert record["sha256"] == hashlib.sha256(b"TEMPLATE").hexdigest()
    messages, template, tokenize, generation_prompt = tok.calls[0]
    assert messages == [{"role": "user", "content": run_config.TEMPLATE_PLACEHOLDER}]
    assert (template, tokenize, generation_prompt) == ("TEMPLATE", False, True)


def test_template_record_is_stable_and_marks_an_override():
    a = run_config.render_template_record(Tokenizer(), "T", True, "override")
    b = run_config.render_template_record(Tokenizer(), "T", True, "override")
    assert a == b and a["source"] == "override"


@pytest.mark.parametrize(
    "tokenizer, template, apply, key, expected",
    [
        (Tokenizer(), "T", False, "applied", False),
        (Tokenizer(chat_template=None), None, True, "applied", False),
        (Tokenizer(fail=True), "T", True, "available", False),
        (Tokenizer(drop_text=True), "T", True, "available", False),
    ],
)
def test_template_record_never_raises_and_never_claims_what_it_could_not_see(tokenizer, template, apply, key, expected):
    record = run_config.render_template_record(tokenizer, template, apply, "tokenizer")
    assert record[key] is expected
    assert "reason" in record and "prefix" not in record


def test_template_parameters_are_found_by_name():
    template = "{% if enable_thinking %}...{% endif %}{{ reasoning_effort }}"
    assert run_config.template_parameters(template) == ["enable_thinking", "reasoning_effort"]
    assert run_config.template_parameters("plain") == []
    assert run_config.template_parameters(None) == []


def test_describe_precision_reads_the_loaded_dtype():
    model = SimpleNamespace(dtype="torch.bfloat16", config=SimpleNamespace(quantization_config=None))
    assert run_config.describe_precision("full", model) == {
        "mode": "full", "quantization": "none", "compute_dtype": "bfloat16", "weights_dtype": "bfloat16",
    }
    four = run_config.describe_precision("4bit", SimpleNamespace(dtype="torch.float16"))
    assert four["quantization"] == "bitsandbytes-nf4" and four["compute_dtype"] == "float16"
    published = SimpleNamespace(dtype="torch.float16", config=SimpleNamespace(quantization_config={"quant_method": "awq"}))
    assert run_config.describe_precision("as_published", published)["quantization"] == "awq"


def test_pre_quantized_checkpoints_only_load_as_published():
    awq = {"quant_method": "awq"}
    run_config.check_precision("as_published", awq)
    for mode in ("4bit", "full"):
        with pytest.raises(run_config.PrecisionError, match="pre-quantized \\(awq\\).*as_published"):
            run_config.check_precision(mode, awq)
    for mode in run_config.PRECISIONS:
        run_config.check_precision(mode, None)
    with pytest.raises(run_config.PrecisionError):
        run_config.check_precision("fp16", None)


def test_library_versions_report_what_is_missing():
    def version(name):
        if name == "torch":
            return "2.8.0"
        raise importlib.metadata.PackageNotFoundError(name)

    found = run_config.library_versions(("torch", "vllm"), version)
    assert found == {"torch": "2.8.0", "vllm": "not installed"}


def completed(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout)


def test_git_state_clean_dirty_and_not_a_checkout(tmp_path):
    def runner(status):
        def run(cmd, **kwargs):
            return completed("c0ffee\n" if "rev-parse" in cmd else status)

        return run

    assert run_config.git_state(tmp_path, runner("")) == {"commit": "c0ffee", "dirty": False, "dirty_paths": []}
    dirty = run_config.git_state(tmp_path, runner(" M scripts/cases.py\n?? scripts/new.py\n"))
    assert dirty["dirty"] is True and dirty["dirty_paths"] == ["scripts/cases.py", "scripts/new.py"]

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    assert run_config.git_state(tmp_path, fail)["available"] is False


def test_git_state_only_looks_at_code_and_config(tmp_path):
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd)
        return completed("c0ffee\n")

    run_config.git_state(tmp_path, run)
    status = next(cmd for cmd in seen if "status" in cmd)
    assert status[-2:] == ["scripts", "config"]


def test_local_checkpoint_identity_lists_weights_and_hashes_config(tmp_path):
    (tmp_path / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "model-00001.safetensors").write_bytes(b"1234")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    identity = run_config.local_checkpoint_identity(tmp_path)
    assert identity["files"] == [{"name": "model-00001.safetensors", "size": 4}]
    assert identity["config_sha256"] == hashlib.sha256(b'{"a": 1}').hexdigest()


def test_resolve_revision_cases(tmp_path):
    commit = "d" * 40
    assert run_config.resolve_revision("org/m", commit, FakeHub()) == commit  # already a commit: no lookup
    hub = FakeHub(commit="e" * 40)
    assert run_config.resolve_revision("org/m", None, hub) == "e" * 40
    assert hub.asked == [("org/m", None)]
    assert run_config.resolve_revision(str(tmp_path), None, hub) == {"available": False, "reason": "local path"}
    with pytest.raises(run_config.RevisionError, match="local path"):
        run_config.resolve_revision(str(tmp_path), "main", hub)


def test_resolve_revision_falls_back_to_the_cache_then_fails():
    assert run_config.resolve_revision("org/m", None, FakeHub(fail=True, cached="f" * 40)) == "f" * 40
    with pytest.raises(run_config.RevisionError, match="could not resolve"):
        run_config.resolve_revision("org/m", None, FakeHub(fail=True))


def test_stage_report_round_trips_and_validates(tmp_path):
    out = tmp_path / "responses.jsonl"
    report = stage_report_for(["--model", "m"])
    run_config.write_stage_report(out, report)
    assert run_config.read_stage_report(out) == report
    run_config.sidecar_path(out).write_text(json.dumps({**report, "schema": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        run_config.read_stage_report(out)
    with pytest.raises(ValueError, match="wrote no run configuration"):
        run_config.read_stage_report(tmp_path / "other.jsonl")


def test_check_forwarded_lists_each_difference():
    report = stage_report_for(["--model", "m", "--batch-size", "4"])
    assert run_config.check_forwarded({"batch_size": 4, "model": "m"}, report) == []
    problems = run_config.check_forwarded({"batch_size": 8, "precision": "full"}, report)
    assert len(problems) == 2 and "batch_size" in problems[0] and "precision" in problems[1]


def test_merge_reports_separates_tested_fields_from_environment():
    a = stage_report_for(["--model", "m"])
    same = stage_report_for(["--model", "m"], n_truncated_inputs=5)  # per-case count is not compared
    baseline, errors, flags = run_config.merge_reports({"x": a, "y": same})
    assert baseline is a and errors == [] and flags == []

    moved = stage_report_for(["--model", "m"])
    moved["provenance"]["model_revision"] = "9" * 40
    moved["provenance"]["gpu"] = {"device": "cpu"}
    _, errors, flags = run_config.merge_reports({"x": a, "y": moved})
    assert [e["field"] for e in errors] == ["provenance.model_revision"]
    assert errors[0]["values"] == {"x": "a" * 40, "y": "9" * 40}
    assert [f["field"] for f in flags] == ["provenance.gpu"]


def test_merge_reports_catches_a_template_or_precision_change():
    a = stage_report_for(["--model", "m"])
    b = stage_report_for(["--model", "m"])
    b["model_input"]["chat_template"]["sha256"] = "0" * 64
    b["precision"]["weights_dtype"] = "float32"
    _, errors, _ = run_config.merge_reports({"x": a, "y": b})
    assert {e["field"] for e in errors} == {"model_input", "precision"}


def test_gpu_info_with_and_without_a_device():
    props = SimpleNamespace(name="RTX 5070", total_memory=12 * 1024**3)
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_properties=lambda i: props)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.8", hip=None))
    info = run_config.gpu_info(torch, lambda *a, **k: completed("572.16\n"))
    assert info == {
        "devices": [{"name": "RTX 5070", "total_memory_gb": 12.0}], "cuda": "12.8", "rocm": "none", "driver": "572.16",
    }

    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert run_config.gpu_info(torch, no_smi)["driver"]["available"] is False
    cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert run_config.gpu_info(cpu) == {"device": "cpu"}


def test_per_case_caps_may_differ_but_other_settings_may_not():
    a = stage_report_for(["--model", "m", "--max-new-tokens", "256", "--max-input-tokens", "2048"])
    b = stage_report_for(["--model", "m", "--max-new-tokens", "1024", "--max-input-tokens", "4096"])
    _, errors, _ = run_config.merge_reports({"x": a, "y": b})
    assert errors == []
    c = stage_report_for(["--model", "m", "--temperature", "0.7"])
    _, errors, _ = run_config.merge_reports({"x": a, "y": c})
    assert [e["field"] for e in errors] == ["args"]


def test_tested_mismatches_include_the_caps():
    a = stage_report_for(["--model", "m", "--max-new-tokens", "256"])
    b = stage_report_for(["--model", "m", "--max-new-tokens", "512"])
    assert run_config.tested_mismatches(a, a) == []
    assert run_config.tested_mismatches(a, b) == ["args", "per_model"]


TABLE = {
    "cases": {"en_text": {"max_new_tokens": 300, "max_input_tokens": 1500}},
    "models": {"Mango": {"en_text": {"max_new_tokens": 4096}, "th_text": {"max_new_tokens": 2048}}},
}


def test_hf_caps_fall_back_to_case_defaults_then_protocol_defaults():
    assert run_config.resolve_generation(TABLE, "en_text", "other", "hf") == {"max_new_tokens": 300, "max_input_tokens": 1500}
    assert run_config.resolve_generation(TABLE, "en_text", "Mango", "hf") == {"max_new_tokens": 4096, "max_input_tokens": 1500}
    assert run_config.resolve_generation(None, "th_text", "x", "hf") == {"max_new_tokens": 256, "max_input_tokens": 2048}


def test_api_caps_have_no_fallback():
    assert run_config.resolve_generation(TABLE, "th_text", "Mango", "api") == {"max_new_tokens": 2048, "max_input_tokens": None}
    for table, case, model in ((TABLE, "over_refusal", "Mango"), (TABLE, "en_text", "other"), (None, "en_text", "Mango")):
        with pytest.raises(run_config.GenerationConfigError, match="no max_new_tokens"):
            run_config.resolve_generation(table, case, model, "api")
    bad = {"models": {"Mango": {"en_text": {"max_new_tokens": True}}}}
    with pytest.raises(run_config.GenerationConfigError):
        run_config.resolve_generation(bad, "en_text", "Mango", "api")


def test_generation_config_file_loading(tmp_path):
    assert run_config.load_generation_config(tmp_path / "missing.json") is None
    path = tmp_path / "g.json"
    path.write_text(json.dumps(TABLE), encoding="utf-8")
    assert run_config.load_generation_config(path) == TABLE
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(run_config.GenerationConfigError):
        run_config.load_generation_config(path)


def test_generation_hash_depends_only_on_the_resolved_caps():
    one = {"en_text": {"max_new_tokens": 4096, "max_input_tokens": None}}
    assert run_config.generation_sha256(one) == run_config.generation_sha256(dict(one))
    assert run_config.generation_sha256(one) != run_config.generation_sha256({"en_text": {"max_new_tokens": 2048, "max_input_tokens": None}})


def test_effective_sampling_flags_anything_but_the_protocol():
    assert run_config.effective_sampling(0.0, 1.0, None) == {
        "temperature": 0.0, "top_p": 1.0, "top_k": None, "top_k_location": None,
        "presence_penalty": None, "presence_penalty_location": None, "matches_protocol": True,
    }
    vendor = run_config.effective_sampling(1.0, 0.95, 20)
    assert vendor["matches_protocol"] is False and vendor["top_k_location"] == "extra_body"
    assert run_config.effective_sampling(0.0, 1.0, 20)["matches_protocol"] is False
    penalised = run_config.effective_sampling(0.0, 1.0, None, 1.5)
    assert penalised["matches_protocol"] is False
    assert (penalised["presence_penalty"], penalised["presence_penalty_location"]) == (1.5, "top_level")
    assert run_config.effective_sampling(0.0, 1.0, None, 0.0)["matches_protocol"] is False
