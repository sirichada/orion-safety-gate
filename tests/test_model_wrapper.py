import json
import sys
from types import SimpleNamespace

import pytest

import model_wrapper
import run_config


class Tokenizer:
    chat_template = "TEMPLATE"

    def apply_chat_template(self, messages, chat_template=None, tokenize=False, add_generation_prompt=False):
        return "<s>[INST] " + messages[0]["content"] + " [/INST]"


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"id": f"p{i}", "model_input": f"q{i}"}) + "\n" for i in range(4)), encoding="utf-8")
    loads = []

    def load_model(model, adapter, **kwargs):
        loads.append((model, adapter, kwargs))
        return SimpleNamespace(dtype="torch.bfloat16", config=SimpleNamespace(quantization_config=None)), Tokenizer()

    def make_generate(model, tokenizer, max_input_tokens=2048, apply_chat_template=True):
        def generate(prompts, max_new_tokens, temperature, top_p, seed=None, chat_template=None):
            return ["answer"] * len(prompts)

        generate.count_truncated = lambda prompts, chat_template=None: 1
        return generate

    monkeypatch.setattr(model_wrapper, "load_model", load_model)
    monkeypatch.setattr(model_wrapper, "make_generate", make_generate)
    monkeypatch.setattr(model_wrapper, "unload", lambda model: None)
    monkeypatch.setattr(run_config, "gpu_info", lambda: {"device": "cpu"})
    monkeypatch.setattr(run_config, "library_versions", lambda names=run_config.LIBRARIES: {"torch": "2.8.0"})

    def run_model(model, *argv):
        monkeypatch.setattr(sys, "argv", ["model_wrapper.py", "--model", model, "--prompts", str(prompts), *argv])
        model_wrapper.main()

    def run(*argv):
        run_model("org/m", *argv)

    return SimpleNamespace(run=run, run_model=run_model, out=tmp_path / "responses.jsonl", loads=loads)


def test_wrapper_writes_the_run_configuration(wrapper):
    wrapper.run("--out", str(wrapper.out), "--model-revision", "a" * 40, "--precision", "full")
    report = run_config.read_stage_report(wrapper.out)
    assert report["args"]["precision"] == "full" and report["args"]["batch_size"] == 8
    assert report["args"]["temperature"] == 0.0 and report["args"]["max_new_tokens"] == 256
    assert report["precision"] == {
        "mode": "full", "quantization": "none", "compute_dtype": "bfloat16", "weights_dtype": "bfloat16",
    }
    assert report["model_input"]["chat_template"]["prefix"] == "<s>[INST] "
    assert report["provenance"]["model_revision"] == "a" * 40
    assert report["n_truncated_inputs"] == 1
    assert wrapper.loads[0][2] == {"precision": "full", "revision": "a" * 40, "adapter_revision": None}
    assert len(wrapper.out.read_text(encoding="utf-8").splitlines()) == 4


def test_no_4bit_is_an_alias_for_full_precision(wrapper):
    wrapper.run("--out", str(wrapper.out), "--no-4bit")
    assert wrapper.loads[0][2]["precision"] == "full"


def test_a_stopped_run_still_leaves_its_configuration(wrapper, monkeypatch):
    def failing(prompts, *args, **kwargs):
        raise RuntimeError("out of memory")

    failing.count_truncated = lambda prompts, chat_template=None: 0
    monkeypatch.setattr(model_wrapper, "make_generate", lambda *a, **k: failing)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(model_wrapper.StoppedNoAnswer):
        wrapper.run("--out", str(wrapper.out), "--max-consecutive-failures", "1")
    assert run_config.read_stage_report(wrapper.out)["precision"]["mode"] == "4bit"
    assert (wrapper.out.parent / "responses.jsonl.stopped.json").exists()


def test_local_checkpoint_is_identified_by_its_files(wrapper, tmp_path):
    local = tmp_path / "ckpt"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    (local / "model.safetensors").write_bytes(b"xx")
    wrapper.run_model(str(local), "--out", str(wrapper.out))
    provenance = run_config.read_stage_report(wrapper.out)["provenance"]
    assert provenance["model_revision"]["available"] is False
    assert provenance["model_files"]["files"] == [{"name": "model.safetensors", "size": 2}]


def test_api_backend_rejects_hf_only_settings(wrapper):
    with pytest.raises(SystemExit):
        wrapper.run("--backend", "api", "--precision", "full")
    with pytest.raises(SystemExit):
        wrapper.run("--backend", "api", "--model-revision", "main")


def test_api_backend_reports_server_side_values_as_unavailable(wrapper, monkeypatch):
    fake_api = SimpleNamespace(BASE_URL="https://api.example/v1")
    monkeypatch.setitem(sys.modules, "api_model", fake_api)

    def make_api_generate(model_id, max_workers=8):
        def generate(prompts, *a, **k):
            generate.finish_reasons = ["stop"] * len(prompts)
            generate.attempts = [1] * len(prompts)
            return ["answer"] * len(prompts)

        generate.finish_reasons, generate.attempts = [], []
        return generate

    fake_api.make_api_generate = make_api_generate
    wrapper.run("--out", str(wrapper.out), "--backend", "api")
    report = run_config.read_stage_report(wrapper.out)
    assert report["precision"]["available"] is False
    assert report["provenance"]["api"]["base_url"] == "https://api.example/v1"
    assert report["n_truncated_inputs"]["available"] is False
    assert wrapper.loads == []
