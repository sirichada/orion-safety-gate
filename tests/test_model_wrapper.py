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


def test_unloadable_precision_exits_with_the_config_error_code_and_message(wrapper, monkeypatch):
    def refuse(model, adapter, **kwargs):
        raise run_config.PrecisionError("the checkpoint is pre-quantized (awq)")

    monkeypatch.setattr(model_wrapper, "load_model", refuse)
    with pytest.raises(SystemExit) as exit_info:
        wrapper.run("--out", str(wrapper.out))
    assert exit_info.value.code == model_wrapper.CONFIG_ERROR_EXIT_CODE
    error = json.loads((wrapper.out.parent / "responses.jsonl.error.json").read_text(encoding="utf-8"))
    assert error == {"kind": "PrecisionError", "message": "the checkpoint is pre-quantized (awq)"}
    assert not wrapper.out.exists()


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


class FakeApi:
    """Stands in for scripts/api_model.py: records calls, and can fail prompts or run out of budget."""

    BASE_URL = "https://api.example/v1"

    def __init__(self):
        self.calls = []
        self.guards = []
        self.fail_texts = set()
        self.length_texts = set()
        self.budget_error_on_call = None
        self.error_on_call = None
        self.make_kwargs = None

    def install(self, monkeypatch):
        api = self

        class SpendGuard:
            def __init__(self, model, limit_thb, ledger_path=None):
                self.model, self.limit_thb, self.ledger_path = model, limit_thb, ledger_path
                api.guards.append(self)

        def make_api_generate(
            model_id, concurrency=3, base_url=None, no_answer_retries=0, spend=None, transport=None, extra_params=None
        ):
            api.make_kwargs = {
                "concurrency": concurrency,
                "no_answer_retries": no_answer_retries,
                "spend": spend,
                "extra_params": extra_params,
            }

            def generate(prompts, max_new_tokens, temperature, top_p, seed=None, chat_template=None):
                api.calls.append(list(prompts))
                if api.budget_error_on_call is not None and len(api.calls) == api.budget_error_on_call:
                    raise run_config.BudgetExceeded("spend limit 1 THB reached")
                if api.error_on_call is not None and len(api.calls) == api.error_on_call:
                    raise PermissionError("ledger is locked")
                texts, metas = [], []
                for prompt in prompts:
                    failed = prompt in api.fail_texts
                    cut = prompt in api.length_texts
                    texts.append("" if failed or cut else "answer")
                    metas.append({
                        "finish_reason": None if failed else ("length" if cut else "stop"),
                        "attempts": 1,
                        "prompt_tokens": 5,
                        "completion_tokens": 7,
                        "latency_s": 0.5,
                        "cost_thb": 0.01,
                        "reasoning_content": "thinking about " + prompt,
                        "failed": failed,
                        "error": "HTTP 400: bad" if failed else None,
                    })
                generate.meta = metas
                generate.finish_reasons = [m["finish_reason"] for m in metas]
                generate.attempts = [m["attempts"] for m in metas]
                return texts

            generate.meta, generate.finish_reasons, generate.attempts = [], [], []
            return generate

        module = SimpleNamespace(BASE_URL=self.BASE_URL, make_api_generate=make_api_generate, SpendGuard=SpendGuard)
        monkeypatch.setitem(sys.modules, "api_model", module)
        return self


@pytest.fixture
def fake_api(monkeypatch):
    return FakeApi().install(monkeypatch)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_api_backend_reports_server_side_values_as_unavailable(wrapper, fake_api):
    wrapper.run("--out", str(wrapper.out), "--backend", "api", "--max-thb", "5")
    report = run_config.read_stage_report(wrapper.out)
    assert report["precision"]["available"] is False
    assert report["provenance"]["api"]["base_url"] == "https://api.example/v1"
    assert report["n_truncated_inputs"]["available"] is False
    assert report["args"]["api_concurrency"] == 3 and report["args"]["no_answer_retries"] == 0
    assert report["args"]["stream"] is True
    assert fake_api.make_kwargs["concurrency"] == 3 and fake_api.make_kwargs["no_answer_retries"] == 0
    assert fake_api.guards[0].limit_thb == 5.0
    assert wrapper.loads == []


@pytest.mark.parametrize(
    "extra",
    [
        [],  # --max-thb is required
        ["--max-thb", "0"],
        ["--max-thb", "5", "--api-concurrency", "4"],
        ["--max-thb", "5", "--api-concurrency", "0"],
    ],
)
def test_api_backend_needs_a_spend_limit_and_at_most_three_in_flight(wrapper, fake_api, extra):
    with pytest.raises(SystemExit):
        wrapper.run("--backend", "api", *extra)
    assert fake_api.calls == []


def test_spend_limit_is_api_only_and_resume_needs_out(wrapper):
    with pytest.raises(SystemExit):
        wrapper.run("--max-thb", "5")
    with pytest.raises(SystemExit):
        wrapper.run("--resume")


def test_api_records_carry_usage_and_the_trace_goes_to_a_sidecar(wrapper, fake_api):
    wrapper.run("--out", str(wrapper.out), "--backend", "api", "--max-thb", "5", "--batch-size", "2", "--max-new-tokens", "64")
    records = read_jsonl(wrapper.out)
    assert [r["id"] for r in records] == ["p0", "p1", "p2", "p3"]
    first = records[0]
    assert first["response"] == "answer" and first["generation_failed"] is False and first["no_answer"] is False
    assert first["finish_reason"] == "stop" and first["attempts"] == 1 and first["cap_used"] == 64
    assert first["usage"] == {"prompt_tokens": 5, "completion_tokens": 7}
    assert first["latency_s"] == 0.5 and first["cost_thb"] == 0.01
    assert all("reasoning_content" not in r and "thinking" not in r["response"] for r in records)
    sidecar = read_jsonl(wrapper.out.parent / "responses.jsonl.reasoning.jsonl")
    assert [(e["index"], e["id"]) for e in sidecar] == [(0, "p0"), (1, "p1"), (2, "p2"), (3, "p3")]
    assert sidecar[2]["reasoning_content"] == "thinking about q2"
    assert [len(c) for c in fake_api.calls] == [2, 2]


def test_a_failed_prompt_is_marked_without_resending_the_batch(wrapper, fake_api):
    fake_api.fail_texts = {"q1"}
    wrapper.run("--out", str(wrapper.out), "--backend", "api", "--max-thb", "5")
    records = read_jsonl(wrapper.out)
    assert [r["generation_failed"] for r in records] == [False, True, False, False]
    assert records[1]["no_answer"] is False and records[1]["generation_error"] == "HTTP 400: bad"
    assert len(fake_api.calls) == 1  # one batch, nothing re-sent per prompt


def test_an_empty_reply_cut_off_at_the_cap_is_excluded_not_scored(wrapper, fake_api):
    fake_api.length_texts = {"q2"}
    wrapper.run("--out", str(wrapper.out), "--backend", "api", "--max-thb", "5")
    records = read_jsonl(wrapper.out)
    assert (records[2]["no_answer"], records[2]["generation_failed"]) == (True, True)
    assert records[2]["finish_reason"] == "length"
    assert [r["no_answer"] for r in records] == [False, False, True, False]


def test_budget_stop_keeps_finished_batches_and_resume_continues_them(wrapper, fake_api):
    argv = ["--out", str(wrapper.out), "--backend", "api", "--max-thb", "5", "--batch-size", "2"]
    fake_api.budget_error_on_call = 2
    with pytest.raises(run_config.BudgetExceeded):
        wrapper.run(*argv)
    assert [r["id"] for r in read_jsonl(wrapper.out)] == ["p0", "p1"]
    stop = json.loads((wrapper.out.parent / "responses.jsonl.budget.json").read_text(encoding="utf-8"))
    assert stop["n_completed"] == 2 and stop["n_prompts"] == 4

    fake_api.budget_error_on_call = None
    fake_api.calls.clear()
    wrapper.run(*argv, "--resume")
    assert fake_api.calls == [["q2", "q3"]]
    records = read_jsonl(wrapper.out)
    assert [r["id"] for r in records] == ["p0", "p1", "p2", "p3"]
    assert all(r["response"] == "answer" and r["usage"]["completion_tokens"] == 7 for r in records)
    sidecar = read_jsonl(wrapper.out.parent / "responses.jsonl.reasoning.jsonl")
    assert [e["index"] for e in sidecar] == [0, 1, 2, 3]


def test_resume_refuses_a_run_with_different_settings(wrapper, fake_api):
    argv = ["--out", str(wrapper.out), "--backend", "api", "--max-thb", "5", "--batch-size", "2"]
    fake_api.budget_error_on_call = 2
    with pytest.raises(run_config.BudgetExceeded):
        wrapper.run(*argv, "--max-new-tokens", "64")
    fake_api.budget_error_on_call = None
    with pytest.raises(SystemExit) as exit_info:
        wrapper.run(*argv, "--max-new-tokens", "128", "--resume")
    assert exit_info.value.code == model_wrapper.CONFIG_ERROR_EXIT_CODE
    error = json.loads((wrapper.out.parent / "responses.jsonl.error.json").read_text(encoding="utf-8"))
    assert error["kind"] == "ResumeMismatch" and "args" in error["message"]
    assert [r["id"] for r in read_jsonl(wrapper.out)] == ["p0", "p1"]  # untouched


def test_resume_drops_a_torn_final_line(wrapper, fake_api):
    argv = ["--out", str(wrapper.out), "--backend", "api", "--max-thb", "5", "--batch-size", "2"]
    fake_api.budget_error_on_call = 2
    with pytest.raises(run_config.BudgetExceeded):
        wrapper.run(*argv)
    with open(wrapper.out, "a", encoding="utf-8") as f:
        f.write('{"id": "p2", "respon')
    fake_api.budget_error_on_call = None
    fake_api.calls.clear()
    wrapper.run(*argv, "--resume")
    assert fake_api.calls == [["q2", "q3"]]
    assert len(read_jsonl(wrapper.out)) == 4


def test_unexpected_api_error_stops_the_run_without_re_sending_the_batch(wrapper, fake_api):
    argv = ["--out", str(wrapper.out), "--backend", "api", "--max-thb", "5", "--batch-size", "2"]
    fake_api.error_on_call = 2
    with pytest.raises(run_config.GenerationAborted, match="PermissionError: ledger is locked"):
        wrapper.run(*argv)
    assert len(fake_api.calls) == 2  # no per-prompt fallback: the failing batch was not split and re-sent
    assert [r["id"] for r in read_jsonl(wrapper.out)] == ["p0", "p1"]
    stop = json.loads((wrapper.out.parent / "responses.jsonl.aborted.json").read_text(encoding="utf-8"))
    assert stop["n_completed"] == 2 and stop["n_prompts"] == 4 and "PermissionError" in stop["reason"]

    fake_api.error_on_call = None
    fake_api.calls.clear()
    wrapper.run(*argv, "--resume")
    assert fake_api.calls == [["q2", "q3"]]
    assert [r["id"] for r in read_jsonl(wrapper.out)] == ["p0", "p1", "p2", "p3"]


def test_hf_backend_still_falls_back_to_per_prompt_on_an_unexpected_error(wrapper, monkeypatch):
    calls = []

    def flaky(prompts, *args, **kwargs):
        calls.append(len(prompts))
        if len(prompts) > 1:
            raise RuntimeError("CUDA out of memory")
        return ["answer"]

    flaky.count_truncated = lambda prompts, chat_template=None: 0
    monkeypatch.setattr(model_wrapper, "make_generate", lambda *a, **k: flaky)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    wrapper.run("--out", str(wrapper.out), "--batch-size", "4")
    assert calls == [4, 1, 1, 1, 1]
    assert all(r["response"] == "answer" for r in read_jsonl(wrapper.out))


def test_top_k_is_sent_in_extra_body_and_reported(wrapper, fake_api):
    wrapper.run("--out", str(wrapper.out), "--backend", "api", "--max-thb", "5", "--temperature", "1.0",
                "--top-p", "0.95", "--top-k", "20")
    report = run_config.read_stage_report(wrapper.out)
    assert fake_api.make_kwargs["extra_params"] == {"extra_body": {"top_k": 20}}
    assert (report["args"]["temperature"], report["args"]["top_p"], report["args"]["top_k"]) == (1.0, 0.95, 20)


def test_no_top_k_sends_no_extra_params_and_reports_none(wrapper, fake_api):
    wrapper.run("--out", str(wrapper.out), "--backend", "api", "--max-thb", "5")
    assert fake_api.make_kwargs["extra_params"] is None
    assert run_config.read_stage_report(wrapper.out)["args"]["top_k"] is None


@pytest.mark.parametrize("extra", [["--top-k", "20"], ["--backend", "api", "--max-thb", "5", "--top-k", "0"]])
def test_top_k_is_api_only_and_positive(wrapper, fake_api, extra):
    with pytest.raises(SystemExit):
        wrapper.run(*extra)
