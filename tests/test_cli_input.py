import json
import sys

import pytest

import safety_suite
from capabilities import CapabilitiesError, parse_capabilities
from cli_input import DEFAULT_THRESHOLDS, Cancelled, resolve_inputs, suggest_model_id
from orchestrator import SuiteError


def args(*argv):
    return safety_suite.parse_args(["run", "--model", "org/My-Model-7B", *argv])


class FakeAsk:
    """Records every prompt and answers from a script; None simulates Ctrl-C."""

    def __init__(self, checkbox_answers=(), text_answer="typed-id", confirm_answer=False):
        self.checkbox_answers = list(checkbox_answers)
        self.text_answer = text_answer
        self.confirm_answer = confirm_answer
        self.asked = []
        self.text_default = None

    def confirm(self, message, default=True):
        self.asked.append(message)
        return self.confirm_answer

    def choice(self, title, value, checked=False, disabled=None):
        return {"title": title, "value": value, "checked": checked, "disabled": disabled}

    def checkbox(self, message, choices, validate=None):
        self.asked.append(message)
        return self.checkbox_answers.pop(0)

    def text(self, message, default=""):
        self.asked.append(message)
        self.text_default = default
        return self.text_answer


def test_flags_match_the_equivalent_file(tmp_path):
    caps = tmp_path / "c.json"
    caps.write_text(json.dumps({"declared_modalities": ["text"], "declared_languages": ["en", "th"]}))
    from_file = resolve_inputs(args("--model-id", "x", "--capabilities", str(caps)), False)
    from_flags = resolve_inputs(args("--model-id", "x", "--modalities", "text", "--languages", "en", "th"), False)
    assert from_file.capabilities == from_flags.capabilities
    assert (from_file.source, from_flags.source) == ("file", "flags")
    assert from_flags.thresholds_path == DEFAULT_THRESHOLDS


def test_file_and_flags_together_is_an_error(tmp_path):
    with pytest.raises(SuiteError, match="not both"):
        resolve_inputs(args("--model-id", "x", "--capabilities", "c.json", "--languages", "en"), False)


def test_non_interactive_names_what_is_missing():
    with pytest.raises(SuiteError, match="no interactive terminal") as e:
        resolve_inputs(args(), False)
    assert "--modalities" in str(e.value) and "--languages" in str(e.value) and "--model-id" in str(e.value)


def test_flags_without_text_hit_the_text_rule():
    with pytest.raises(CapabilitiesError, match="must include 'text'"):
        resolve_inputs(args("--model-id", "x", "--modalities", "image", "--languages", "en"), False)


def test_partial_flags_only_ask_for_what_is_missing():
    ask = FakeAsk(checkbox_answers=[["th"]])
    inputs = resolve_inputs(args("--model-id", "x", "--modalities", "text"), True, lambda: ask)
    assert ask.asked == ["Which languages does it support?"]
    assert inputs.capabilities == parse_capabilities({"declared_modalities": ["text"], "declared_languages": ["th"]})
    assert inputs.source == "interactive"


def test_text_only_when_images_declined():
    ask = FakeAsk(checkbox_answers=[["th"]], confirm_answer=False)
    inputs = resolve_inputs(args("--model-id", "x"), True, lambda: ask)
    assert inputs.capabilities.declared_modalities == frozenset({"text"})


def test_interactive_fills_everything_and_keeps_text():
    ask = FakeAsk(checkbox_answers=[["en"]], text_answer="my-model", confirm_answer=True)
    inputs = resolve_inputs(args(), True, lambda: ask)
    assert inputs.capabilities.declared_modalities == frozenset({"text", "image"})
    assert inputs.model_id == "my-model"


@pytest.mark.parametrize(
    "backend_flags, expected",
    [(["--backend", "api", "--dry-run"], "Qwen 3.8-Flash-Next"), (["--backend", "hf"], "qwen-3.8-flash-next")],
)
def test_model_id_prompt_default_depends_on_the_backend(backend_flags, expected):
    ask = FakeAsk(checkbox_answers=[["en"]])
    resolve_inputs(args("--model", "Qwen 3.8-Flash-Next", *backend_flags), True, lambda: ask)
    assert ask.text_default == expected


@pytest.mark.parametrize("position", ["images", "languages", "model_id"])
def test_none_from_any_prompt_cancels(position):
    ask = FakeAsk(
        checkbox_answers=[None if position == "languages" else ["en"]],
        text_answer=None if position == "model_id" else "id",
        confirm_answer=None if position == "images" else False,
    )
    with pytest.raises(Cancelled):
        resolve_inputs(args(), True, lambda: ask)


def test_unavailable_prompts_fall_back_to_the_flags_message():
    def broken():
        raise RuntimeError("no console screen buffer")

    with pytest.raises(SuiteError, match="no interactive terminal"):
        resolve_inputs(args(), True, broken)


@pytest.mark.parametrize(
    "model, expected",
    [("org/My-Model-7B", "my-model-7b"), ("/Users/me/models/Llama 2 Chat/", "llama-2-chat"), ("Mango 1.5-9B", "mango-1.5-9b")],
)
def test_model_id_suggestion_uses_the_last_segment(model, expected):
    assert suggest_model_id(model) == expected


@pytest.mark.skipif(
    not (sys.platform == "win32" and sys.stdin.isatty() and sys.stdout.isatty()),
    reason="builds real questionary prompts, which need a Windows console (run pytest -s from a console)",
)
def test_real_questionary_accepts_every_prompt_shape(monkeypatch):
    questionary = pytest.importorskip("questionary")
    from cli_input import QuestionaryAsk

    built = []

    def capture(factory):
        def wrapper(*a, **k):
            factory(*a, **k)  # raises if questionary rejects the arguments
            built.append(factory.__name__)
            return type("Q", (), {"ask": lambda self: "x"})()

        return wrapper

    for name in ("checkbox", "text", "confirm"):
        monkeypatch.setattr(questionary, name, capture(getattr(questionary, name)))
    ask = QuestionaryAsk()
    ask.checkbox("m", [ask.choice("image", "image")])
    ask.checkbox("l", [ask.choice("English", "en", checked=True)], validate=lambda picked: bool(picked) or "pick")
    ask.text("id", default="x")
    ask.confirm("go?", default=True)
    assert built == ["checkbox", "checkbox", "text", "confirm"]
