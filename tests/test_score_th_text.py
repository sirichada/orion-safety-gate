"""Thai scorer: the classifier input is NFKC-normalized and the run records it."""

import unicodedata

import score_th_text

SARA_AM = "ำ"


def record(prompt="ทำ", response="น้ำ"):
    return {"id": "r1", "prompt": prompt, "response": response}


def test_input_has_sara_am_split_into_nikhahit_and_sara_aa():
    text = score_th_text.build_classifier_input(record(prompt=f"ท{SARA_AM}", response=f"น{SARA_AM}"))
    assert SARA_AM not in text
    assert text == "input: ทํา output: นํา"


def test_template_and_ascii_are_unchanged():
    assert score_th_text.build_classifier_input(record("a b", "c")) == "input: a b output: c"


def test_input_is_already_normalized():
    text = score_th_text.build_classifier_input(record())
    assert text == unicodedata.normalize("NFKC", text)


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, texts, **_kwargs):
        import torch

        batch = [texts] if isinstance(texts, str) else list(texts)
        self.texts.extend(batch)
        ids = torch.ones((len(batch), 3), dtype=torch.long)
        return FakeBatch(input_ids=ids) if not isinstance(texts, str) else {"input_ids": [1, 2, 3]}


class FakeModel:
    device = "cpu"

    def __call__(self, input_ids):
        import torch

        return type("Out", (), {"logits": torch.tensor([[0.0, 1.0]] * input_ids.shape[0])})()


def test_tokenizer_sees_normalized_text_and_records_note_it(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(score_th_text, "load_sequence_classifier", lambda *_a, **_k: (FakeModel(), tokenizer))
    monkeypatch.setattr(score_th_text, "unload", lambda _model: None)

    records = score_th_text.score_records([record(), {**record(), "id": "r2", "generation_failed": True}])

    assert tokenizer.texts and all(SARA_AM not in t for t in tokenizer.texts)
    assert records[0]["input_normalization"] == "NFKC"
    assert records[1]["input_normalization"] is None
    result = score_th_text.aggregate(records)
    assert result["input_normalization"] == "NFKC"
    assert "input normalized NFKC" in result["notes"]
