import pytest

import api_model

NAME = api_model.API_KEY_ENV


def use_dotenv(monkeypatch, tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(api_model, "DOTENV_PATH", path)


def test_dotenv_key_is_used_when_the_environment_has_none(monkeypatch, tmp_path):
    monkeypatch.delenv(NAME, raising=False)
    use_dotenv(monkeypatch, tmp_path, f"{NAME}=sk-from-dotenv\n")
    assert api_model._read_api_key() == "sk-from-dotenv"


def test_dotenv_wins_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(NAME, "sk-stale-env")
    use_dotenv(monkeypatch, tmp_path, f"{NAME}=sk-from-dotenv\n")
    assert api_model._read_api_key() == "sk-from-dotenv"


def test_environment_is_used_without_a_dotenv(monkeypatch):
    monkeypatch.setenv(NAME, "sk-from-env")
    assert api_model._read_api_key() == "sk-from-env"


@pytest.mark.parametrize(
    "text",
    [
        f'{NAME}="sk-quoted"',
        f"{NAME}='sk-quoted'",
        f"export {NAME}=sk-quoted",
        f"# comment\n\nOTHER=x\n{NAME} = sk-quoted  \n",
        f"{NAME}=old\n{NAME}=sk-quoted\n",
        f"﻿{NAME}=sk-quoted\r\n",
    ],
)
def test_dotenv_line_forms(monkeypatch, tmp_path, text):
    monkeypatch.delenv(NAME, raising=False)
    use_dotenv(monkeypatch, tmp_path, text)
    assert api_model._read_api_key() == "sk-quoted"


def test_an_empty_dotenv_value_falls_back_to_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(NAME, "sk-from-env")
    use_dotenv(monkeypatch, tmp_path, f"{NAME}=\n")
    assert api_model._read_api_key() == "sk-from-env"


def test_no_key_anywhere_exits_without_naming_a_value(monkeypatch, capsys):
    monkeypatch.delenv(NAME, raising=False)
    with pytest.raises(SystemExit) as exc:
        api_model._read_api_key()
    assert NAME in str(exc.value)
