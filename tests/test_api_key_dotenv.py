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


def test_base_url_comes_from_the_dotenv_before_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(api_model.BASE_URL_ENV, "https://from-env.example/v1")
    use_dotenv(monkeypatch, tmp_path, f"{api_model.BASE_URL_ENV}=https://from-dotenv.example/v1/\n")
    assert api_model.read_base_url() == "https://from-dotenv.example/v1"


def test_base_url_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv(api_model.BASE_URL_ENV, "https://from-env.example/v1")
    assert api_model.read_base_url() == "https://from-env.example/v1"


def test_no_base_url_anywhere_exits_naming_the_variable(monkeypatch):
    monkeypatch.delenv(api_model.BASE_URL_ENV, raising=False)
    with pytest.raises(SystemExit) as exc:
        api_model.read_base_url()
    assert api_model.BASE_URL_ENV in str(exc.value)


def test_the_module_holds_no_built_in_address():
    assert not hasattr(api_model, "BASE_URL")
    assert "://" not in api_model.BASE_URL_RECORDED
