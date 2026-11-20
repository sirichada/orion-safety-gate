import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import weight_diff as wd  # noqa: E402

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def write_model(folder, tensors, shards=1):
    folder.mkdir(parents=True)
    names = sorted(tensors)
    if shards == 1:
        safetensors_torch.save_file(tensors, str(folder / "model.safetensors"))
        return
    weight_map = {}
    for i in range(shards):
        part = {n: tensors[n] for n in names[i::shards]}
        file = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        safetensors_torch.save_file(part, str(folder / file))
        weight_map.update({n: file for n in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")


def base_tensors():
    g = torch.Generator().manual_seed(0)
    return {
        "model.embed_tokens.weight": torch.randn(4, 3, generator=g),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(3, 3, generator=g),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(3, 3, generator=g),
        "model.layers.0.mlp.up_proj.weight": torch.randn(3, 3, generator=g),
    }


def test_identical_models_have_no_difference(tmp_path):
    tensors = base_tensors()
    write_model(tmp_path / "a", tensors)
    write_model(tmp_path / "b", dict(tensors), shards=2)
    result = wd.compare(tmp_path / "a", tmp_path / "b")
    assert result["n_tensors_compared"] == 4 and result["overall"]["n_changed"] == 0 and result["overall"]["relative_l2"] == 0.0


def test_a_small_edit_to_one_group_is_found_and_ranked_first(tmp_path):
    tensors = base_tensors()
    edited = dict(tensors)
    edited["model.layers.0.self_attn.o_proj.weight"] = tensors["model.layers.0.self_attn.o_proj.weight"] + 0.5
    write_model(tmp_path / "a", tensors)
    write_model(tmp_path / "b", edited)
    result = wd.compare(tmp_path / "a", tmp_path / "b")
    top = result["groups"][0]
    assert top["group"] == "model.layers.N.self_attn.o_proj.weight" and top["n_tensors"] == 2
    assert top["n_changed"] == 9 and top["fraction_changed"] == pytest.approx(0.5) and top["max_abs_diff"] == pytest.approx(0.5)
    changed = next(t for t in result["tensors"] if t["name"] == "model.layers.0.self_attn.o_proj.weight")
    assert changed["fraction_changed"] == 1.0 and changed["layer"] == 0 and changed["mean_abs_diff"] == pytest.approx(0.5)
    assert next(t for t in result["tensors"] if t["name"] == "model.embed_tokens.weight")["n_changed"] == 0
    text = wd.summary_text(result)
    assert "model.layers.N.self_attn.o_proj.weight" in text and "groups with no differing value: 2 of 3" in text


def test_missing_and_mismatched_tensors_are_listed_not_hidden(tmp_path):
    tensors = base_tensors()
    other = dict(tensors)
    del other["model.layers.0.mlp.up_proj.weight"]
    other["extra.weight"] = torch.zeros(2)
    other["model.embed_tokens.weight"] = torch.zeros(5, 3)
    write_model(tmp_path / "a", tensors)
    write_model(tmp_path / "b", other)
    result = wd.compare(tmp_path / "a", tmp_path / "b")
    assert result["only_in_a"] == ["model.layers.0.mlp.up_proj.weight"] and result["only_in_b"] == ["extra.weight"]
    assert result["shape_mismatches"][0]["name"] == "model.embed_tokens.weight" and result["n_tensors_compared"] == 2


def test_bfloat16_tensors_are_compared(tmp_path):
    a = {"w": torch.ones(4, dtype=torch.bfloat16)}
    b = {"w": torch.ones(4, dtype=torch.bfloat16) * 2}
    write_model(tmp_path / "a", a)
    write_model(tmp_path / "b", b)
    result = wd.compare(tmp_path / "a", tmp_path / "b")
    assert result["overall"]["relative_l2"] == pytest.approx(1.0) and result["tensors"][0]["dtype_a"] == "torch.bfloat16"


def test_main_writes_json_and_text_and_states_no_conclusion(tmp_path, capsys):
    tensors = base_tensors()
    write_model(tmp_path / "a", tensors)
    write_model(tmp_path / "b", tensors)
    out = tmp_path / "out"
    wd.main(["--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"), "--out", str(out)])
    assert json.loads((out / "weight_diff.json").read_text(encoding="utf-8"))["n_tensors_compared"] == 4
    assert "overall:" in (out / "weight_diff.txt").read_text(encoding="utf-8")
