"""
Compare the weights of two checkpoints tensor by tensor, without loading either model.

Reads the safetensors shards of both sides one tensor at a time (memory-mapped), so peak memory is one tensor in
float32. Reports, per tensor and per tensor group (the name with its layer number removed): how many values differ,
the largest and mean absolute difference, and the relative L2 norm of the difference. It states numbers only and
draws no conclusion about what the edit is.

    python tools/weight_diff.py --a Qwen/Qwen3-8B --b mlabonne/Qwen3-8B-abliterated [--out DIR]

--a and --b are local folders or Hugging Face repo ids already in the local cache (nothing is downloaded).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LAYER = re.compile(r"\.(\d+)\.")
TOP = 12


def resolve(spec: str) -> Path:
    path = Path(spec)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec, local_files_only=True, allow_patterns=["*.safetensors", "*.json"]))


def tensor_files(folder: Path) -> dict[str, Path]:
    """Tensor name -> the safetensors file that holds it."""
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return {name: folder / shard for name, shard in weight_map.items()}
    from safetensors import safe_open

    single = folder / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"{folder}: no model.safetensors.index.json or model.safetensors")
    with safe_open(str(single), framework="pt") as handle:
        return {name: single for name in handle.keys()}


def group_of(name: str) -> tuple[str, int | None]:
    """The tensor name with its layer number removed, and that number."""
    match = LAYER.search(name)
    if not match:
        return name, None
    return LAYER.sub(".N.", name, count=1), int(match.group(1))


def compare_tensor(a, b) -> dict:
    """Difference statistics of two tensors of the same shape (computed in float32)."""
    import torch

    n = a.numel()
    changed = int((a != b).sum().item())
    af, bf = a.to(torch.float32), b.to(torch.float32)
    diff = (af - bf).abs()
    sum_sq_diff = float((af - bf).double().pow(2).sum().item())
    sum_sq_a = float(af.double().pow(2).sum().item())
    return {
        "n": n,
        "n_changed": changed,
        "max_abs_diff": float(diff.max().item()) if n else 0.0,
        "sum_abs_diff": float(diff.double().sum().item()),
        "sum_sq_diff": sum_sq_diff,
        "sum_sq_a": sum_sq_a,
    }


def finish(stats: dict) -> dict:
    n = stats["n"]
    norm_a = stats["sum_sq_a"] ** 0.5
    return {
        "n": n,
        "n_changed": stats["n_changed"],
        "fraction_changed": stats["n_changed"] / n if n else 0.0,
        "max_abs_diff": stats["max_abs_diff"],
        "mean_abs_diff": stats["sum_abs_diff"] / n if n else 0.0,
        "relative_l2": (stats["sum_sq_diff"] ** 0.5 / norm_a) if norm_a else None,
    }


def merge(total: dict | None, stats: dict) -> dict:
    if total is None:
        return dict(stats, n_tensors=1)
    return {
        "n": total["n"] + stats["n"],
        "n_changed": total["n_changed"] + stats["n_changed"],
        "max_abs_diff": max(total["max_abs_diff"], stats["max_abs_diff"]),
        "sum_abs_diff": total["sum_abs_diff"] + stats["sum_abs_diff"],
        "sum_sq_diff": total["sum_sq_diff"] + stats["sum_sq_diff"],
        "sum_sq_a": total["sum_sq_a"] + stats["sum_sq_a"],
        "n_tensors": total["n_tensors"] + 1,
    }


def compare(folder_a: Path, folder_b: Path, progress=None) -> dict:
    from safetensors import safe_open

    files_a, files_b = tensor_files(folder_a), tensor_files(folder_b)
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    tensors, groups, overall = [], {}, None
    mismatched = []
    handles: dict[Path, object] = {}

    def handle(path: Path):
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt")
        return handles[path]

    try:
        for i, name in enumerate(sorted(set(files_a) & set(files_b))):
            a, b = handle(files_a[name]).get_tensor(name), handle(files_b[name]).get_tensor(name)
            if a.shape != b.shape:
                mismatched.append({"name": name, "shape_a": list(a.shape), "shape_b": list(b.shape)})
                continue
            stats = compare_tensor(a, b)
            group, layer = group_of(name)
            tensors.append(
                {"name": name, "group": group, "layer": layer, "shape": list(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype), **finish(stats)}
            )
            groups[group] = merge(groups.get(group), stats)
            overall = merge(overall, stats)
            if progress:
                progress(i, name)
    finally:
        handles.clear()

    group_rows = sorted(({"group": g, "n_tensors": s["n_tensors"], **finish(s)} for g, s in groups.items()), key=lambda r: -(r["relative_l2"] or 0.0))
    return {
        "a": str(folder_a),
        "b": str(folder_b),
        "n_tensors_compared": len(tensors),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "shape_mismatches": mismatched,
        "overall": finish(overall) if overall else None,
        "groups": group_rows,
        "tensors": tensors,
    }


def summary_text(result: dict) -> str:
    lines = [f"A: {result['a']}", f"B: {result['b']}", f"tensors compared: {result['n_tensors_compared']}"]
    for key, label in (("only_in_a", "only in A"), ("only_in_b", "only in B")):
        if result[key]:
            lines.append(f"{label}: {len(result[key])} (first: {', '.join(result[key][:3])})")
    if result["shape_mismatches"]:
        lines.append(f"shape mismatches: {len(result['shape_mismatches'])} (first: {result['shape_mismatches'][0]['name']})")
    total = result["overall"]
    if total:
        lines.append(
            f"overall: {total['n_changed']:,} of {total['n']:,} values differ ({total['fraction_changed']:.4%}); "
            f"max abs diff {total['max_abs_diff']:.6g}, mean abs diff {total['mean_abs_diff']:.6g}, relative L2 {total['relative_l2']:.6g}"
        )
    lines += ["", f"tensor groups with the largest relative L2 (top {TOP})", "group | tensors | fraction changed | mean abs diff | relative L2"]
    for row in result["groups"][:TOP]:
        lines.append(f"{row['group']} | {row['n_tensors']} | {row['fraction_changed']:.4%} | {row['mean_abs_diff']:.6g} | {row['relative_l2']:.6g}")
    unchanged = [r["group"] for r in result["groups"] if r["n_changed"] == 0]
    lines.append(f"groups with no differing value: {len(unchanged)} of {len(result['groups'])}")
    lines += ["", f"individual tensors with the largest relative L2 (top {TOP})", "tensor | fraction changed | relative L2"]
    for row in sorted(result["tensors"], key=lambda r: -(r["relative_l2"] or 0.0))[:TOP]:
        lines.append(f"{row['name']} | {row['fraction_changed']:.4%} | {row['relative_l2']:.6g}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="Reference checkpoint: a folder or a cached Hugging Face repo id")
    parser.add_argument("--b", required=True, help="Checkpoint compared with it")
    parser.add_argument("--out", default=None, help="Output folder (default: runs/results/weight_diff/<timestamp>)")
    args = parser.parse_args(argv)
    folder_a, folder_b = resolve(args.a), resolve(args.b)
    out = Path(args.out) if args.out else Path("runs") / "results" / "weight_diff" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    result = compare(folder_a, folder_b, progress=lambda i, name: print(f"{i + 1} {name}", file=sys.stderr) if i % 50 == 0 else None)
    result["a_spec"], result["b_spec"] = args.a, args.b
    (out / "weight_diff.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    text = summary_text(result)
    (out / "weight_diff.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")
    return result


if __name__ == "__main__":
    main()
