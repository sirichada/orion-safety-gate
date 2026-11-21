#!/usr/bin/env python3
"""
Runs model_wrapper.py in fixed-size chunks against a single prompt file, so a
driver reset (TDR) that kills the whole model_wrapper.py process only loses
the current chunk instead of the entire run.

Test-running tooling for an unreliable GPU on this machine -- not part of the
safety-gate pipeline itself, which is deliberately not resumable by design.
model_wrapper.py is invoked as a separate, unmodified subprocess per chunk;
this script never imports or edits it.

Usage:
    python run_chunked.py --model meta-llama/Llama-2-7b-chat-hf \\
        --prompts ../data/prompt-sets/en_text.jsonl --out ../responses_en_plain.jsonl

Any extra arguments (e.g. --adapter, --batch-size) are forwarded to
model_wrapper.py unchanged for every chunk.

Rerun the same command to resume: a chunk is only considered done once it has
a matching marker file (written after a clean exit AND an exact line-count
match), so a chunk left partial by model_wrapper.py's own failure-rate abort,
a chunk truncated mid-write by a driver reset, or a chunk produced by a
different --model/--adapter/extra-args combination are all rerun rather than
silently accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CHUNK_SIZE = 80


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def chunk_ranges(n: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def line_count_matches(out_path: Path, expected_count: int) -> bool:
    if not out_path.exists():
        return False
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines) == expected_count


def marker_signature(model: str, extra_args: list[str], expected_count: int, chunk_in: Path) -> dict:
    content_hash = hashlib.sha256(chunk_in.read_bytes()).hexdigest()
    return {"model": model, "extra_args": extra_args, "count": expected_count, "input_sha256": content_hash}


def is_chunk_done(marker_path: Path, out_path: Path, signature: dict) -> bool:
    # A marker is only written after a clean subprocess exit AND a verified
    # line-count match (see main()) -- its mere existence, plus a signature
    # match, is what "done" means. Re-checking the line count here as well
    # guards against the file having been corrupted or truncated (e.g. by a
    # driver reset landing mid-write on some *other* process) since the
    # marker was written.
    if not marker_path.exists():
        return False
    try:
        recorded = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if recorded != signature:
        return False
    return line_count_matches(out_path, signature["count"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True, help="Final joined output path")
    parser.add_argument(
        "--chunk-dir", default=None, help="Directory for per-chunk files (default: <out-stem>_chunks alongside --out)"
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args, extra_args = parser.parse_known_args()

    prompts_path = Path(args.prompts)
    out_path = Path(args.out)
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else out_path.parent / f"{out_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(prompts_path)
    ranges = chunk_ranges(len(records), args.chunk_size)

    for idx, (start, end) in enumerate(ranges):
        chunk_records = records[start:end]
        chunk_in = chunk_dir / f"chunk_{idx:02d}_in.jsonl"
        chunk_out = chunk_dir / f"chunk_{idx:02d}_out.jsonl"
        chunk_marker = chunk_dir / f"chunk_{idx:02d}.done"

        with open(chunk_in, "w", encoding="utf-8") as f:
            for r in chunk_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        signature = marker_signature(args.model, extra_args, len(chunk_records), chunk_in)

        if is_chunk_done(chunk_marker, chunk_out, signature):
            print(f"chunk {idx}: already complete ({len(chunk_records)} records) -- skipping")
            continue

        # Remove any stale marker before rerunning so the chunk dir never has
        # a marker that doesn't correspond to the current chunk_out on disk.
        chunk_marker.unlink(missing_ok=True)

        print(f"chunk {idx}: running model_wrapper.py on records {start}-{end} ({len(chunk_records)} prompts)")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "scripts" / "model_wrapper.py"),
            "--model", args.model,
            "--prompts", str(chunk_in),
            "--out", str(chunk_out),
            *extra_args,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0 or not line_count_matches(chunk_out, len(chunk_records)):
            print(f"chunk {idx}: failed or incomplete (exit code {result.returncode}) -- stopping; rerun this script to retry")
            sys.exit(1)

        # Only a clean exit + exact line count earns the marker -- an abort
        # that happens to still produce the full record count (e.g. failing
        # on the very last batch) never reaches this line, since it exits
        # non-zero above.
        chunk_marker.write_text(json.dumps(signature), encoding="utf-8")

    joined: list[dict] = []
    for idx, (start, end) in enumerate(ranges):
        chunk_out = chunk_dir / f"chunk_{idx:02d}_out.jsonl"
        joined.extend(load_records(chunk_out))

    expected_ids = [r["id"] for r in records]
    joined_ids = [r["id"] for r in joined]

    if len(joined) != len(records):
        raise RuntimeError(f"Joined record count {len(joined)} != prompt file record count {len(records)}")
    if len(set(joined_ids)) != len(joined_ids):
        raise RuntimeError("Joined output contains duplicate ids")
    if set(joined_ids) != set(expected_ids):
        raise RuntimeError("Joined output ids don't match the prompt file's ids")
    if joined_ids != expected_ids:
        raise RuntimeError("Joined output order doesn't match the prompt file's order")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(joined)} joined responses -> {out_path}")


if __name__ == "__main__":
    main()
