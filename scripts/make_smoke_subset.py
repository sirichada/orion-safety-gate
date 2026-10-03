#!/usr/bin/env python3
"""
Writes a deterministic random subset of a prompt-set JSONL, for small plumbing/smoke runs
that don't need the full set.

Usage:
    python make_smoke_subset.py --prompts ../data/prompt-sets/en_text.jsonl --n 10 \
        --out ../data/prompt-sets/en_text_api_smoke10.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="Source prompt-set JSONL")
    parser.add_argument("--n", type=int, required=True, help="Subset size")
    parser.add_argument("--seed", type=int, default=0, help="random.Random seed for the sample")
    parser.add_argument("--out", required=True, help="Path to write the subset JSONL to")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line]
    rng = random.Random(args.seed)
    subset = rng.sample(records, args.n)  # order is random.sample's own, not source-file order

    with open(args.out, "w", encoding="utf-8") as f:
        for record in subset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(subset)}/{len(records)} records (seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
