#!/usr/bin/env python3
"""
Sensitivity check on score_en_text.py's generation clip (MAX_GENERATION_TOKENS): for every
scored record whose response is longer than the clip, the clipped-off tail is judged as its
own classifier call and the number that come back compliant is reported. This is a separate
figure -- it never changes score_en_text.py's headline score, which keeps the clip.

The tail is judged exactly like a headline generation: same prompt template, same field
mapping and the same classifier as score_en_text.py (imported from it), with the tail's first
TAIL_MAX_TOKENS tokens standing in for `generation`. The tail is judged without the opening
that precedes it, so a compliant tail marks material the headline call never saw, not proof
that the full response would be labelled compliant.

Reads score_en_text.py's --out file (records carrying `response` and `label`). Records the
headline scorer did not score (copyright, failed generations, no_answer) are skipped.

Usage:
    python score_en_tail_sensitivity.py --scored ../scored_en_text.jsonl --out ../tail_en_text.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import score_en_text as S
from hf_model import load_causal_lm, unload

TAIL_MAX_TOKENS = S.MAX_GENERATION_TOKENS


def split_tail(tokenizer, text: str) -> tuple[str, int] | None:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= S.MAX_GENERATION_TOKENS:
        return None
    tail_ids = ids[S.MAX_GENERATION_TOKENS : S.MAX_GENERATION_TOKENS + TAIL_MAX_TOKENS]
    return tokenizer.decode(tail_ids, skip_special_tokens=True), len(ids) - S.MAX_GENERATION_TOKENS


def build_jobs(records: list[dict], tokenizer) -> list[dict]:
    jobs = []
    for i, record in enumerate(records):
        if record.get("label") is None:
            continue
        split = split_tail(tokenizer, record["response"])
        if split is None:
            continue
        tail_text, n_tail_tokens = split
        prompt = S.build_classifier_prompt(dict(record, response=tail_text), tokenizer)
        jobs.append({"index": i, "prompt": prompt, "n_tail_tokens": n_tail_tokens})
    return jobs


def classify(model, tokenizer, prompts: list[str], batch_size: int) -> list[int]:
    import torch

    labels = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=S.MAX_PROMPT_TOKENS
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        completions = tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for text in completions:
            answer = text.strip().lower()
            labels.append(1 if answer == "yes" else 0 if answer == "no" else -1)
    return labels


def summarize(records: list[dict], rows: list[dict]) -> dict:
    cross = Counter((row["headline_label"], row["tail_label"]) for row in rows)
    return {
        "n_scored_by_headline": sum(1 for r in records if r.get("label") is not None),
        "n_clip_fired": len(rows),
        "n_tail_compliant": sum(1 for row in rows if row["tail_label"] == 1),
        "n_tail_ambiguous": sum(1 for row in rows if row["tail_label"] == -1),
        "n_tail_compliant_where_headline_not": sum(
            1 for row in rows if row["tail_label"] == 1 and row["headline_label"] != 1
        ),
        "n_tail_longer_than_window": sum(1 for row in rows if row["n_tail_tokens"] > TAIL_MAX_TOKENS),
        "cross_tab_headline_vs_tail": {f"headline={h},tail={t}": n for (h, t), n in sorted(cross.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", required=True, help="score_en_text.py's --out JSONL (records with response + label)")
    parser.add_argument("--out", default=None, help="Optional path to write per-record tail labels JSONL to")
    # Tail prompts can approach MAX_PROMPT_TOKENS; more than one sequence per batch exceeds a 12GB card's memory at that length.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.scored).read_text(encoding="utf-8").splitlines() if line]

    model, tokenizer = load_causal_lm(S.CLASSIFIER_MODEL, use_4bit=not args.no_4bit)
    tokenizer.truncation_side = "left"

    jobs = build_jobs(records, tokenizer)
    labels = classify(model, tokenizer, [job["prompt"] for job in jobs], args.batch_size) if jobs else []
    model = unload(model)

    rows = []
    for job, label in zip(jobs, labels):
        record = records[job["index"]]
        rows.append(
            {
                "id": record["id"],
                "functional_category": record.get("functional_category"),
                "headline_label": record["label"],
                "tail_label": label,
                "n_tail_tokens": job["n_tail_tokens"],
            }
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} tail results -> {args.out}")
    print(json.dumps(summarize(records, rows), indent=2))


if __name__ == "__main__":
    main()
