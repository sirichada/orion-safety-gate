#!/usr/bin/env python3
"""
Measures how much of a Thai reply ThaiSafetyClassifier actually sees, and whether the part it
cuts changes the verdict. Local only: it never calls the API.

The scorer (scripts/score_th_text.py) classifies f"input: {prompt} output: {response}" cut at 256
tokens, keeping the head. This tool re-scores the same replies under four settings:

    S0  the scorer's own setting: head, 256 tokens (must reproduce the stored labels)
    S1  head, 512 tokens (the checkpoint's max_position_embeddings; past its 256-token training length)
    S2  the whole reply in 256-token windows, each led by the prompt; harmful if any window is harmful
    S3  the prompt plus the last 256-token window of the reply

By default the input is classified as is (no normalization), which reproduces the labels stored by
runs scored before the scorer normalized its input; --normalize NFKC applies NFKC to the prompt and
the reply before every setting.

S2 is the main comparison: every window stays at the classifier's training length. The first window
of S2 is reported against S0 as a control for the tokenization difference (prompt and reply are
tokenized separately in S2/S3).

Records come from the scored_th_text.jsonl files of finished suite runs (stored labels, so S0 is
checked) and from the finished replies in calibration folders (no stored labels).

Usage (from the repo root, with the repo's .venv python):
    python tools/check_th_truncation.py --out-dir runs/results/th_truncation_check/<ts>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hf_model import load_sequence_classifier, unload  # noqa: E402
from score_th_text import CLASSIFIER_MODEL, MAX_LENGTH  # noqa: E402

MODELS = {
    "Mango 1.5-9B": "Mango_1.5-9B",
    "Qwen 3.8-27B": "Qwen_3.8-27B",
    "Qwen 3.8-Flash-Next": "Qwen_3.8-Flash-Next",
}
# Calibration folders: first runs, then the reruns of their cut-off prompts.
CALIBRATION = {
    "Mango 1.5-9B": ["20261027-103656"],
    "Qwen 3.8-27B": ["20261027-105504", "20261027-115536"],
    "Qwen 3.8-Flash-Next": ["20261027-111400", "20261027-125322"],
}
MAX_PROMPT_TOKENS = 128  # keeps at least ~126 reply tokens in every window
SETTINGS = ("S0", "S1", "S2", "S3")


def prepare(text: str, normalize: str) -> str:
    return unicodedata.normalize(normalize, text) if normalize != "none" else text


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_records():
    """One dict per classifiable reply: model, group, id, prompt, response, stored label (or None)."""
    out = []
    for model, prefix in MODELS.items():
        for path in sorted(glob.glob(str(REPO_ROOT / "runs" / "results" / f"{prefix}_2026*" / "scored_th_text.jsonl"))):
            run = os.path.basename(os.path.dirname(path))
            for r in read_jsonl(path):
                if r.get("excluded"):
                    continue
                out.append(dict(model=model, group="suite_run", run=run, id=r["id"], prompt=r["prompt"],
                                response=r["response"], stored=r["label"]))
    prompts = {r["id"]: r["prompt"] for r in read_jsonl(REPO_ROOT / "data" / "prompt-sets" / "th_text.jsonl")}
    for model, folders in CALIBRATION.items():
        for folder in folders:
            path = REPO_ROOT / "runs" / "results" / "calibration" / folder / "raw_th_text.jsonl"
            if not path.exists():
                continue
            for r in read_jsonl(path):
                if r.get("finish_reason") == "length" or r.get("failed"):
                    continue
                out.append(dict(model=model, group="calibration", run=folder, id=r["id"],
                                prompt=prompts[r["id"].split("__")[0]], response=r["response"], stored=None))
    return out


def window_ids(tok, prompt: str, response: str, normalize: str = "none"):
    """Token-id lists for S2 (all windows) and S3 (last window), plus the reply's token count."""
    cls, sep = tok.cls_token_id, tok.sep_token_id
    p = tok(prepare(f"input: {prompt} output:", normalize), add_special_tokens=False)["input_ids"][:MAX_PROMPT_TOKENS]
    r = tok(prepare(response, normalize), add_special_tokens=False)["input_ids"]
    budget = MAX_LENGTH - len(p) - 2
    chunks = [r[i : i + budget] for i in range(0, len(r), budget)] or [[]]
    wins = [[cls] + p + c + [sep] for c in chunks]
    tail = [cls] + p + (r[-budget:] if r else []) + [sep]
    return wins, tail, len(r), len(p), budget


def classify(model, tok, id_lists, batch_size=16):
    import torch

    labels = []
    for i in range(0, len(id_lists), batch_size):
        batch = id_lists[i : i + batch_size]
        width = max(len(x) for x in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, x in enumerate(batch):
            ids[j, : len(x)] = torch.tensor(x)
            mask[j, : len(x)] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        labels.extend(logits.argmax(dim=-1).tolist())
        del ids, mask, logits  # per-batch tensors are released, as in the scorers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return labels


def head_labels(model, tok, texts, max_length, batch_size=8):
    import torch

    labels = []
    for i in range(0, len(texts), batch_size):
        inputs = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        labels.extend(logits.argmax(dim=-1).tolist())
        del inputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return labels


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None


def summarise(rows):
    n = len(rows)
    over = sum(1 for r in rows if r["full_tokens"] > MAX_LENGTH)
    surv = [r["resp_tokens_seen_s0"] for r in rows]
    frac = [r["resp_tokens_seen_s0"] / r["resp_tokens"] for r in rows if r["resp_tokens"]]
    s = dict(n=n, over_256=over, share_over_256=round(over / n, 3) if n else None,
             full_tokens_median=statistics.median(r["full_tokens"] for r in rows) if n else None,
             full_tokens_p90=pct([r["full_tokens"] for r in rows], 0.9),
             full_tokens_max=max((r["full_tokens"] for r in rows), default=None),
             reply_tokens_seen_by_s0_median=statistics.median(surv) if n else None,
             reply_tokens_seen_by_s0_p10=pct(surv, 0.1),
             share_of_reply_seen_by_s0_median=round(statistics.median(frac), 3) if frac else None,
             windows_median=statistics.median(r["n_windows"] for r in rows) if n else None,
             windows_max=max((r["n_windows"] for r in rows), default=None))
    for st in SETTINGS:
        s[f"harmful_rate_{st}"] = round(sum(r[st] for r in rows) / n, 3) if n else None
    for st in ("S1", "S2", "S3", "S2_first_window"):
        up = sum(1 for r in rows if r["S0"] == 0 and r[st] == 1)
        down = sum(1 for r in rows if r["S0"] == 1 and r[st] == 0)
        s[f"flips_{st}_vs_S0"] = dict(safe_to_harmful=up, harmful_to_safe=down, total=up + down)
    stored = [r for r in rows if r["stored"] is not None]
    s["s0_vs_stored_label_mismatches"] = f"{sum(1 for r in stored if r['S0'] != r['stored'])}/{len(stored)}" if stored else None
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--normalize", choices=["none", "NFKC"], default="none",
                        help="Unicode normalization applied before tokenizing (default: none)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N records (for a quick plumbing check)")
    args = parser.parse_args()

    records = collect_records()
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} replies: " + ", ".join(
        f"{m} {sum(1 for r in records if r['model'] == m and r['group'] == g)} {g}" for m in MODELS for g in ("suite_run", "calibration")))

    model, tok = load_sequence_classifier(CLASSIFIER_MODEL)
    try:
        texts = [prepare(f"input: {r['prompt']} output: {r['response']}", args.normalize) for r in records]
        full_len = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]

        all_windows, spans, tails = [], [], []
        for r in records:
            wins, tail, n_resp, n_prompt, budget = window_ids(tok, r["prompt"], r["response"], args.normalize)
            spans.append((len(all_windows), len(wins)))
            all_windows.extend(wins)
            tails.append(tail)
            r["resp_tokens"], r["prompt_tokens"], r["n_windows"] = n_resp, n_prompt, len(wins)
            r["resp_tokens_seen_s0"] = min(n_resp, max(0, MAX_LENGTH - len(tok(prepare(f"input: {r['prompt']} output:", args.normalize))["input_ids"])))

        s0 = head_labels(model, tok, texts, MAX_LENGTH)
        s1 = head_labels(model, tok, texts, 512)
        win_labels = classify(model, tok, all_windows)
        s3 = classify(model, tok, tails)
        for i, r in enumerate(records):
            start, n = spans[i]
            w = win_labels[start : start + n]
            r.update(full_tokens=full_len[i], S0=s0[i], S1=s1[i], S2=int(any(w)), S2_first_window=w[0], S3=s3[i],
                     windows_harmful=sum(w))
    finally:
        model = unload(model)

    summary = {}
    for m in MODELS:
        for g in ("suite_run", "calibration"):
            rows = [r for r in records if r["model"] == m and r["group"] == g]
            if rows:
                summary[f"{m} | {g}"] = summarise(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("prompt", "response")}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
