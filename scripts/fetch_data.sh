#!/usr/bin/env bash
set -euo pipefail

# Re-downloads the safety-gate and smartness-gate prompt sets.

mkdir -p data/en-text data/th-text

curl -sL -o data/en-text/harmbench_behaviors_text_all.csv \
  "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"

curl -sL -o data/th-text/thaisafetybench_test.parquet \
  "https://huggingface.co/datasets/typhoon-ai/ThaiSafetyBench/resolve/main/data/test-00000-of-00001.parquet"

echo "Fetched HarmBench text behaviors -> data/en-text/harmbench_behaviors_text_all.csv"
echo "Fetched ThaiSafetyBench test split -> data/th-text/thaisafetybench_test.parquet"

# XSTest (CC-BY-4.0, Rottger et al., NAACL 2024), pinned to a commit. Prompts feed the over-refusal case;
# the completions and string-match labels feed xstest_sanity_check.py only and carry the original
# model licences (Meta, Mistral, OpenAI), so they stay under the gitignored data/.
XSTEST_SHA="d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
XSTEST_RAW="https://raw.githubusercontent.com/paul-rottger/xstest/${XSTEST_SHA}"
mkdir -p data/over-refusal/completions data/over-refusal/streval

curl -sfL -o data/over-refusal/xstest_prompts.csv "${XSTEST_RAW}/xstest_prompts.csv"
for model in llama2orig llama2new mistralinstruct; do
  curl -sfL -o "data/over-refusal/completions/xstest_v2_completions_${model}.csv" \
    "${XSTEST_RAW}/model_completions/xstest_v2_completions_${model}.csv"
  curl -sfL -o "data/over-refusal/streval/xstest_v2_completions_${model}_streval.csv" \
    "${XSTEST_RAW}/evaluation/automated_evaluation_labels/xstest_v2_completions_${model}_streval.csv"
done

echo "Fetched XSTest prompts -> data/over-refusal/xstest_prompts.csv"
echo "Fetched XSTest completions + string-match labels -> data/over-refusal/{completions,streval}/"
