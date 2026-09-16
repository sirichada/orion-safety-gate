#!/usr/bin/env bash
set -euo pipefail

# Re-downloads the safety-gate prompt sets.

mkdir -p data/en-text data/th-text

curl -sL -o data/en-text/harmbench_behaviors_text_all.csv \
  "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"

curl -sL -o data/th-text/thaisafetybench_test.parquet \
  "https://huggingface.co/datasets/typhoon-ai/ThaiSafetyBench/resolve/main/data/test-00000-of-00001.parquet"

echo "Fetched HarmBench text behaviors -> data/en-text/harmbench_behaviors_text_all.csv"
echo "Fetched ThaiSafetyBench test split -> data/th-text/thaisafetybench_test.parquet"
