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

# fastText language-ID model for the en_integrity scorer (lid.176.bin, CC-BY-SA 3.0 per the fastText language-identification page).
# While LID_SHA256 is TODO the file is fetched once and its hash printed. Only this step then exits non-zero,
# after everything above has been fetched.
LID_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_SHA256="TODO"
LID_FILE="data/lid/lid.176.bin"

lid_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${LID_FILE}" | cut -d' ' -f1
  else
    shasum -a 256 "${LID_FILE}" | cut -d' ' -f1
  fi
}

echo "All other data fetched; fetching the language-ID model."
mkdir -p data/lid
if [ -f "${LID_FILE}" ] && { [ "${LID_SHA256}" = "TODO" ] || [ "$(lid_hash)" = "${LID_SHA256}" ]; }; then
  echo "Reusing ${LID_FILE}"
else
  curl -sfL -o "${LID_FILE}" "${LID_URL}"
fi
LID_ACTUAL="$(lid_hash)"
if [ "${LID_SHA256}" = "TODO" ]; then
  echo "WARNING: ${LID_FILE} is not pinned. Its sha256 is ${LID_ACTUAL}; set LID_SHA256 in scripts/fetch_data.sh to this value." >&2
  exit 1
fi
if [ "${LID_ACTUAL}" != "${LID_SHA256}" ]; then
  echo "ERROR: ${LID_FILE} sha256 mismatch: expected ${LID_SHA256}, got ${LID_ACTUAL}" >&2
  exit 1
fi
echo "Fetched fastText language-ID model -> ${LID_FILE}"
