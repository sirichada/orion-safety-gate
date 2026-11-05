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

# ThaiExam (Apache-2.0, typhoon-ai/thai_exam v1.0), pinned to a revision and per-file sha256. Feeds the th_knowledge case.
THAI_EXAM_REV="d78aef04ea3cc5095545e6951cb39e17c64e26a1"
THAI_EXAM_DIR="data/th-knowledge"
THAI_EXAM_FILES="
a_level_test:284b6e504e08a7f454fef85e2428624ee2835acdcc086981cd18757f98e5b1d5
a_level_train:bdd9388b4fc5a89bf714f733e09327ebd80e9c3a7d9210877ae5b91cb3bfecd9
ic_test:211bc6f5e63578b34f5fcfc5191b925f16a5842408504ac2d865f38d06b71359
ic_train:7cdb5e6204012c8bd6e85cb0f5a559d9cc523089f6310e0c870c94e4bbcabbc4
onet_test:9e43f86192caec19a78c316688eb607dcdaae3dbd85fc3de0f67176f4c89fc65
onet_train:58dc4bc5d23ce2de94aebc7bb5b4c18b54ff2e8f349ccbc000634e6a03686403
tgat_test:1f4f48a7756abb57d7685c102a26f58d83f7e65cc9e8267585bc723c50105e9f
tgat_train:a276d4efd5209085724c22be9cba880bc254e7149f4184a5db373ca2a3ec3c31
tpat1_test:27bc3276b53360537ded2b6113a3fae532102836231e3b4e9da0c84bd7effc8e
tpat1_train:2b6a90f1c7585c6bada5919a4015624e6697aca65f33fc18a85732b714b7c172
"

file_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

mkdir -p "${THAI_EXAM_DIR}"
for entry in ${THAI_EXAM_FILES}; do
  name="${entry%%:*}"
  want="${entry##*:}"
  exam="${name%_*}"
  dest="${THAI_EXAM_DIR}/${name}.jsonl"
  curl -sfL -o "${dest}" \
    "https://huggingface.co/datasets/typhoon-ai/thai_exam/resolve/${THAI_EXAM_REV}/data/${exam}/${name}.jsonl"
  got="$(file_hash "${dest}")"
  if [ "${got}" != "${want}" ]; then
    echo "ERROR: ${dest} sha256 mismatch: expected ${want}, got ${got}" >&2
    exit 1
  fi
done
echo "Fetched ThaiExam (revision ${THAI_EXAM_REV}) -> ${THAI_EXAM_DIR}/"

# fastText language-ID model for the en_integrity scorer (lid.176.bin, CC-BY-SA 3.0 per the fastText language-identification page).
# While LID_SHA256 is TODO the file is fetched once and its hash printed. Only this step then exits non-zero,
# after everything above has been fetched.
LID_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_SHA256="7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
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
