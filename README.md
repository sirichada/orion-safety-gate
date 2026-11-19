# Orion Safety Gate

A pre-publish admission check for models hosted on Orion. An admin runs it against a candidate model before registering it, and gets one verdict back.

The suite tests the raw model, with no guardrail in front of it, through two gates:

- **Safety gate**: does the model comply with harmful requests?
- **Smartness gate**: is the model still useful (not over-refusing, replying in well-formed English and Thai, answering Thai knowledge questions)?

A model is **admitted only when both gates pass.** Scope is English and Thai, text only.

## Cases

| Case | Gate | Benchmark | Score (direction) |
|---|---|---|---|
| `en_text` | safety | HarmBench text behaviours, scored by `cais/HarmBench-Llama-2-13b-cls` | harmful rate (lower is better) |
| `th_text` | safety | ThaiSafetyBench test split, scored by `typhoon-ai/ThaiSafetyClassifier` | harmful rate (lower is better) |
| `over_refusal` | smartness | XSTest safe prompts | refusal rate (lower is better) |
| `en_integrity` | smartness | Re-checks the English replies: script, language ID, loops, empty and length checks; the 50 lyrics-tagged HarmBench replies are not scored | pass rate (higher is better) |
| `th_integrity` | smartness | The same checks on the Thai replies | pass rate (higher is better) |
| `th_knowledge` | smartness | ThaiExam multiple choice, exact match | accuracy (higher is better) |

Only the cases that apply to the model's declared languages run. Thai cases run only when Thai is declared.

Each case ends as `pass`, `fail` or `insufficient_data`. A case is `insufficient_data` when too few replies could be scored. A gate is `fail` if any of its cases fail, otherwise `insufficient_data` if any case is insufficient, otherwise `pass`. A gate with no applicable case is `not_testable`.

Tool calling and coding are not tested and are not cases, so they never affect a gate. Every verdict lists them under `declared_scope.not_tested`, so the omission stays visible.

> **Thresholds are provisional.** `config/thresholds.json` carries `"provisional": true`. Several pass marks are placeholders, and the `th_integrity` and `th_knowledge` marks are 0, so those two always pass for now. Verdicts produced with this file are not final.

## Requirements

- Python 3.11 or newer.
- An NVIDIA GPU with about 12 GB of memory for scoring. The HarmBench classifier is a 13B model and runs in 4-bit. The candidate model itself runs on Orion and needs no local GPU.
- About 30 GB of disk for the scorer models in the Hugging Face cache.
- An Orion API key, for API runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/fetch_data.sh         # benchmark data and the language-ID model, into data/
```

`fasttext` has no Windows wheel. On Windows, build it from source with `CL="/std:c++17 /Dssize_t=ptrdiff_t"` set before `pip install`.

`fetch_data.sh` pins XSTest, ThaiExam and the fastText model to fixed revisions and checks their hashes. It stops with an error if a download does not match.

Create a `.env` file in the repo root, or export the variable in your shell:

```bash
ORION_API_KEY=<your key>
```

The key is only ever sent in the `Authorization` header. `.env` is git-ignored. Never commit it.

## Running the suite

Check the plan and worst-case spend first. This needs no key and makes no requests:

```bash
python scripts/safety_suite.py run --backend api --model <orion-model-id> \
  --model-id "<stable name>" --languages en th --modalities text \
  --max-thb 60 --dry-run
```

Then run it for real by dropping `--dry-run`:

```bash
python scripts/safety_suite.py run --backend api --model <orion-model-id> \
  --model-id "<stable name>" --languages en th --modalities text \
  --max-thb 60
```

In a terminal, any of `--model-id`, `--languages` or `--modalities` left out is asked for, and a summary is shown for confirmation before anything runs. Use `--yes` to skip the confirmation and `--no-input` to never prompt.

Points to know before a real run:

- **API runs require a spend limit** (`--max-thb`). The run stops before it can exceed it. The gateway bills each request up to its token cap even when the stream is closed early, so the guard counts the worst case.
- **Runs are long.** A full run takes from about 7 hours to about 40 hours, depending on the model, most of it in `th_text`. Orion allows at most three requests in flight per key, and nothing else can use the key meanwhile.
- **Interrupted runs can be resumed** with the same settings: `--resume runs/results/<run folder>`.
- **Sampling follows each model's protocol** in `config/generation.json`. A model with no entry there is refused. Passing a sampling flag that differs from the protocol makes the run non-admitting.

Partial runs, for testing:

```bash
--only over_refusal,en_text                 # run only these cases
--sample-per-case 50 --sample-seed 1        # a seeded sample of each prompt set
```

Both are marked in the verdict and are never an admission.

A local Hugging Face model can be tested with `--backend hf --model <repo or path>`, optionally with `--adapter <PEFT adapter>` and `--precision {4bit,full,as_published}`.

## Output

Each run writes a folder `runs/results/<model-id>_<timestamp>/` containing:

- `responses_<case>.jsonl`: the model's replies, with reasoning traces split into a separate file;
- `scored_<case>.jsonl`: the per-reply scores;
- `spend_ledger.json`: what the run spent;
- `verdict.json`: the admission verdict.

`verdict.json` records the declared scope, the cases run and not attempted (with reasons), both gate values, `admitted`, any protocol deviations, and per case the score, threshold, number scored and number excluded.

Every run, including one that ends in an error, appends a row to `runs/_index.md`, and each stage logs to `runs/logs/`. `runs/` and `data/` are git-ignored.

## Re-scoring a recorded run

`python scripts/safety_suite.py rescore runs/results/<run folder> [--thresholds config/thresholds.json]` re-derives a run's verdict from its folder in seconds, with no model, no API key and no network. It reads the recorded `scored_<case>.jsonl` files, re-aggregates them and applies the thresholds file, which is how verdicts are re-derived once the provisional pass marks are replaced.

The declared scope, model id, run scope and protocol deviations come from the run's `verdict.json`, so a re-scored partial or sampled run is still never an admission. A recorded `not_attempted` entry for tool calling, from before it was taken out of the smartness gate, is dropped when re-scoring, and the ids dropped are listed in the verdict's `rescore.dropped_not_attempted`; every other recorded entry is kept as recorded. The result is written next to the original as `verdict.rescore-<timestamp>.json` (or to `--out`), marked `"rescored": true` with the source folder, the thresholds file and its hash, and the previous gates and case statuses, and the command prints what changed against the recorded verdict. The original `verdict.json` is never touched, and an existing output file is never replaced without `--force`.

A thresholds file that sets an integrity case's scorer settings (`line_rule`, `share_threshold` and the Thai settings) differently from the run is refused, because those settings shaped the scored files. `--rerun-scorers` scores `responses_<case>.jsonl` again into `rescore-<timestamp>/` in the run folder instead, which needs the scorers' dependencies and a GPU for the `en_text` and `th_text` classifiers.

## Configuration

- `config/thresholds.json`: the pass mark, minimum scored count and maximum excluded share per case.
- `config/generation.json`: token caps per case and per model, and each API model's sampling protocol.

## Tests

```bash
pytest tests
```

## Known limitations

- **Lyrics are left out of `en_integrity`.** The 50 HarmBench behaviours tagged `lyrics` are not scored there, because song lyrics repeat by design and the loop checks would flag a chorus as a broken reply. They are not counted as prompts or as exclusions, and `en_text` is unchanged. Re-scoring a run that did not apply this needs `--rerun-scorers`.
- **The Thai harmful verdict is imprecise.** On a hand-labelled sample of 40 replies, only about a quarter of the replies the Thai classifier flagged as harmful were judged harmful. Treat `th_text` as a signal for human review, not a verdict on its own.
- **Not tested:** coding, tool calling, images, and languages other than English and Thai.
- **Sampling.** Only each model's default thinking mode and recommended sampling are tested. Other settings may behave differently.
