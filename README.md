# Orion Safety Gate

A pre-publish admission check for models hosted on Orion. An admin runs it against a candidate model before registering it, and gets one verdict back.

The suite tests the raw model, with no guardrail in front of it, through two gates:

- **Safety gate**: does the model comply with harmful requests?
- **Smartness gate**: is the model still useful (not over-refusing, replying in well-formed English and Thai, answering Thai knowledge questions)?

A model is **admitted only when both gates pass.** Scope is English and Thai, text only. The result is one of five states: **Not admitted** (a case fails), **Not decided** (nothing fails but a case lacks data, a gate could not be tested, or the run was partial or off protocol), **Review needed** (only a person's review of the Thai harmful result is pending), **Admitted**, or **Admitted after review**.

**Contents:** [Cases](#cases) · [How a run works](#how-a-run-works) · [Repository layout](#repository-layout) · [Requirements](#requirements) · [Setup](#setup) · [Running the suite](#running-the-suite) · [Output](#output) · [Re-scoring a recorded run](#re-scoring-a-recorded-run) · [Admission report and review](#admission-report-and-review) · [Configuration](#configuration) · [Data and model licences](#data-and-model-licences) · [Tests](#tests) · [Known limitations](#known-limitations)

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

Each case ends as `pass`, `fail`, `insufficient_data` or `needs_review`. A case is `insufficient_data` when too few replies could be scored or too many failed to generate. `needs_review` applies only to `th_text`: above its mark the case asks for a person to review a sample instead of failing, because the Thai classifier is unreliable. After a review the case is `pass_after_review` (accepted) or `fail` (rejected). A gate is `fail` if any of its cases fail, otherwise `insufficient_data` if any case is insufficient, otherwise `needs_review` if one is pending, otherwise `pass`. A gate with no applicable case is `not_testable`. A pending review is never an admission: `admitted` stays false until a review is applied.

Tool calling and coding are not tested and are not cases, so they never affect a gate. Every verdict lists them under `declared_scope.not_tested`, so the omission stays visible.

**Pass marks.** Each mark in `config/thresholds.json` is the weakest model Orion already serves, plus or minus √2 times the 95% Wilson half-width of its score (`en_integrity`'s 0.95 is fixed from hand labels), so a pass means the model is no worse than what Orion already serves. The 100-reply minimum and the maximum excluded share are proposals.

## How a run works

Each case goes through the same stages. Every stage is its own script, and stages pass their results on as JSONL files, not in memory.

1. **Fetch** (`scripts/fetch_data.sh`): downloads the benchmark data into `data/`.
2. **Load** (`scripts/load_*.py`): converts each benchmark into the common prompt-set format (`id`, `native_id`, `prompt`, `language`, `source_benchmark`, `native_category`, `functional_category`, `context`, `model_input`) in `data/prompt-sets/`.
3. **Generate** (`scripts/model_wrapper.py`): sends each `model_input` to the candidate model, locally (`--backend hf`) or through the API (`--backend api`), and writes `responses_<case>.jsonl`.
4. **Score** (`scripts/score_*.py`): scores the replies with the case's own scorer and writes `scored_<case>.jsonl`. `en_integrity` and `th_integrity` re-check replies already generated for `en_text` and `th_text`, so they need no generation of their own.
5. **Verdict** (`scripts/verdict.py`): compares each case's score with its pass mark in `config/thresholds.json`, combines the cases into the two gates and writes `verdict.json`.
6. **Report** (`scripts/report.py`): renders `report.html` from the verdict.

`scripts/safety_suite.py` runs stages 3 to 6 for the cases that apply to the declared languages and modalities. `rescore` repeats stages 5 and 6 from the recorded scored files.

## Repository layout

`scripts/` is the pipeline a run goes through; `tools/` holds standalone tools that a run never calls (each runs on its own from the repo root).

| Folder | Contents |
|---|---|
| `scripts/` | The pipeline. `fetch_data.sh` and `load_*.py` build the prompt sets; `model_wrapper.py` (with `hf_model.py` and `api_model.py`) generates replies; `score_*.py` and `integrity_checks.py` score them; `safety_suite.py` is the command line, with `orchestrator.py`, `cases.py`, `capabilities.py`, `run_config.py` and `cli_input.py` planning and running the cases; `verdict.py`, `rescore.py`, `review.py`, `report.py` and `case_summary.py` produce and re-derive the verdict and its report. |
| `tools/` | Standalone tools, one per line below. |
| `config/` | Pass marks (`thresholds.json`), token caps and sampling protocols (`generation.json`), and the prices template. |
| `templates/` | The report page template and its wording. |
| `tests/` | The offline test suite. |
| `progress/` | Fortnightly progress reports. |
| `surveys/` | The hazard-ranking survey tally; not part of a run. |

Tools in `tools/`:

- `calibrate_caps.py`: samples prompts per case to measure real reply lengths and propose per-case token caps and a cost estimate for an API model.
- `probe_thinking.py`: measures how Orion's API models return thinking, token use and default sampling on coding prompts.
- `reply_lengths.py`: reply lengths per model and case, read from saved responses.
- `check_th_truncation.py`: measures how much of a Thai reply the Thai classifier sees and whether the cut changes the verdict.
- `sensitivity_integrity.py`: sensitivity grid for the integrity cases' settings, from hand-labelled review files.
- `score_en_tail_sensitivity.py`: judges the part of long replies that the English classifier's clip cuts off.
- `xstest_sanity_check.py`: compares the over-refusal matcher with XSTest's published completions and human labels.
- `compare_refusal_scorer.py`: compares the over-refusal matcher with a WildGuard judge.
- `score_wildguardtest.py`: scores WildGuardTest with the HarmBench classifier and compares it with the dataset's human labels.
- `weight_diff.py`: compares two checkpoints tensor by tensor without loading either.
- `run_chunked.py`: runs `model_wrapper.py` in fixed-size chunks so a crash loses only one chunk.
- `make_smoke_subset.py`: writes a seeded random subset of a prompt set for small test runs.
- `make_triggered_en_text.py`: inserts a backdoor trigger word into the English prompts, for adapter tests.

## Requirements

- Python 3.11 or newer.
- An NVIDIA GPU with about 12 GB of memory for scoring. The HarmBench classifier is a 13B model and runs in 4-bit. For API runs the candidate model runs on Orion and needs no local GPU; a local `--backend hf` run loads the candidate model on the same GPU.
- About 30 GB of disk for the scorer models in the Hugging Face cache.
- An Orion API key, for API runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/fetch_data.sh         # benchmark data and the language-ID model, into data/
```

The commands in this README assume the virtual environment is active, so `python` is its interpreter. If `python` resolves to another install, call the environment's own interpreter by path (`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows).

`fasttext` has no Windows wheel. On Windows, build it from source with `CL="/std:c++17 /Dssize_t=ptrdiff_t"` set before `pip install`.

`fetch_data.sh` pins XSTest, ThaiExam and the fastText model to fixed revisions and checks their hashes. It stops with an error if a download does not match.

Create a `.env` file in the repo root, or export the variable in your shell:

```bash
ORION_API_KEY=<your key>
ORION_BASE_URL=<the API address, ending in /v1>
```

The key is only ever sent in the `Authorization` header, and run reports record the address as `(set by ORION_BASE_URL)`. `.env` is git-ignored. Never commit it.

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

### Local models

A local Hugging Face model is tested with `--backend hf --model <repo or path>`, optionally with `--adapter <PEFT adapter>`, `--precision {4bit,full,as_published}` and `--batch-size`. It needs the GPU described under Requirements; the model loads in 4-bit by default, and the scorers load on the same card after generation. `--no-thinking` renders the model's chat template with thinking turned off, for models whose template supports it.

One case, English only (the over-refusal case, about 250 prompts, is the cheapest):

```bash
python scripts/safety_suite.py run --model Qwen/Qwen3-0.6B --model-id Qwen3-0.6B \
  --backend hf --languages en --modalities text --only over_refusal --no-thinking --yes --no-input
```

A small seeded sample of every case (never an admission):

```bash
python scripts/safety_suite.py run --model Qwen/Qwen3-0.6B --model-id Qwen3-0.6B \
  --backend hf --languages en th --modalities text --no-thinking --batch-size 4 \
  --sample-per-case 20 --sample-seed 1 --yes --no-input
```

A full run is the same command without `--only` and the sample flags. Measured on a 12 GB card, a full run takes about 3.5 hours for a 0.6B model and 4.5 to 5 hours for an 8B model. The `en_text` scorer loads the 13B HarmBench classifier (about 8 GB in 4-bit) after generation, so the GPU should be otherwise idle.

## Output

Each run writes a folder `runs/results/<model-id>_<timestamp>/` containing:

- `responses_<case>.jsonl`: the model's replies, with reasoning traces split into a separate file;
- `scored_<case>.jsonl`: the per-reply scores;
- `spend_ledger.json`: what the run spent;
- `verdict.json`: the admission verdict;
- `report.html`: the admission report, a static page written from the verdict (see below).

`verdict.json` records the declared scope, the cases run and not attempted (with reasons), both gate values, `admitted`, `verdict_state`, any protocol deviations, the run's start and finish times, and per case the score, threshold, number scored and number excluded. Each scored case also carries a per-category summary, the served models' scores it is compared with, and seeded random sample replies (flagged and not flagged; for a `needs_review` case, 30 flagged replies in full). The samples are drawn with a seed fixed to the model id and case, so a re-score shows the same replies.

Every run, including one that ends in an error, appends a row to `runs/_index.md`, and each stage logs to `runs/logs/`. `runs/` and `data/` are git-ignored.

## Re-scoring a recorded run

`python scripts/safety_suite.py rescore runs/results/<run folder> [--thresholds config/thresholds.json]` re-derives a run's verdict from its folder in seconds, with no model, no API key and no network. It reads the recorded `scored_<case>.jsonl` files, re-aggregates them and applies the thresholds file, which is how verdicts are re-derived once the provisional pass marks are replaced.

The declared scope, model id, run scope and protocol deviations come from the run's `verdict.json`, so a re-scored partial or sampled run is still never an admission. A recorded `not_attempted` entry for tool calling, from before it was taken out of the smartness gate, is dropped when re-scoring, and the ids dropped are listed in the verdict's `rescore.dropped_not_attempted`; every other recorded entry is kept as recorded. The result is written next to the original as `verdict.rescore-<timestamp>.json` (or to `--out`), marked `"rescored": true` with the source folder, the thresholds file and its hash, and the previous gates and case statuses, and the command prints what changed against the recorded verdict. The original `verdict.json` is never touched, and an existing output file is never replaced without `--force`.

A thresholds file that sets an integrity case's scorer settings (`line_rule`, `share_threshold` and the Thai settings) differently from the run is refused, because those settings shaped the scored files. `--rerun-scorers` scores `responses_<case>.jsonl` again into `rescore-<timestamp>/` in the run folder instead, which needs the scorers' dependencies and a GPU for the `en_text` and `th_text` classifiers.

## Admission report and review

`python scripts/safety_suite.py report runs/results/<run folder> [--verdict <file>]` writes the report next to the verdict it reads: `report.html` for `verdict.json`, `report.rescore-<stamp>.html` for a re-scored verdict, `report.reviewed-<stamp>.html` for a reviewed one. It also runs at the end of `run`, `rescore` and `review`, and a failure while rendering never fails the run. A report is never overwritten.

The page shows the verdict state, the two gates, one row per case (score, pass mark, a bar on a 0-1 scale with the served models' range, category table and sample replies), what was not tested and the run settings. It reads the stored statuses, gates and `admitted` and recomputes nothing, so the verdict file stays authoritative. All wording is in `templates/report.html.tmpl`, and every value from the verdict or the replies is HTML-escaped. It is one self-contained file and reads offline.

Reading the report, top to bottom:

- **Verdict state**: one of the five states described at the top of this README.
- **The two gates**: safety and smartness, each `pass`, `fail`, `insufficient_data`, `needs_review` or `not_testable`.
- **One row per case**: the score next to its pass mark (and whether lower or higher is better), a bar on a 0-1 scale showing where the score falls against the range the served models scored, and the facts behind it: prompts sent, what scored them, replies left out (failed to generate, empty, or by design), cost on API runs, and notes.
- **Category table and sample replies**: the per-category rates, and seeded random replies that were flagged and not flagged, so a reader can judge what the scorer counted.
- **Not tested**: what the run did not cover, such as tool calling and coding.
- **Run settings**: the model and where it ran, sampling, token cap, prompts sent, spend on API runs, duration, and any deviation from the model's sampling protocol, so the run can be compared with others.

When `th_text` needs review, its details hold the 30 sampled flagged replies with a harmful / not-harmful choice for each, plus reviewer, date, decision and note fields. The Download button writes `review.json`; the command shown beside it applies it:

```bash
python scripts/safety_suite.py review "<run folder>" --review review.json [--verdict <file>]
```

`review` checks that the file names a case that needs review, judges exactly the sampled flagged replies, has counts that agree with its judgements and a valid decision (`accept` or `reject`), and that the same review was not applied before. It re-runs no model or classifier and writes `verdict.reviewed-<stamp>.json` and its report; the originals are never touched. Accept makes the case `pass_after_review`, reject makes it `fail`. An accept where most judged replies were harmful is applied but flagged with a warning line in the report.

## Configuration

- `config/thresholds.json`: per case the pass mark, minimum scored count, maximum excluded share, `direction`, the three served models' scores (`served`), a short `source`, and `needs_review` for `th_text`. A malformed entry is refused before a run.
- `config/generation.json`: token caps per case and per model, and each API model's sampling protocol.
- `config/prices.json` (git-ignored; copy `config/prices.example.json`): each API model's input and output price in THB per 1M tokens, which the `--max-thb` spend limit needs. It is read only when an API run needs a price, so local `--backend hf` runs, `rescore`, `report` and `review` work without it. An API run, or an API `--dry-run`, with the file missing or a model unpriced is refused with a message saying so.

## Data and model licences

The repository ships none of this data or these models. `scripts/fetch_data.sh` downloads the data into the git-ignored `data/`, and the models come from the Hugging Face hub at run time. Each stays under its own licence.

| Data or model | Used for | Licence | Credit |
|---|---|---|---|
| [HarmBench](https://github.com/centerforaisafety/HarmBench) text behaviours | `en_text` prompts | MIT | Mazeika et al., HarmBench, Center for AI Safety, 2024 (arXiv 2402.04249) |
| [ThaiSafetyBench](https://huggingface.co/datasets/typhoon-ai/ThaiSafetyBench) test split | `th_text` prompts | Apache-2.0 | Ukarapol et al., ThaiSafetyBench, ICLR 2026 Workshop (arXiv 2603.04992); hosted by typhoon-ai |
| [XSTest](https://github.com/paul-rottger/xstest) safe prompts | `over_refusal` prompts | CC-BY-4.0 | Röttger et al., XSTest, NAACL 2024. Its published model completions carry the original model licences and are fetched only for `tools/xstest_sanity_check.py`. |
| [ThaiExam](https://huggingface.co/datasets/typhoon-ai/thai_exam) | `th_knowledge` questions | Apache-2.0 | Pipatanakul et al., Typhoon: Thai Large Language Models, 2023 (arXiv 2312.13951); hosted by typhoon-ai |
| fastText [language identification model](https://fasttext.cc/docs/en/language-identification.html) (`lid.176.bin`) | `en_integrity` and `th_integrity` language checks | CC-BY-SA 3.0 | Joulin et al., Facebook AI Research, 2016. Fetched, not redistributed. |
| [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite) | `tools/probe_thinking.py` only | Creative Commons; the dataset card gives `cc` without a variant. Its code is MIT. | Jain et al., LiveCodeBench, 2024 (arXiv 2403.07974). The problems come from LeetCode, AtCoder and Codeforces. |
| [WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) | `tools/score_wildguardtest.py` and `tools/compare_refusal_scorer.py` only | ODC-BY (gated) | Han et al., WildGuard, NeurIPS 2024 Datasets and Benchmarks (arXiv 2406.18495); Allen Institute for AI and University of Washington |
| [cais/HarmBench-Llama-2-13b-cls](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) | `en_text` scorer | MIT (per its model card) | Center for AI Safety |
| [typhoon-ai/ThaiSafetyClassifier](https://huggingface.co/typhoon-ai/ThaiSafetyClassifier) | `th_text` scorer | Apache-2.0 | Ukarapol et al., ThaiSafetyBench (as above); hosted by typhoon-ai |
| [allenai/wildguard](https://huggingface.co/allenai/wildguard) | refusal judge in `tools/compare_refusal_scorer.py` | Apache-2.0 (gated) | Han et al., WildGuard (as above) |

## Tests

```bash
python -m pytest -s
```

The tests are offline. `-s` turns off output capture, which the one test that builds real terminal prompts needs; that test is skipped unless it runs in a Windows console.

## Known limitations

- **The Download and Copy buttons on the report have been tested only offline.** Try them by hand in a browser before relying on the review flow.

- **Lyrics are left out of `en_integrity`.** The 50 HarmBench behaviours tagged `lyrics` are not scored there, because song lyrics repeat by design and the loop checks would flag a chorus as a broken reply. They are not counted as prompts or as exclusions, and `en_text` is unchanged. Re-scoring a run that did not apply this needs `--rerun-scorers`.
- **The Thai harmful verdict is imprecise.** On a hand-labelled sample of 40 replies, only about a quarter of the replies the Thai classifier flagged as harmful were judged harmful. That is why `th_text` cannot fail a model on its own: above its mark a person reviews a sample.
- **Not tested:** coding, tool calling, images, and languages other than English and Thai.
- **Sampling.** Only each model's default thinking mode and recommended sampling are tested. Other settings may behave differently.
