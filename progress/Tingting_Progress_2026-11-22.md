# Progress Report

**Student Name:** Sirichada

**Nickname:** Tingting

**Date Range:** 2026-11-09 to 2026-11-22

## Activities and Accomplishments

### Week 11 (9/11 - 15/11)
- Completed full runs on Qwen 3.8-27B and Qwen 3.8-Flash-Next, so all three served models now have full results.
- Decided how a high Thai harm score is handled: a person reviews it instead of it failing the model, since the Thai classifier is weak.
- Made the English harm classifier fit a 12 GB GPU by scoring replies one at a time.
- Moved the API key into a local `.env` file.

### Week 12 (16/11 - 22/11)
- Took tool calling out of the smartness gate for time; it is reported as untested.
- Left song-lyrics prompts out of the English integrity check, since repetition is expected there.
- Fixed the Thai exam answer parser and re-scored.
- Set the pass marks from the weakest served model per case, with a noise margin.
- Built the verdict states (including "review needed"), the HTML admission report, the review command and an offline re-score command.
- Validated the gate:
  - Qwen3-8B against a safety-removed (abliterated) copy: English harm 0.25 → 0.77, confirmed by a second judge (WildGuard).
  - The HarmBench classifier against human labels: 86.4% agreement.
  - Batch size changes scores by at most 13% of the margin.
  - Hand-read 147 replies against the classifiers' own rules: HarmBench undercounts the abliterated copy (29 of 30 harmful by hand, 25 flagged), so the measured gap is a lower bound; it agreed with the hand read on 19 of 20 served-model replies; and the over-refusal check's errors nearly cancel on the served models.
- Prepared the repository to go public: removed private pricing and survey responses from the history, moved standalone tools into `tools/`, read the API address from the environment, and rewrote the README.
- Packaged the suite as a Docker container and ran it on a local NVIDIA GPU, with the same scores as outside the container.
- Changed the report to say why a case has too little data, instead of always advising a re-run.
- Drafted the public design and evaluation documents, including requirements, a coverage table per test case, evaluation criteria and known limits.
- Grounded the "why trust the gate" argument in an organizational-trust model (Mayer et al., 1995).

## Problems or Issues Encountered
- Qwen 3.8-Flash-Next's full run was long, and the API key allows one model at a time.
- The English harm classifier did not load on the CMKL server (host memory), so scoring has only run locally.
- Coding, tool calling and knowledge-currency checks do not fit before the deadline.

## Next Steps
- Merge the container, report and documentation changes, and make the repository public.