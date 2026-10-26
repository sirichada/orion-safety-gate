# Progress Report

**Student Name:** Sirichada

**Nickname:** Tingting

**Date Range:** 2026-10-12 to 2026-10-25

## Activities and Accomplishments

### Week 7 (12/10 - 18/10)
- Documented the GPU server validation results and updated the project plan.
- Switched the coding check from BigCodeBench Hard to LiveCodeBench, since open-weight model cards report it and none report BigCodeBench.
- Merged the Thai output check and the Thai exam into one Thai capability check.
- Surveyed what coding, knowledge and safety benchmarks labs report at release; confirmed BFCL and XSTest are commonly used.
- Extended the survey of how labs gate model access to five more developers.
- Probed Orion's API for thinking mode, streaming and default decoding; all three served models think by default.
- Built API generation into the suite: streaming, spend limit, resume, dry-run and per-case output-length limits.
- Built a tool that measures reply lengths to propose those limits, and added options for vendor sampling settings and seeded prompt samples.
- Decided to keep thinking on as served, with a 10% ceiling on excluded (empty) replies.
- Ran small test runs on Flash-Next, Mango and Qwen 3.8-27B to compare decoding settings.

### Week 8 (19/10 - 25/10)
- No project work this week due to other commitments.

## Problems or Issues Encountered
- Thinking models often use the whole token budget on reasoning, leaving empty replies; the original 256-token limit was unusable.
- Some empty replies are repetition loops, mostly on Flash-Next and Mango at temperature 0; vendor sampling reduces them.
- Thai results flip between identical runs more than English or over-refusal, and the Thai scorer truncates long inputs (36 of 50 in one run).

## Next Steps
- Decide Mango's decoding settings.
- Record the vendor sampling decision in the design.
- Build the coding, tool-use and capability-honesty checks.
- Set pass thresholds for both gates.
