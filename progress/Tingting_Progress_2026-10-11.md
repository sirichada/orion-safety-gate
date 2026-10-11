# Progress Report

**Student Name:** Sirichada

**Nickname:** Tingting

**Date Range:** 2026-09-28 to 2026-10-11

## Activities and Accomplishments

### Week 5 (28/9 - 4/10)
- Improved the evaluation harness: clean stop when the model stops answering, retries for empty responses, recorded exclusion reasons.
- Built case selection from a model's declared languages/modalities.
- Built a command-line tool that runs the selected cases and writes an admission verdict.
- Picked BigCodeBench (hard subset) for the coding check.

### Week 6 (5/10 - 11/10)
- Met P’ Most about how Orion is moving to a new server and scope smartness gate (added tool-use check).
- Got access to GPU server and baseline model; connected over VPN and confirmed Docker works.
- Built the over-refusal check (XSTest) and validated it against XSTest's published completions.
- Added partial runs (`--only`) for specific checks.
- Verdicts now record the full run configuration (precision, model revision, chat-template text).
- Defined "raw model" as the model as shipped by the providers. 
- Decided on BFCL as the tool-use check.
- Reviewed non-lab literature on access tiers; drafted the gate's claim, assumptions and limits.

## Problems or Issues Encountered
- Dropped the hallucination check because it needs an LLM judge. Might come back to later if I can find a more up-to-date and cheaper check. 

## Next Steps
- Run Mistral on the GPU server to validate the setup.
- Decide the safety-gate threshold.
- Build the coding and tool-use checks.
- Draft the user journey and Docker Compose setup.
