# Progress Report

**Student Name:** Sirichada

**Nickname:** Tingting

**Date Range:** 2026-09-14 to 2026-09-27

## Activities and Accomplishments

### Week 3 (14/9 - 20/9)
- Interviewed Dr. Sarun and Dr. Fawad.
- Cleaned up public-facing references in the repo; batch-committed design doc changes.
- Ran peer review on the design doc, fixed round 1 findings, produced a frozen initial-submission snapshot.
- Built the EN-text (HarmBench) and TH-text (ThaiSafetyBench) prompt loaders.
- Built a generic HF model-calling wrapper, shared quantized load/unload helpers, and a reported-capability schema/parser.

### Week 4 (21/9 - 27/9)
- Interviewed Dr. Natnatee; re-ran the hazard-ranking tally with all four respondents.
- Added the capability-honesty check to the smartness gate.
- Built EN-text and TH-text scorers; fixed a VRAM growth bug in the EN scorer.
- Ran sanity checks (Llama-2/Mistral, English/Thai, with and without the BadNets trigger).
- Added a chunked runner so long evaluation runs survive mid-run resets.
- Added an OpenAI-compatible HTTP API generation backend alongside the existing HF backend.
- Investigated closing the smartness gate's tool-use and knowledge-currency coverage gap, prompted by a stakeholder report that MANGO hallucinated and misused its own tools. Surveyed BFCL, AbstentionBench, and Petri as candidate tools.

## Problems or Issues Encountered
- TH-text pipeline was silently truncating some responses; caught during sanity checks and resolved.
- The smartness gate has a real coverage gap (tool-use, knowledge-currency) that no surveyed tool can close this semester within budget.
  - Documented as a known limitation

## Next Steps
- Finish the TH-text scorer validation pass and begin end-to-end runs against Orion's existing models.
- Continue scoping the smartness gate's limits, including its lack of a tool-use or up-to-date-knowledge check.