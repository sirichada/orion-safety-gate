# Progress Report

**Student Name:** Sirichada

**Nickname:** Tingting

**Date Range:** 2026-10-26 to 2026-11-08

## Activities and Accomplishments

### Week 9 (26/10 - 1/11)
- Re-planned the remaining work: deferred the coding check and raised the pace to about 15 hours a week.
- Settled the decoding settings: each model uses its vendor's recommended sampling, and Mango also gets a presence penalty.
- Fixed the Thai scorer so long replies are no longer cut off.
- Narrowed the capability-honesty check to over-refusal replies.
- Split the capability checks into English integrity, Thai integrity and Thai knowledge.
- Built the English integrity scorer and tuned it on the first four full runs.
- Reviewed how the literature defines AI safety and added two papers on the topic.

### Week 10 (2/11 - 8/11)
- Set the English integrity threshold from hand labels.
- Built the Thai integrity scorer and the Thai knowledge check (565 ThaiExam questions), and tuned the Thai integrity threshold on Mango's pilot run.
- Set one 32,768-token output limit for every model and case.
- Completed Mango's first full run and started Qwen 3.8-27B's.
- Added a reply-length tool and a per-case run summary that counts empty replies.
- Mapped the harm categories in the major labs' usage policies against the benchmarks used.
- Measured the Thai harmful-content classifier on 40 hand-labelled Mango replies.
- Drafted an HTML mock-up of the verdict report.

## Problems or Issues Encountered
- The Thai harmful-content verdict is unreliable: precision 0.25 on 40 hand-labelled replies, and loading or scoring it differently did not help.
- Flash-Next's full run needs about 40 hours, and only one model can use the API key at a time.
- No published benchmark gives pass thresholds, so each one has to be proposed and justified.
- The schedule is tight: coding and a full tool-calling check does not fit.

## Next Steps
- Decide how each harmful verdict is presented in the report. 
- Finish Qwen 3.8-27B's run and decide between a full or sampled Thai run for Flash-Next.
- Review the thresholds.