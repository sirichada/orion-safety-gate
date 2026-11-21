# Hazard Priority Ranking survey

Pairwise severity and likelihood rankings of MLCommons AILuminate's 12 hazard categories, collected from four CMKL faculty in September 2026. Not part of the gate's run path. The raw responses are not published; `tally.py` reads them from `responses/` when they are present.

Outcome: the faculty disagreed a lot, so harm categories are tagged and reported in the verdict, not used as a filter.

```
python3 tally.py responses/*.json
```
