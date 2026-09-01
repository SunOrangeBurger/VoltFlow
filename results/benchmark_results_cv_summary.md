# VoltFlow Walk-Forward CV Summary

3 folds x 5 seeds. Each fold trains only on years before its eval year (never seen during that fold's training).

| Fold | Train years | Eval year | Seeds OK | Net PnL mean±std ($) | Min | Max |
|---|---|---|---|---|---|---|
| fold1 | 2015 | 2016 | 5/5 | 421.90±11.01 | 402.60 | 432.72 |
| fold2 | 2015-2016 | 2017 | 5/5 | 325.27±47.47 | 275.46 | 386.94 |
| fold3 | 2015-2017 | 2018 | 5/5 | 331.13±29.11 | 285.10 | 369.95 |

**Overall across all 15 successful (fold, seed) runs: mean net PnL $359.44 ± $55.05** (min $275.46, max $432.72)

Compare this distribution against the heuristic baselines' PnL on the same held-out years (see per-run reports in `results/cv/*.md`) to judge genuine generalization rather than a single-seed result.

## Raw per-run results

| Fold | Seed | Net PnL ($) | Status |
|---|---|---|---|
| fold1 | 1 | 432.72 | OK |
| fold1 | 2 | 402.60 | OK |
| fold1 | 3 | 427.63 | OK |
| fold1 | 4 | 416.98 | OK |
| fold1 | 5 | 429.59 | OK |
| fold2 | 1 | 378.06 | OK |
| fold2 | 2 | 275.46 | OK |
| fold2 | 3 | 386.94 | OK |
| fold2 | 4 | 300.16 | OK |
| fold2 | 5 | 285.72 | OK |
| fold3 | 1 | 338.56 | OK |
| fold3 | 2 | 285.10 | OK |
| fold3 | 3 | 314.37 | OK |
| fold3 | 4 | 369.95 | OK |
| fold3 | 5 | 347.69 | OK |
