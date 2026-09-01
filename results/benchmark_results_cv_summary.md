# VoltFlow Walk-Forward CV Summary

3 folds x 5 seeds. Each fold trains only on years before its eval year (never seen during that fold's training).

| Fold | Train years | Eval year | Seeds OK | Net PnL mean±std ($) | Min | Max |
|---|---|---|---|---|---|---|
| fold1 | 2015 | 2016 | 5/5 | 422.20±5.28 | 413.78 | 427.89 |
| fold2 | 2015-2016 | 2017 | 5/5 | 295.18±82.53 | 183.28 | 388.26 |
| fold3 | 2015-2017 | 2018 | 5/5 | 338.55±28.97 | 308.84 | 391.66 |

**Overall across all 15 successful (fold, seed) runs: mean net PnL $351.98 ± $73.06** (min $183.28, max $427.89)

Compare this distribution against the heuristic baselines' PnL on the same held-out years (see per-run reports in `results/cv/*.md`) to judge genuine generalization rather than a single-seed result.

## Raw per-run results

| Fold | Seed | Net PnL ($) | Status |
|---|---|---|---|
| fold1 | 1 | 418.57 | OK |
| fold1 | 2 | 427.89 | OK |
| fold1 | 3 | 426.50 | OK |
| fold1 | 4 | 413.78 | OK |
| fold1 | 5 | 424.26 | OK |
| fold2 | 1 | 374.83 | OK |
| fold2 | 2 | 216.43 | OK |
| fold2 | 3 | 388.26 | OK |
| fold2 | 4 | 313.09 | OK |
| fold2 | 5 | 183.28 | OK |
| fold3 | 1 | 316.43 | OK |
| fold3 | 2 | 337.00 | OK |
| fold3 | 3 | 308.84 | OK |
| fold3 | 4 | 338.81 | OK |
| fold3 | 5 | 391.66 | OK |
