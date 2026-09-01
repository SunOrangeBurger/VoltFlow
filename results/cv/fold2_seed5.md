# VoltFlow Benchmark Results (7-day episodes, n=5, csv=data/cv/fold2_eval.csv)

| Strategy | Net PnL ($) mean±std | Revenue ($) mean±std | Degradation ($) mean±std | Final SoH | Reward mean±std |
|---|---|---|---|---|---|
| Threshold Rule Heuristic | 43.64±0.98 | 46.38±0.97 | 2.74±0.02 | 1.0000 | 0.4364±0.0098 |
| TOU Heuristic | 102.06±6.93 | 108.41±6.91 | 6.35±0.03 | 1.0000 | 1.0206±0.0693 |
| VoltFlow RL (PPO) | 183.28±4.26 | 195.66±4.13 | 12.39±0.15 | 0.9999 | 1.8328±0.0426 |

**RL vs. best heuristic net PnL improvement: 79.6%** (Gate target: >= 15%, mean over 5 held-out episodes)
