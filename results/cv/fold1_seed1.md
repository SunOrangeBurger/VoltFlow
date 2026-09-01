# VoltFlow Benchmark Results (7-day episodes, n=5, csv=data/cv/fold1_eval.csv)

| Strategy | Net PnL ($) mean±std | Revenue ($) mean±std | Degradation ($) mean±std | Final SoH | Reward mean±std |
|---|---|---|---|---|---|
| Threshold Rule Heuristic | 75.08±1.09 | 79.75±1.10 | 4.67±0.03 | 1.0000 | 0.7508±0.0109 |
| TOU Heuristic | 44.87±6.94 | 51.98±6.91 | 7.11±0.04 | 1.0000 | 0.4487±0.0694 |
| VoltFlow RL (PPO) | 418.57±9.45 | 433.32±9.37 | 14.75±0.14 | 0.9999 | 4.1857±0.0945 |

**RL vs. best heuristic net PnL improvement: 457.5%** (Gate target: >= 15%, mean over 5 held-out episodes)
