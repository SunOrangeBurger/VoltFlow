# VoltFlow Benchmark Results (7-day episodes, n=5, csv=data/cv/fold1_eval.csv)

| Strategy | Net PnL ($) mean±std | Revenue ($) mean±std | Degradation ($) mean±std | Final SoH | Reward mean±std |
|---|---|---|---|---|---|
| Threshold Rule Heuristic | 75.08±1.09 | 79.75±1.10 | 4.67±0.03 | 1.0000 | 0.7508±0.0109 |
| TOU Heuristic | 44.87±6.94 | 51.98±6.91 | 7.11±0.04 | 1.0000 | 0.4487±0.0694 |
| VoltFlow RL (PPO) | 416.98±10.53 | 431.40±10.45 | 14.42±0.14 | 0.9999 | 4.1698±0.1053 |

**RL vs. best heuristic net PnL improvement: 455.4%** (Gate target: >= 15%, mean over 5 held-out episodes)
