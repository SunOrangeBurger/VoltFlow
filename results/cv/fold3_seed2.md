# VoltFlow Benchmark Results (7-day episodes, n=5, csv=data/cv/fold3_eval.csv)

| Strategy | Net PnL ($) mean±std | Revenue ($) mean±std | Degradation ($) mean±std | Final SoH | Reward mean±std |
|---|---|---|---|---|---|
| Threshold Rule Heuristic | 39.49±1.15 | 43.54±1.15 | 4.04±0.01 | 1.0000 | 0.3949±0.0115 |
| TOU Heuristic | 120.40±6.94 | 127.08±6.91 | 6.67±0.04 | 1.0000 | 1.2040±0.0694 |
| VoltFlow RL (PPO) | 285.10±4.73 | 298.79±4.66 | 13.70±0.13 | 0.9999 | 2.8510±0.0473 |

**RL vs. best heuristic net PnL improvement: 136.8%** (Gate target: >= 15%, mean over 5 held-out episodes)
