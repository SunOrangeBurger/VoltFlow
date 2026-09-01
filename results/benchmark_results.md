# VoltFlow Benchmark Results (7-day simulation)

| Strategy | Net PnL ($) | Total Revenue ($) | Degradation Cost ($) | Final SoH | Total Reward |
|---|---|---|---|---|---|
| Threshold Rule Heuristic | 10.36 | 13.55 | 3.19 | 1.0000 | 0.1036 |
| TOU Heuristic | 111.82 | 117.40 | 5.58 | 1.0000 | 1.1182 |
| VoltFlow RL (PPO) | 348.42 | 359.37 | 10.94 | 0.9999 | 3.4842 |

**RL vs. best heuristic net PnL improvement: 211.6%** (Gate target: >= 15%)
