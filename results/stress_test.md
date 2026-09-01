# VoltFlow Stress Test Results

CSV: `data/raw/energy_weather_spain.csv`

Forces adversarial actions through the live simulation loop (not just unit-testing the clamp functions in isolation) to verify the SoC and thermal safety envelope holds under worst-case conditions, sampled across the full dataset rather than one cherry-picked window.

### Sustained max discharge (action=-1.0 every step): FAIL
- Episodes: 30 (random-start, sampled across the full dataset)
- Steps evaluated: 2880
- SoC range observed: [0.0500, 0.3698] (hard bounds: [0.05, 0.95])
- Max cell temperature observed: 319.58 K (T_crit: 318.15 K, margin: -1.43 K)
- Hard bound violations: 7
- Non-finite (NaN/Inf) states: 0

### Sustained max charge (action=+1.0 every step): PASS
- Episodes: 30 (random-start, sampled across the full dataset)
- Steps evaluated: 2880
- SoC range observed: [0.6200, 0.9500] (hard bounds: [0.05, 0.95])
- Max cell temperature observed: 313.72 K (T_crit: 318.15 K, margin: 4.43 K)
- Hard bound violations: 0
- Non-finite (NaN/Inf) states: 0

### Oscillating extreme (+1.0/-1.0 alternating every step): PASS
- Episodes: 30 (random-start, sampled across the full dataset)
- Steps evaluated: 2880
- SoC range observed: [0.0500, 0.6200] (hard bounds: [0.05, 0.95])
- Max cell temperature observed: 314.80 K (T_crit: 318.15 K, margin: 3.35 K)
- Hard bound violations: 0
- Non-finite (NaN/Inf) states: 0

### Price spike response (trained policy, `models/cv/ppo_voltflow_fold3_seed4.zip`): PASS
- Steps evaluated: 2880
- Mean action above p75 (Gate 3's threshold): -0.6741
- Steps in top 1% price tail: 29
- Mean action in extreme tail: -0.8298
- Fraction charging in extreme tail: 0.0%

Checks whether the agent's price-threshold behavior (Gate 3) holds at genuine tail prices, not just the moderate p75 threshold Gate 3 itself checks -- i.e. does the learned policy generalize, or does it only work in the price regime it happened to see most during training.

