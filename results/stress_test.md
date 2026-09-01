# VoltFlow Stress Test Results

CSV: `data/raw/energy_weather_spain.csv`

Forces adversarial actions through the live simulation loop (not just unit-testing the clamp functions in isolation) to verify the SoC and thermal safety envelope holds under worst-case conditions, sampled across the full dataset rather than one cherry-picked window.

### Sustained max discharge (action=-1.0 every step): PASS
- Episodes: 30 (random-start, sampled across the full dataset)
- Steps evaluated: 2880
- SoC range observed: [0.0500, 0.3698] (hard bounds: [0.05, 0.95])
- Max cell temperature observed: 318.15 K (T_crit: 318.15 K, margin: 0.00 K)
- Hard bound violations: 0
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

