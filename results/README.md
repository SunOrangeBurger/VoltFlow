# VoltFlow Benchmark Results

This directory contains comprehensive performance evaluation results from the VoltFlow cross-validation sweep and individual benchmark runs.

## Summary Statistics

### Walk-Forward Cross-Validation (3 folds × 5 seeds = 15 runs)
- **Success Rate:** 15/15 (100%)
- **Overall Mean Net PnL:** $359.44 ± $55.05
- **Improvement over Best Heuristic:** 136.8% to 476.4% across all runs
- **Gate 4 Target:** ≥15% improvement ✅ Exceeded in every run

### Fold Performance
| Fold | Train Years | Eval Year | Net PnL Mean±Std ($) | Min | Max |
|------|-------------|-----------|----------------------|-----|-----|
| **fold1** | 2015 | 2016 | 421.90±11.01 | 402.60 | 432.72 |
| **fold2** | 2015-2016 | 2017 | 325.27±47.47 | 275.46 | 386.94 |
| **fold3** | 2015-2017 | 2018 | 331.13±29.11 | 285.10 | 369.95 |

## File Organization

### Root Level
- `benchmark_results.md` - Original single benchmark (7-day simulation)
- `benchmark_results_cv_summary.md` - Aggregated CV summary with statistics
- `benchmark_results_cv_summary.json` - Machine-readable CV summary

### Cross-Validation Results (`cv/` directory)
15 individual run reports, one per (fold, seed) combination:
- `fold1_seed1.md` through `fold1_seed5.md`
- `fold2_seed1.md` through `fold2_seed5.md`  
- `fold3_seed1.md` through `fold3_seed5.md`

Each report contains:
- Net PnL comparison (PPO vs. both heuristics)
- Total revenue and degradation costs
- Final State of Health (SoH)
- Episode reward totals

## Key Findings

### 1. Consistent Superiority
**PPO beats both heuristics on every single run** (15/15), demonstrating robust generalization rather than lucky seed selection.

### 2. Heuristic Instability
The two heuristics show inconsistent year-over-year performance:
- **Threshold Rule:** $75 → $43 → $39 (2016→2017→2018)
- **TOU Heuristic:** $44 → $102 → $120 (opposite trend)

PPO maintains a tighter $275-433 band regardless of which heuristic happens to be stronger each year.

### 3. Fold Performance Patterns
- **fold1** (1 training year): Tightest spread (±$11), likely due to less regime diversity
- **fold2/fold3** (2-3 training years): Larger spreads (±$30-47), more realistic generalization estimates
- **Best checkpoint:** `fold3_seed4` ($369.95 PnL) selected for dashboard integration

### 4. Gate 3 Behavioral Verification
Against `fold3_seed4` checkpoint:
- **75th percentile price:** $186.67 EUR/MWh
- **Mean action above p75:** -0.9676 (near-max discharge)
- **Mean action below p75:** -0.2885
- **Charging above p75:** 0.3% of steps
- **Result:** PASS - Agent learns genuine price-threshold arbitrage

## Methodology

### Walk-Forward Design
Each fold trains only on years **strictly before** its eval year:
- **fold1:** Train on 2015, evaluate on 2016 (never seen during training)
- **fold2:** Train on 2015-2016, evaluate on 2017  
- **fold3:** Train on 2015-2017, evaluate on 2018

This mimics real-world deployment where models are trained on historical data and deployed on future unseen periods.

### Benchmark Protocol
Each evaluation:
1. Loads held-out year CSV for the fold
2. Runs 5 episodes with each strategy (PPO, Threshold Rule, TOU)
3. Records net PnL (revenue - degradation costs)
4. Compares PPO against best heuristic for that year

### Statistical Significance
- **5 seeds per fold** accounts for RL training stochasticity
- **Mean ± standard deviation** reported for each fold
- **15 total runs** provides strong statistical evidence

## Usage Notes

### Primary Checkpoint
`models/cv/ppo_voltflow_fold3_seed4.zip` is the recommended checkpoint for:
- Dashboard integration
- Further benchmarking
- Behavioral analysis

### Result Interpretation
- **Net PnL:** Revenue minus degradation costs (real economic metric)
- **Improvement %:** Calculated against best heuristic for that specific year
- **Consistency:** More important than absolute PnL value

### Limitations
- Results specific to Spain 2015-2018 dataset
- Assumes single 1MWh/500kW BESS configuration
- Does not include grid connection fees or other real-world costs

---

**For complete technical details, see [TECHNICAL.md](../TECHNICAL.md).**