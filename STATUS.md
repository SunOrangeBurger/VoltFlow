# VoltFlow Project Status

## Overview

Last updated 2026-09-01, after running the Gate 1 throughput benchmark and completing the walk-forward cross-validation sweep (15/15 runs successful). All four specification gates are cleared.

## Current Status

**✅ ALL FOUR GATES CLEARED**

### Gate Verification Status
| Gate | Description | Status | Evidence |
|------|-------------|--------|----------|
| **Gate 1** | Throughput: >2M steps/sec across 4 threads | **PASS** | `cargo bench`, release profile: 35.9M steps/sec (4-thread parallel), 22.6M steps/sec (single-thread). See caveat below. |
| **Gate 2** | Electrochemical sanity: Discharge/charge physics verified | **PASS** | `crates/voltflow_core/src/battery/cell.rs` tests |
| **Gate 3** | Behavioral: Agent learns price-threshold arbitrage | **PASS** | Mean action above p75: -0.9676 (discharge) vs -0.2885 below |
| **Gate 4** | Performance: PPO beats heuristics by ≥15% net PnL | **PASS** | 15/15 CV runs successful, worst case +136.8% improvement |

**Gate 1 caveat:** 4 threads measured only ~1.6x the single-thread throughput, not ~4x. The >2M steps/sec target is cleared with a wide margin either way, but the sub-linear scaling itself hasn't been diagnosed (likely contention in the benchmark harness rather than the simulation core, but unconfirmed) — see Remaining Tasks.

### Current Checkpoints
- **Tracked in repo (for quick smoke-testing only):** `models/ppo_voltflow.zip`, `models/best_model.zip` — early single-seed runs, not CV-validated.
- **Not tracked (reproduce locally):** `models/cv/ppo_voltflow_fold3_seed4.zip` — the CV-selected best policy referenced in `results/README.md`. `models/` is gitignored by design (binaries shouldn't live in git); run `python python/voltflow/scripts/run_cv_sweep.py` to regenerate it. See STATUS item under "Medium Priority" if you want this changed.

## Phase Completion Status

### Phase 1 — Data Acquisition & Validation ✅ COMPLETE
- **Dataset:** Kaggle "Energy Consumption, Generation, Prices and Weather" (Spain)
- **File:** `data/raw/energy_weather_spain.csv` (35,064 hourly rows, 2015–2018)
- **Validation:** Schema matches loader expectations, price normalization bounds derived from actual data

### Phase 2 — Rust Core Engine ✅ COMPLETE
- **Components:** `battery/cell.rs`, `battery/thermal.rs`, `battery/degradation.rs`, `env/simulation.rs`, `data/loader.rs`
- **Tests:** 28/28 passing, including thermal stability and price normalization fixes
- **Benchmarks:** `benches/sim_benchmark.rs` — Gate 1 verified, see table above

### Phase 3 — Maturin Build & Gymnasium Environment ✅ COMPLETE
- **Build:** `maturin develop --release` successful
- **Wrapper:** `gym_wrapper.py` with bug fixes for RNG seeding and options handling
- **Integration:** Training and benchmark pipelines exercise real `.step()` calls

### Phase 4 — Baseline Models & PPO Training ✅ COMPLETE
- **Baselines:** `ThresholdRuleBaseline` (percentile-based) and `TouHeuristicBaseline` (fixed-hour tariff)
- **Training:** Full 2,000,000-timestep PPO pipeline with `train_ppo.py`
- **CV Sweep:** 3 folds × 5 seeds = 15 runs, all successful (see [results/README.md](./results/README.md))

### Phase 5 — Telemetry Server & Interactive Web UI ✅ COMPLETE
- **Backend:** FastAPI WebSocket server with background simulation loop
- **Frontend:** Next.js dashboard with dark control-room theme
- **Components:** Battery gauge, metrics grid, live charts with Recharts

## Known Specification Deviations

### 1. Thermal Integration Stability ✅ RESOLVED
- **Issue:** Explicit-Euler integration unstable with 15-min step size (thermal time constant: 600s < 900s step)
- **Fix:** Replaced with exact analytic solution to linear ODE
- **Test:** `large_timestep_does_not_overshoot_ambient` regression test

### 2. Price Normalization Bounds ✅ RESOLVED
- **Issue:** Real Spain prices (9.33–116.80 EUR/MWh) narrower than spec's assumed -50/300 range
- **Fix:** Derive normalization bounds from loaded dataset with 15% padding
- **Test:** Four tests in `simulation.rs` verify bound derivation

### 3. Thermal Cooling Capacity ✅ RESOLVED
- **Issue:** Default `h*A = 25.0 W/K` undersized relative to `P_max = 500kW`
- **Fix:** Auto-size cooling capacity from cell's rated power and dataset's hottest ambient
- **Test:** `sustained_max_power_stays_near_but_under_t_crit_with_derived_cooling`

### 4. Revenue Calculation ✅ RESOLVED
- **Issue:** Revenue must reflect actual (post-clamp) energy delta, not raw requested action
- **Fix:** Reconstruct Δenergy from clamped SoC change rather than raw P_eff
- **Impact:** Financial correctness when actions clip at SoC boundaries

### 5. Dataset Deduplication ⚠️ PENDING PATCH
- **Issue:** `weather_features.csv` contains ~2,214 duplicate-timestamp rows for Madrid
- **Current:** Deduped in one-off merge producing current CSV
- **Pending:** Patch not yet applied to `download_data.py` itself
- **Priority:** Low (current CSV works, future downloads would need fix)

### 6. Thermal Safety Envelope Under Adversarial Discharge ⚠️ DOCUMENTED, HARD ENFORCEMENT PLANNED
- **What we did:** Built `python/voltflow/scripts/stress_test.py` to force adversarial actions (sustained max discharge, sustained max charge, rapid +1/-1 oscillation) through the live simulation loop across 30 random-start episodes sampled from the full dataset, plus a separate check of the trained policy's behavior at the top 1% price tail. This goes beyond the existing unit tests, which check `clamp_soc`/`step_temperature` in isolation rather than under adversarial full-episode conditions. Full results: [results/stress_test.md](./results/stress_test.md).
- **What we found:** SoC held as a true hard constraint in all scenarios — 0 violations across 8640 forced-action steps, `clamp_soc` enforces it physically every step. Thermal is currently a **soft constraint only** (enforced via the `kappa*(T-T_crit)^2` reward penalty, not a physical interlock): under sustained forced max discharge, 7 of 2880 steps (0.24%) exceeded `T_crit` (318.15K), peaking at 319.58K (1.43K over).
- **Root cause:** `derive_h_times_a` sizes cooling capacity against the *historical* hottest ambient in the loaded CSV, with a 10K margin. Live simulation adds OU noise on top of historical ambient (`env/stochastic.rs`), so simulated ambient can occasionally exceed the historical max the cooling system was sized against, eating into that margin.
- **Why this doesn't affect existing Gate results:** the price-spike-response part of the same stress test confirms the *trained* policy never sustains max discharge long enough to trigger this — it moderates in response to the thermal penalty well before T_crit, which is exactly the arbitrage behavior Gate 3 measures. The breach only appears under forced adversarial actions no trained policy here has been observed to take.
- **Long-term fix (not yet implemented):** add a real hard thermal interlock — clamp/override the requested action when the *predicted* post-step temperature would exceed `T_crit`, the same enforcement pattern `clamp_soc` already uses for SoC, rather than relying solely on the reward penalty to discourage it. This requires a Rust change (predict-then-clamp in `env/simulation.rs`'s step function) and a full retrain to confirm PPO still learns useful arbitrage behavior under the added constraint (~2 hours for the full 15-fold CV sweep on current hardware).
- **Priority:** Medium. Not a blocker — the soft constraint has never been breached by any trained policy in this project, only by intentionally adversarial forced actions designed to find this. Tracked here as a known, quantified limitation rather than left undiscovered.

## Remaining Tasks

### High Priority
1. **Fix Python package discovery** - Add `python-source = "python"` to `[tool.maturin]` in `pyproject.toml`, re-run `maturin develop --release`, confirm `python -m voltflow.models.train_ppo --help` works from repo root without `PYTHONPATH` set. This affects every script that does `from voltflow...` (train_ppo.py, run_benchmarks.py, verify_gate3.py, stress_test.py), not just the FastAPI server — confirmed by hitting `ModuleNotFoundError: No module named 'voltflow'` running stress_test.py fresh. Once confirmed, simplify the `--app-dir python` workaround out of the README's dashboard instructions too.
2. **Point dashboard at a validated checkpoint** - `models/cv/ppo_voltflow_fold3_seed4.zip` is now committed to the repo (previously gitignored under the blanket `/models/` rule — that rule has been narrowed). Update dashboard env var to use it instead of the smoke-test-only `ppo_voltflow.zip`.

### Medium Priority
3. **Add hard thermal interlock** - See "Known Specification Deviations #6": thermal is currently a soft (reward-penalty) constraint and can be breached by adversarial forced actions (confirmed via stress_test.py, not yet observed under any trained policy). Fix is a predict-then-clamp change in `env/simulation.rs`'s step function, same pattern as the existing SoC clamp, followed by a full CV retrain (~2 hours) to confirm PPO still learns useful arbitrage under the added constraint. Not a blocker for submission; tracked as a known, quantified, and deliberately deferred limitation.
4. **Diagnose Gate 1 sub-linear thread scaling** - 4 threads measured ~1.6x single-thread throughput, not ~4x; profile whether this is benchmark-harness contention or a real bottleneck in the simulation core
5. **Patch dedup fix into `download_data.py`** - For future Kaggle re-downloads
6. **Run `npm install` / `npm run dev`** - Confirm dashboard boots
7. **Configure WS URL for deployments** - Currently hardcoded to localhost

### Low Priority / Future Work
8. **Hyperparameter tuning** - Once CV identifies specific weaknesses
9. **Training curve analysis** - Inspect tensorboard logs for convergence patterns
10. **Mobile layout polish** - Basic Tailwind grid collapse exists, could be improved
11. **Theme toggle** - Currently dark-only by design
12. **Fix stale price-normalization constants in `verify_gate3.py`** - hardcodes the old -50/300 bounds to denormalize price for its printed EUR/MWh figure; doesn't invalidate the PASS/FAIL verdict (percentile grouping is invariant to affine rescaling) but the printed absolute price number is wrong. Cosmetic, low priority.

## Out of Scope (Explicitly Deferred)
- CityLearn integration - Rejected as dataset/env source
- Cloud deployment (Docker, k8s, CI/CD) - Spec only requires local runnability
- Multi-battery / fleet-level dispatch - Spec is single-BESS throughout
- Authentication/authorization - Fine for local single-user dev

## Files & Artifacts

### Documentation
- `README.md` - Main setup guide and overview
- `TECHNICAL.md` - Complete mathematical specification
- `STATUS.md` - This file (progress + remaining tasks)
- `results/README.md` - Benchmark results summary

### Key Checkpoints
- `models/ppo_voltflow.zip`, `models/best_model.zip` - Tracked, single-seed, quick-test only
- `models/cv/ppo_voltflow_fold3_seed4.zip` - Primary validated policy; reproduce via `run_cv_sweep.py`, not committed
- `models/cv/` - Full archive of 15 CV run checkpoints; reproduce locally, not committed

### Results & Logs
- `results/cv/*.md` - Detailed per-run CV reports (15 files)
- `logs/cv/` - Tensorboard logs and training outputs
- `results/benchmark_results.md` - Original single benchmark
- `results/benchmark_results_cv_summary.md` - Aggregated CV summary

---

**Next Steps:** Diagnose Gate 1 thread scaling, decide on committing a CV checkpoint for reviewers, finalize dashboard integration.