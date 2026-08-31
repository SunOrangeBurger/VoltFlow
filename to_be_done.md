# VoltFlow — To Be Done

Things still outstanding after the first full training pass. Rust
toolchain, Python venv, `maturin develop --release`, and a full
2,000,000-timestep PPO run have all now happened on real (GPU) hardware —
the items below are what's left, in rough priority order.

## Locked sequence — STATUS: steps 1-4 all complete. Frontend is unblocked.

The walk-forward CV sweep is done: 15/15 runs succeeded (3 folds x 5 seeds),
each benchmarked only on its own fold's held-out year. Full results in
`benchmark_results_cv_summary.md` and per-run detail in `results/cv/*.md`.

**Result: PPO beats both heuristics on every single one of the 15 runs.**
Worst case (fold3 seed2): +136.8% over the best heuristic that year. Best
case (fold1 seed1): +476.4%. Gate 4's >=15% target is cleared by a wide
margin in every run, not just on average.

| Fold | Train years | Eval year | Net PnL mean+/-std ($) | Min | Max |
|---|---|---|---|---|---|
| fold1 | 2015 | 2016 | 421.90+/-11.01 | 402.60 | 432.72 |
| fold2 | 2015-2016 | 2017 | 325.27+/-47.47 | 275.46 | 386.94 |
| fold3 | 2015-2017 | 2018 | 331.13+/-29.11 | 285.10 | 369.95 |

Overall across all 15 runs: mean net PnL $359.44 +/- $55.05.

Notable: the two heuristics are *not* consistently ranked across years
(Threshold Rule: $75->$43->$39 across 2016/17/18; TOU: $44->$102->$120,
opposite trend) -- PPO stays in a much tighter $275-433 band regardless of
which heuristic happens to be stronger that year. This is a stronger
generalization signal than beating any single fixed baseline.

fold1's unusually tight spread (+/-11 vs +/-30-47 for fold2/fold3) is worth
a mental flag, not an action item: trained on only one year of data, so a
policy converging this consistently across seeds might mean 2016 was an
easy/predictable eval year, or that a single training year offers less
regime diversity, making seeds converge to a similar (possibly narrower)
strategy. fold2/fold3, with 2-3 years of training data, show more
seed-to-seed variance ($85-111 spread) which is arguably the more
trustworthy read on real-world seed sensitivity. Not worth chasing further
right now -- flagging for awareness only.

**Gate 3 verification — DONE, PASS.** Ran against `fold3_seed4`:
mean action -0.9676 above the 75th percentile price ($186.67/MWh, near-max
discharge) vs. -0.2885 below it, only 0.3% of above-threshold steps showing
any charging. Confirms genuine price-arbitrage behavior, independent
evidence alongside the CV sweep's PnL dominance.

**Confirmed checkpoint for frontend integration:
`models/cv/ppo_voltflow_fold3_seed4.zip`** (best fold3 PnL at $369.95,
Gate 3 verified, most training data / most recent held-out eval year).

**Next: only after Gate 3 passes**, decide which checkpoint to wire into
the frontend. Current leading candidate: `models/cv/ppo_voltflow_fold3_seed4.zip`.

## Other blocking items (independent of the CV sweep, can be done in parallel)

5. **Run `cargo bench` and check Gate 1** (>2M steps/sec across 4 threads).
   Still not run. If the number looks low, the likely culprit is the
   `StdRng` reseeding per parallel task in the bench (currently constructs
   a fresh `BessSimulation` per thread per bench iteration) — consider
   restructuring to reuse simulation instances across `iter()` calls.
6. **Verify the FastAPI server actually finds the `voltflow` package.**
   `voltflow.server.app` lives at `python/voltflow/server/app.py`; running
   `uvicorn voltflow.server.app:app` from repo root will fail with
   `ModuleNotFoundError` unless run with `--app-dir python`, or after
   `pip install -e .`. Point it at `models/cv/ppo_voltflow_fold3_seed4.zip`.

## Data (minor, not blocking, can be done in parallel)

7. **Patch the dedup-before-merge fix into `download_data.py` itself.**
   Currently the fix (dedupe `weather_features.csv` on `dt_iso` before
   joining) was only applied in the one-off merge that produced the
   current `data/raw/energy_weather_spain.csv`. A future from-scratch
   Kaggle re-download would reintroduce the duplicate-row issue unless
   `download_data.py` is patched to do this itself.

## Training follow-ups (after the CV sweep, not before)

8. **Consider a hyperparameter-tuned run** once the CV sweep establishes a
   generalization baseline. The current hyperparameters are spec defaults
   (`lr=3e-4`, `n_steps=2048`, `batch_size=256`, `n_epochs=10`,
   `gamma=0.99`), untuned. Only worth doing once the CV summary shows
   *where* the policy struggles (which fold/year), so tuning targets a
   real weakness instead of guessing.
9. **Inspect per-run training curves** in `logs/cv/*/` (tensorboard) for
   any fold/seed combos that failed to converge, not just the final
   eval PnL — a policy that converges late or unstably is a different
   problem than one that converges to a mediocre optimum.

## Frontend (unblocked — CV sweep + Gate 3 both passed)

10. **`npm install` inside `ui/`** and confirm the Next.js 14 app boots
    (`npm run dev`). Point `server/app.py` at
    `models/cv/ppo_voltflow_fold3_seed4.zip` — not the original
    single-seed `models/ppo_voltflow.zip`, which is now superseded.
11. **Wire the WS URL for non-localhost deployments** — currently hardcoded
    default is `ws://localhost:8000/ws/telemetry`, overridable via
    `NEXT_PUBLIC_VOLTFLOW_WS_URL`. Fine for local dev, needs attention if
    deployed anywhere else.
12. Optional polish (explicitly deferred, not required for a working
    dashboard): loading skeletons before first WS frame arrives, mobile
    layout pass beyond the basic Tailwind grid collapse, dark/light theme
    toggle (currently dark-only by design).

## Explicitly out of scope for this repo pass (not forgotten, just deferred)

- CityLearn integration — rejected as a dataset/env source (see
  `progress.md`), not revisited.
- Any cloud deployment (Docker, k8s, CI/CD) — spec never asked for this,
  only local runnability.
- Multi-battery / fleet-level dispatch — spec is single-BESS throughout.
- Authentication/authorization on the FastAPI server or WS endpoint — fine
  for local single-user dev, would need adding for anything shared.