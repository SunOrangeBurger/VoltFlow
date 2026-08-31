# VoltFlow — To Be Done

Things explicitly **not** completed in the initial build pass, in rough
priority order. This is scope you (or a future agent session) need to pick
up locally, since this sandbox has no Rust toolchain and can't reach
kaggle.com.

## Blocking (must do before anything else works)

1. **Install Rust toolchain locally** (`rustup`, stable 1.78+) — done if
   you're reading this after a successful `cargo test` run.
2. ~~Run `cargo test` inside `crates/voltflow_core/`~~ — **done**, 17/17
   passing. Along the way this surfaced and fixed a real numerical
   instability bug in the thermal model (see `progress.md` deviation #3) —
   worth reading if you're curious why cell temperatures behave the way
   they do near the 15-min timestep boundary. The Gate 2 discrepancy
   (spec prose said ~96% for discharge) is also resolved, no action needed.
3. **Build the Python venv with `uv`, then run `maturin develop --release`.**
   Use `uv venv --python 3.11 .venv` (this sandbox's Python is 3.12, and
   the pinned `torch==2.4.0`/`stable-baselines3==2.3.2` may lack 3.12
   wheels — 3.11 is safer). **Important:** `cargo`/pyo3's build script
   resolves whichever `python3` is first on `PATH` at build time, which may
   NOT be your venv's interpreter even if the venv is activated, especially
   on systems where a newer system Python (e.g. 3.14) shadows it. If you
   hit an error like `the configured Python interpreter version (3.14) is
   newer than PyO3's maximum supported version`, explicitly set:
   ```bash
   export PYO3_PYTHON=$(pwd)/.venv/bin/python3
   cargo clean   # clears stale build config cached against the wrong interpreter
   ```
   before retrying `cargo test` or `maturin develop --release`. Keep
   `PYO3_PYTHON` exported for the rest of the session.
4. **Run `cargo bench` and check Gate 1** (>2M steps/sec across 4 threads).
   Not yet run as of this update. If it doesn't hit that bar, the likely
   culprits are the `StdRng` reseeding per parallel task in the bench
   (currently constructs a fresh `BessSimulation` per thread per bench
   iteration, somewhat unfair to steady-state throughput) — consider
   restructuring the bench to reuse simulation instances across `iter()`
   calls if the number looks low.
5. **Verify the FastAPI server actually finds the `voltflow` package.**
   `voltflow.server.app` lives at `python/voltflow/server/app.py`, so
   running `uvicorn voltflow.server.app:app` from the repo root will fail
   with a `ModuleNotFoundError` unless you either run it with
   `--app-dir python`, or `pip install -e .` the package first. Not yet
   tested locally — flagging before you hit it.


## Data

5. **Get a Kaggle API token** (kaggle.com/settings → API → Create New
   Token), place at `~/.kaggle/kaggle.json`, then run:
   ```
   python python/voltflow/scripts/download_data.py
   ```
   This downloads + merges the real Spain dataset into
   `data/raw/energy_weather_spain.csv`, replacing the synthetic placeholder.
6. ~~Validate the real merged CSV~~ — **done**. Real data merged (35,064
   rows). Confirmed prices are 9.33-116.80 EUR/MWh with zero negative-price
   hours, well inside the old hardcoded -50/300 normalization bounds — this
   was indeed the flattened-observation problem this item predicted, and has
   been fixed (see progress.md "Real Spain dataset merged" entry): price
   normalization bounds are now derived from the dataset at load time
   instead of hardcoded. Needs `cargo test` re-run locally to confirm the
   new tests pass (sandbox can't compile Rust).
   - Minor follow-up, not blocking: patch the same dedup-before-merge fix
     into `download_data.py` itself (currently only applied in the one-off
     merge that produced the current CSV) so a future from-scratch Kaggle
     download doesn't reintroduce the duplicate-row issue.
7. ~~Thermal cooling capacity undersized for the cell's power rating~~ —
   **done**. See progress.md item 6: `h*A` is now auto-derived from
   `P_max`/climate instead of the spec's flat (and undersized) 25.0 W/K.
   **Needs `cargo test` locally** — 5 new tests added, not yet run.

## Training (Phase 4, deferred)

7. **Smoke-test PPO training** with a short run first:
   ```
   python -m voltflow.models.train_ppo --timesteps 50000 --n-envs 2
   ```
   before committing to the full 2,000,000-step run (spec Phase 4). Expect
   this to take hours on CPU; a GPU with `torch` CUDA build will help but
   SB3's default MlpPolicy is small enough that CPU is often fine too.
8. **Verify Gate 3**: after training, check whether the agent learns to
   idle/discharge instead of charge when price (obs index 4, normalized)
   exceeds roughly the 75th percentile of observed prices. No automated
   test for this yet — would be worth adding one that loads a checkpoint,
   replays a fixed price trace, and asserts the correlation.
9. **Run `run_benchmarks.py` against a real trained checkpoint** to check
   the >=15% net PnL improvement over both heuristics (Gate 4). The script
   is written and will run against heuristics-only right now (no checkpoint
   exists yet), but the RL comparison row won't appear until step 7-8 done.

## Frontend

10. **`npm install` inside `ui/`** and confirm the Next.js 14 app boots
    (`npm run dev`) — untested in this session. Package versions were
    hand-picked to match the spec's stated stack but not resolved/installed.
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
  progress.md), not revisited.
- Any cloud deployment (Docker, k8s, CI/CD) — spec never asked for this,
  only local runnability.
- Multi-battery / fleet-level dispatch — spec is single-BESS throughout.
- Authentication/authorization on the FastAPI server or WS endpoint — fine
  for local single-user dev, would need adding for anything shared.