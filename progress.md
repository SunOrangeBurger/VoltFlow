# VoltFlow — Progress

Last updated: this session (initial build pass).

## Status: Rust core compiled & tests passing locally (17/17)

`cargo test` has now been run on real hardware (this sandbox still can't
compile Rust or reach kaggle.com — those constraints are unchanged). First
run surfaced a real numerical stability bug in `thermal.rs` (see deviation
#3 below); fixed and now all 17 unit tests pass. `maturin develop --release`
and the Python↔Rust bridge, PPO training, and the frontend are still
unverified locally as of this update — see `to_be_done.md`.

---

## Phase 1 — Data Acquisition & Validation

- [x] Dataset decision made: **Kaggle "Energy Consumption, Generation,
      Prices and Weather" (Spain)** only. CityLearn rejected (multi-building
      demand-response benchmark, wrong observation/action contract for a
      single-BESS arbitrage task).
- [x] `download_data.py` written — documents the Kaggle CLI flow, merges
      `energy_dataset.csv` + `weather_features.csv` (Madrid rows) into
      VoltFlow's flat schema.
- [x] `generate_synthetic_data.py` written **and run successfully** —
      produced `data/raw/energy_weather_spain.csv`, 8,640 rows (90 days,
      15-min resolution), validated against the loader's expected schema.
- [ ] Real Kaggle CSV not yet downloaded/merged (sandbox can't reach
      kaggle.com — this is a "you run it locally" step, see README.md).

## Phase 2 — Rust Core Engine (`voltflow_core`)

- [x] `battery/cell.rs` — coulomb counting + inverter efficiency (spec 4.1).
      Unit tests written, including a Gate-2 test that documents a spec
      inconsistency (see "Known Spec Deviations" below) rather than papering
      over it.
- [x] `battery/thermal.rs` — lumped-capacitance ODE (spec 4.2), explicit
      Euler integration, dt converted hours→seconds explicitly.
- [x] `battery/degradation.rs` — cycle + calendar aging (spec 4.3).
- [x] `env/stochastic.rs` — Ornstein-Uhlenbeck noise for price/ambient-temp
      domain randomization during training.
- [x] `env/simulation.rs` — `BessSimulation` state machine: reset/step,
      8-element observation vector (spec 5.1), reward function (spec 4.4).
      Buffers pre-allocated at construction; `step()` does no heap
      allocation in steady state.
- [x] `data/loader.rs` — CSV ingestion, column-name-tolerant (accepts a few
      aliases per field), skips malformed rows instead of panicking, no
      `unwrap()` in the parse path.
- [x] `lib.rs` — PyO3 bindings (`RustBessEnv::new/reset/step/get_state`),
      `todo!()` stubs from the spec fully replaced with real implementation.
- [x] `benches/sim_benchmark.rs` — Criterion bench for Gate 1 (single-thread
      + 4-thread rayon parallel throughput).
- [x] `cargo test` — run locally, 17/17 passing (after thermal.rs fix above).
- [ ] **Not run**: `cargo bench` / Gate 1 throughput verification.

## Phase 3 — Maturin Build & Gymnasium Environment

- [x] `pyproject.toml` configured with `[tool.maturin]` cross-referencing
      `crates/voltflow_core/Cargo.toml`.
- [x] `gym_wrapper.py` — adapted from spec's given code, with two bugs fixed:
      `options=None` is now handled safely (spec's version called
      `.get()` on it unconditionally), and the Rust RNG seed is now
      threaded through from Python instead of being hardcoded in Rust.
- [ ] **Not run**: `maturin develop --release` (needs local Rust toolchain).
- [ ] **Not run**: `python -m pytest` against the built extension.

## Phase 4 — Baseline Models & PPO Training Pipeline

- [x] `baselines.py` — `ThresholdRuleBaseline` (percentile-based) and
      `TouHeuristicBaseline` (fixed-hour tariff), plus a shared
      `run_episode()` helper.
- [x] `train_ppo.py` — full SB3 PPO pipeline, `SubprocVecEnv`/`DummyVecEnv`
      switch based on `--n-envs`, `EvalCallback` with best-model checkpointing.
- [x] `run_benchmarks.py` — Gate 4 script: runs heuristics (+ PPO if a
      checkpoint exists) over a 7-day episode, writes `benchmark_results.md`
      with a PnL comparison table and the >=15% improvement check.
- [ ] **Not run**: actual 2,000,000-step PPO training (explicitly out of
      scope for this session — would take hours; see to_be_done.md).
- [ ] **Not verified**: Gate 3 (PPO halts charging above 75th percentile
      price) — can't verify without a trained model.

## Phase 5 — Telemetry Server & Interactive Web UI

- [x] `server/app.py` — FastAPI app, background simulation loop, WebSocket
      broadcast to all connected clients, `/health` endpoint, graceful
      fallback to an idle policy if no PPO checkpoint is found.
- [x] Next.js dashboard — dark control-room theme (deliberately avoided the
      generic cream/terracotta and SaaS-card-kit defaults per design
      guidance; chose IBM Plex Sans/Mono + amber/cyan/green accent set
      instead).
  - [x] `useSimulationSocket.ts` — WS hook with auto-reconnect, 200-frame
        rolling history.
  - [x] `BatteryGauge.tsx` — radial SoC/SoH gauge, custom SVG arc math (no
        chart library needed for this one).
  - [x] `MetricsGrid.tsx` — 4-tile stat grid (PnL, price, reward, degradation
        cost) with lucide-react icons.
  - [x] `LiveChart.tsx` — reusable recharts line chart, used 4x on the page
        (PnL, price, SoC, cell temp).
  - [x] `page.tsx` — ties it all together.
- [ ] **Not run**: `npm install` / `npm run dev` (not attempted in this
      sandbox; no reason to expect issues, but unverified).

---

## Known Spec Deviations (intentional, documented in code)

1. **Gate 2 discharge percentage — RESOLVED.** Spec section 7's prose states
   discharging at 1.0C for 1 hour should drop SoC by "$(1.0 \times
   \eta_{inverter})/1.0 \approx 96\%$". Checked against the physics rather
   than just matching the arithmetic: inverter conversion loses energy in
   *both* directions, so charging draws $P_{action}$ but only
   $P_{action}\times\eta$ reaches the pack (multiply, loss), while
   discharging must pull *more* than $P_{action}$ from the pack to deliver
   $P_{action}$ to the grid (divide, loss). The section 4.1 formula (multiply
   on charge, divide on discharge) is physically correct as written — no
   code change needed. Gate 2's prose simply misquoted the charge-branch
   number (~96%) for the discharge case; the correct discharge figure is
   ~104.2% (pack gives up more than it delivers). `cell.rs`'s test now
   asserts this directly rather than hedging between two branches.

2. **`panic = "abort"` removed from the release profile.** The spec's
   `Cargo.toml` (section 2.1) includes this, but it's incompatible with
   `cargo test` and the criterion bench harness (both require unwinding).
   Removed; all other release optimizations (`opt-level=3`, `lto="fat"`,
   `codegen-units=1`) kept.

3. **Thermal integration was numerically unstable — FOUND & FIXED via
   `cargo test` on real hardware** (this sandbox can't compile Rust, so
   this only surfaced once you ran it locally). The original explicit-Euler
   integration converted $\Delta t$ to seconds correctly, but Euler
   integration itself overshot past ambient temperature in a single step:
   with default params, the thermal time constant $C_{thermal}/(hA) =
   15000/25 = 600$s is *shorter* than the 15-min (900s) simulation step, so
   Euler isn't just imprecise here, it's unstable (observed: 310K -> 292.2K
   in one step, undershooting past the 298.15K ambient it was cooling
   toward). Replaced with the exact analytic solution to the linear ODE
   (valid since $Q_{gen}$ is held constant within a step):
   $$T(t+\Delta t) = T_{eq} + (T(t) - T_{eq})\,e^{-\frac{hA}{C}\Delta t}, \quad T_{eq} = T_{ambient} + \frac{Q_{gen}}{hA}$$
   Unconditionally stable for any step size, costs one extra `exp()` call.
   Added a regression test (`large_timestep_does_not_overshoot_ambient`) in
   `thermal.rs` covering this exact failure mode.

4. **Revenue calculation reconstructs actual (post-clamp) energy delta**,
   not the raw requested action, so that revenue reflects what actually
   happened when the SoC boundary clips an action (e.g., agent tries to
   charge past `soc_max`). This wasn't explicit in spec 4.4 but seemed
   necessary for financial correctness — otherwise the ledger would count
   money for energy that was never actually stored/delivered.

## Files not yet delivered as downloads

None outstanding as of this update — full tree is ready to present.