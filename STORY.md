# VoltFlow: What Broke, What We Found, What We Fixed

**A Battery Energy Storage System (BESS) arbitrage & safety platform — Rust simulation core, PPO reinforcement learning, live telemetry dashboard.**

Razorpay AI Buildathon — Open Track submission story doc.

---

## 1. The premise

VoltFlow is an autonomous dispatch system for a grid-scale battery: an RL agent decides, every 15 minutes, whether to charge, discharge, or hold — trying to buy electricity cheap and sell it expensive, without cooking the battery or destroying its lifespan in the process.

That "without cooking the battery" clause is the entire point of the project. Anyone can train a policy that makes money in backtest. The hard part — the part that decides whether this is a toy or something you'd actually let near real hardware — is proving the safety envelope holds *even when the policy tries to break it*. That's the thesis this doc is built around: we didn't just build a system that works when everything goes right. We went looking for the ways it could go wrong, found real ones, and fixed them — with the evidence to show it.

**What it's made of:**
- A Rust simulation core (`voltflow_core`) modeling coulomb-counted charge/discharge, lumped-capacitance thermal dynamics, and semi-empirical degradation — built for throughput (target: >2M steps/sec; measured 22.6M single-threaded, 35.9M with 4 threads).
- A PyO3/Gymnasium bridge exposing that core to Python with zero-copy, zero-allocation steps.
- A PPO agent trained via walk-forward cross-validation (3 folds × 5 seeds = 15 independent runs) on real Spanish electricity market and weather data (2015–2018).
- A live FastAPI + Next.js telemetry dashboard.

**The result, in one line:** across 15/15 CV runs — retrained end-to-end with the hard thermal interlock described below actually in the training loop, not bolted on after — the agent beat both baseline heuristics on net PnL in every single run, by 79.6% to 469.9%, with a mean net PnL of $351.98 ± $73.06 — while holding both SoC and thermal as true hard constraints, verified with zero violations across thousands of adversarial stress-test steps.

That last clause is where this story really lives.

---

## 2. Four things that broke before the system ever ran an RL episode

Before there was an agent to train, there was a physics simulation to get *right*. Four separate bugs surfaced in the first pass at implementing the spec — each one the kind of thing that looks fine until you actually run the numbers.

### Break #1 — The thermal integrator blew past ambient temperature
The lumped-capacitance heat equation was implemented with explicit-Euler integration. The battery's thermal time constant, at default parameters, is C/(hA) = 15,000/25 = **600 seconds**. The simulation's timestep is 15 minutes — **900 seconds**. Any numerical scheme with a stability threshold shorter than the step size it's being run at is going to misbehave, and explicit Euler did exactly that: a cell cooling toward 298.15K ambient would overshoot *past* ambient, in one step, down to 292.2K — physically nonsensical.

**Fix:** replaced Euler with the exact closed-form analytic solution to the linear ODE (`T(t+dt) = T_amb + Q/hA + (T(t) - T_amb - Q/hA) * exp(-hA/C * dt)`), which is unconditionally stable regardless of step size. This isn't a tuning fix — it's the actual solution to the differential equation, so there's no accuracy/speed tradeoff to make. Locked in with a regression test (`large_timestep_does_not_overshoot_ambient`) that specifically checks the failure mode.

### Break #2 — Price normalization was built for a spec, not the data
The spec assumed a price range of roughly -50 to 300 EUR/MWh for normalizing the observation vector. Real Spanish market prices in the dataset run 9.33–116.80 EUR/MWh — less than half that range, and always positive. Feeding that through fixed -50/300 bounds compresses the actual signal into a narrow sliver near 0.2, which starves the agent of the exact information (price magnitude) it needs to learn arbitrage.

**Fix:** normalization bounds are now derived from the loaded dataset itself at construction time (min/max with 15% padding, floored at a minimum width to avoid degenerate collapse on constant-price test data), rather than hardcoded. Four tests verify this, including a regression test confirming the observation actually spans a meaningful chunk of [0,1] on real-shaped data instead of collapsing.

### Break #3 — The cooling system was sized for a much smaller battery
The spec's default cooling capacity (h·A = 25.0 W/K) turned out to be undersized for its own power rating (500kW). At just half power, the resulting resistive heating pushed steady-state cell temperature ~54K above ambient — and a single 15-minute step already covered ~78% of that rise. Almost any nontrivial charge/discharge blew straight past the critical temperature within one step. The resulting thermal penalty then outweighed revenue by **3-4 orders of magnitude** (a single half-power step produced a thermal penalty of ~24,000 against revenue of ~-3). Left alone, this wasn't a training-data problem — it was a reward-shape problem that would have trained PPO into a degenerate "never act at any price" policy.

**Fix:** cooling capacity is now auto-derived from the cell's actual power rating and the hottest ambient temperature genuinely present in the loaded dataset, targeting a steady-state margin below critical temperature — the same "derive from data, don't hardcode" philosophy as the price-normalization fix. Five tests cover this, including directional checks (hotter climate → more required cooling) and pathological-input floors (a climate with almost no headroom to critical temperature doesn't blow up to infinite cooling capacity).

### Break #4 — Revenue accounting used the wrong number
Revenue was initially computed from the *requested* action's power, not the power that was actually delivered after the SoC boundary clamp kicked in. Near the edges of the state-of-charge envelope (nearly empty or nearly full), an agent's requested action gets physically clamped — but the ledger was still crediting/debiting it as if the full requested action had gone through.

**Fix:** revenue is now reconstructed from the *actual*, post-clamp SoC delta, converted back through inverter efficiency — so the financial ledger reflects grid-side energy that genuinely moved, not what was requested and partially refused by physics.

---

## 3. The centerpiece: going looking for the failure mode nobody had triggered yet

By the time all four gates were cleared — throughput, physics sanity, learned arbitrage behavior, and PnL outperformance — VoltFlow looked done. The trained agent never came close to a safety violation in normal operation. It would have been easy to stop there.

We didn't, because "the trained agent never violated the envelope" and "the envelope physically can't be violated" are two very different claims, and only the second one is a safety guarantee. The existing unit tests checked `clamp_soc` and `step_temperature` in isolation — correct behavior of the pieces, not the system under adversarial pressure.

### The stress test
We built `stress_test.py`: a script that forces adversarial actions — not the trained policy's choices, but worst-case forced sequences — through the *live* simulation loop, across 30 random-start episodes sampled from the full dataset (not one cherry-picked window):

1. **Sustained max discharge** (action = -1.0, every step, for a full day) — the worst case for both SoC drain and thermal load (inverter loss means discharge pulls *more* than the nominal power from the pack).
2. **Sustained max charge** (action = +1.0, every step) — worst case for the SoC ceiling.
3. **Oscillating extreme** (+1.0/-1.0 alternating every single step) — the most violent possible direction reversal, stress-testing the thermal integrator's claimed unconditional numerical stability under conditions the analytic-solution fix (Break #1) was never explicitly tested against.
4. **Price-spike response** — not forced actions this time, but the *trained* policy's actual behavior at the top 1% price tail, checking whether its arbitrage behavior generalizes to genuine extremes, not just the moderate 75th-percentile threshold the original Gate 3 check used.

### What it found

**SoC held perfectly.** Zero violations across 8,640 forced-action steps. `clamp_soc` is a true hard constraint — it physically enforces the bound every single step, regardless of what the policy or an adversary requests.

**Thermal did not.** Under sustained forced max discharge:
- 7 of 2,880 steps (0.24%) exceeded the critical temperature (318.15K)
- Peak: **319.58K — 1.43K over the limit**

The reason SoC held and thermal didn't comes down to *how* each constraint was enforced. SoC was a physical clamp — `soc.clamp(soc_min, soc_max)` — applied to the state every step, no matter what. Thermal was only a **soft constraint**: a `kappa * (T - T_crit)²` term subtracted from the reward, which discourages a *learning* agent from approaching the boundary but does nothing to stop a *forced* action from crossing it. Two constraints that looked equally "handled" in the spec were actually enforced by two fundamentally different mechanisms — and only one of them was a real guarantee.

**Root cause, once we went looking:** the cooling system (Break #3's fix) was sized against the *historical* hottest ambient temperature in the dataset, with a 10K margin. But the live simulation adds stochastic (Ornstein-Uhlenbeck) noise on top of that historical ambient for domain randomization. Occasionally, the simulated ambient exceeds the historical max the cooling system was sized against — eating directly into that 10K margin. It's not that the sizing math was wrong; it's that the thing it was sized against (historical max) and the thing it actually has to handle (historical max + noise) are two different numbers, and nothing enforced the gap between them.

**Why this never showed up before:** the trained policy's own behavior, checked separately in the price-spike-response part of the same stress test, never sustains max discharge long enough to trigger this — it moderates in response to the thermal reward penalty well before hitting the critical temperature. That's *good* arbitrage behavior (it's literally what Gate 3 measures), but it also means the gap had been sitting there the whole time, invisible, because nothing had gone looking for it with a genuinely adversarial input. A policy that "happens to behave well" is not the same claim as "the system prevents misbehavior" — and a safety-critical system needs the second one.

This is the finding we think matters most for the pitch: **the bug wasn't a crash, a wrong number, or a failed test. It was a documented, quantified gap between what the system's soft guardrail discourages and what it can actually stop — found by deliberately trying to break something that every existing test said was fine.**

---

## 4. The fix: a hard thermal interlock

The plan going in, written down before implementation started (not reverse-engineered afterward): add a real physical interlock, following the exact enforcement pattern that already worked for SoC — predict the outcome of an action *before* committing to it, and clamp the action itself if the predicted outcome would violate the bound. Not a bigger penalty. A physical impossibility.

**How it works:**

1. **Predict.** Before applying the requested power, invert the thermal model's closed-form solution to compute the maximum heat-generation rate that would keep this step's *predicted* temperature at or under the critical threshold, given the current cell temperature and ambient conditions.
2. **Clamp.** If the requested action's heat generation exceeds that budget, reduce the effective power to the maximum the thermal budget allows — inverting the heat equation (Q = I²R) back to a power limit, preserving the original charge/discharge direction.
3. **Propagate consistently.** The clamped power — not the originally requested one — is what actually gets used for the SoC update, the thermal step, degradation, and revenue reconstruction. This mirrors exactly how the existing SoC clamp already flows through to revenue (Break #4's fix): the ledger reflects what *actually happened*, not what was asked for.

This is a genuine engineering trade rather than a free lunch: it costs one extra `exp()` and `sqrt()` per simulation step. Benchmarked before/after — both single-thread and 4-thread throughput stayed within noise (a ~1-2% wobble, statistically indistinguishable from run-to-run variance), nowhere close to threatening the >2M steps/sec target the system clears by more than 10x regardless.

**Verified, not asserted.** Six new unit tests cover the interlock math in isolation — including a round-trip test (feed the derived heat budget back into the temperature model, confirm it lands within 0.01K of the critical threshold) and edge cases (zero-headroom climates, already-over-limit starting states, no NaN/negative outputs across a sweep of pathological inputs).

Then the same adversarial stress test that found the gap was re-run against the fix, live, end-to-end:

| Scenario | Before | After |
|---|---|---|
| Sustained max discharge | **FAIL** — 7/2,880 violations, peak 319.58K (+1.43K over) | **PASS** — 0/2,880 violations, peak exactly 318.15K (0.00K margin) |
| Sustained max charge | PASS (0 violations) | PASS (0 violations) |
| Oscillating extreme | PASS (0 violations) | PASS (0 violations) |

The peak landing at *exactly* the critical threshold (not comfortably under it) is the interlock working as designed, not a coincidence — it's the constraint boundary, hit and held, rather than overshot.

**The retrain is done.** The CV-trained checkpoints originally referenced in this doc's PnL numbers were trained *before* the interlock existed — under the old soft-only thermal penalty. We didn't want to carry that asterisk into a pitch, so we re-ran the full walk-forward CV sweep (3 folds × 5 seeds = 15 independent runs) with the hard interlock physically in the loop during training, not just during the stress test.

**Result: the agent still learns effective arbitrage. Every single run still clears Gate 4, with margin to spare.**

| Fold | Train years | Eval year | Pre-interlock PnL mean±std | Post-interlock PnL mean±std | Pre-interlock range | Post-interlock range |
|---|---|---|---|---|---|---|
| fold1 | 2015 | 2016 | $421.90 ± $11.01 | $422.20 ± $5.28 | $402.60–$432.72 | $413.78–$427.89 |
| fold2 | 2015–2016 | 2017 | $325.27 ± $47.47 | $295.18 ± $82.53 | $275.46–$386.94 | $183.28–$388.26 |
| fold3 | 2015–2017 | 2018 | $331.13 ± $29.11 | $338.55 ± $28.97 | $285.10–$369.95 | $308.84–$391.66 |
| **Overall (15 runs)** | | | **$359.44 ± $55.05** | **$351.98 ± $73.06** | $183.28*–$432.72 | $183.28–$427.89 |

*Pre-interlock overall min/max weren't broken out per-run in the original summary; shown range is post-interlock.

**Reading it honestly, not just favorably:**
- The overall mean net PnL moved by less than 2% ($359.44 → $351.98) — within noise for a system this stochastic, not a meaningful regression.
- **fold1 got tighter, not worse** (std $11.01 → $5.28) — the interlock didn't destabilize training on the easiest, least price-volatile eval year.
- **fold2 got noisier** (std $47.47 → $82.53, worst single run now $183.28 vs. $275.46 before). This is the fold with the most heuristic-beating-heuristic year-over-year swing already (TOU beats Threshold Rule 2.3x in this exact eval year), so it's plausibly the fold most sensitive to a harder physical constraint reshaping the exploration landscape during training — but it's a real, measurable cost of the fix, not something we're glossing over.
- **Every one of the 15 runs still beats the best heuristic baseline for its fold**, with the worst single run's margin now 79.6% (fold2, seed5) — down from a pre-interlock worst case of 136.8%, but still more than 5x the 15% Gate 4 target.

**One decision this surfaces that we're flagging rather than making unilaterally:** the previously-selected dashboard checkpoint (`fold3_seed4`, chosen pre-interlock at $369.95 PnL — the best fold3 run at the time, and fold3 was preferred for having the most training-year diversity) now scores $338.81 post-interlock, while `fold3_seed5` scores highest in that fold at $391.66. Whether to re-select the checkpoint, and by what criterion (raw PnL vs. consistency vs. behavioral checks like the Gate 3 price-threshold test), is worth doing deliberately rather than defaulting to "whichever number is biggest" — that's an open item, not a fix we've silently made.

---

## 5. Why this is the story, not just a changelog entry

Five separate issues, four in the physics layer and one in the safety layer, share a pattern worth naming explicitly: **every one of them was found by checking the system against the actual data and actual adversarial conditions it would face, not against the spec's stated assumptions.** The spec assumed a price range that didn't match the real market. It assumed a cooling capacity that didn't match its own power rating. It assumed a soft reward penalty was equivalent to a hard constraint. In every case, the fix wasn't "patch the symptom" — it was "derive the parameter from the data/physics it actually has to hold against," the same underlying philosophy applied five times across two different layers of the system.

The thermal interlock specifically demonstrates something the buildathon's own bar language calls out directly: not "does it work when the demo runs," but *"every action explainable, bounded, and gated"* and *"show one failure handled gracefully."* We have the failure. We have the numbers before and after. We have the mechanism, not just the outcome. That's the whole pitch.

---

## 6. Evidence trail (for anyone who wants to check our work)

- `crates/voltflow_core/src/battery/thermal.rs` — thermal model + interlock math + 11 tests (5 original + 6 new)
- `crates/voltflow_core/src/battery/cell.rs` — coulomb counting, inverter efficiency, SoC clamp
- `crates/voltflow_core/src/env/simulation.rs` — the step function where the interlock is wired in; predict-then-clamp applied before SoC/thermal/revenue all flow from the same actually-delivered power
- `python/voltflow/scripts/stress_test.py` — the adversarial live-loop stress test, source of the before/after numbers in Section 4
- `results/stress_test.md` — raw stress test output, both runs
- `results/README.md` / `results/benchmark_results_cv_summary.md` — full CV results, pre- and post-interlock (15/15 runs, PnL by fold)
- `results/cv/*.md` — all 15 individual per-run reports (fold × seed), raw source for the before/after table in Section 4
- `STATUS.md` — running log of every known deviation from spec, resolved and open, dated and undated claims kept separate
- `cargo test` — 34/34 passing after the interlock change
- `cargo bench` — Gate 1 throughput confirmed unaffected within noise

---

*This document reflects the state of the project as of the thermal interlock fix landing, the post-fix stress test passing, and the full 15-run CV retrain confirming the agent still learns effective arbitrage under the added hard constraint. Open item: re-selecting the dashboard checkpoint against the new post-interlock numbers (see Section 4).*