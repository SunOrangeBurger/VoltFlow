"""Stress test: verifies the safety envelope holds under adversarial/extreme
conditions, distinct from Gates 1-4 which check throughput, physics sanity,
learned behavior, and PnL under *normal* operation.

Every buildathon track's "bar" calls out the same thing in different words:
"every money action explainable, bounded and gated" (Track 1), "show ...
one failure handled gracefully" (Track 1), "honest metrics including
false-positive cost" (Track 2). This script is that evidence for VoltFlow:
it doesn't just unit-test the clamp functions in isolation (see
crates/voltflow_core/src/battery/cell.rs tests) -- it drives full episodes
through the *real* simulation loop under conditions designed to break the
safety envelope, and reports whether it actually held, with real numbers,
not just a pass/fail.

Scenarios:
  1. sustained_max_discharge - force action=-1.0 every step. Discharge is
     the worse case for both SoC (draining) and thermal (inverter loss
     means pack-side P_eff > P_max on discharge, see cell::effective_power_kw
     doc comment) -- this is the same worst case derive_h_times_a() sized
     the cooling system against, but exercised via the live sim loop rather
     than a closed-form steady-state calculation.
  2. sustained_max_charge - force action=+1.0 every step. Worst case for
     SoC ceiling and SoH (charging degradation).
  3. oscillating_extreme - alternate +1.0/-1.0 every single step. Tests
     the thermal integrator's claimed unconditional numerical stability
     (see battery::thermal::step_temperature doc comment) under the most
     violent possible action sequence, plus checks for non-finite (NaN/Inf)
     state under rapid direction reversal.
  4. price_spike_response - (requires --ppo-model) runs the *trained*
     policy, not forced actions, and checks whether its price-threshold
     behavior (Gate 3's mean-action-above-p75 check) still holds at the
     top 1% of observed prices, not just p75 -- i.e. does it generalize to
     genuine tail events, not just the moderate threshold Gate 3 checks.

Scenarios 1-3 sample many random-start episodes across the whole dataset
(not a single cherry-picked window) so "worst observed" reflects the range
of ambient/price conditions actually in data/raw/energy_weather_spain.csv,
including the hottest days and most volatile prices in 2015-2018.

Known bounds this script checks against (see source for each):
  SOC_MIN / SOC_MAX  -- crates/voltflow_core/src/battery/cell.rs CellParams::default
  T_CRIT_K           -- crates/voltflow_core/src/env/simulation.rs FinancialParams::default
These are duplicated here (not read from Rust) because RustBessEnv doesn't
expose them via PyO3. If they're ever changed in the Rust defaults, this
script's constants must be updated too -- a known limitation, not silent.

Usage:
    python stress_test.py --csv data/raw/energy_weather_spain.csv \\
        --ppo-model models/cv/ppo_voltflow_fold3_seed4.zip
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from voltflow.envs.gym_wrapper import VoltFlowEnv

# Mirrors crates/voltflow_core/src/battery/cell.rs CellParams::default().
SOC_MIN = 0.05
SOC_MAX = 0.95
# Mirrors crates/voltflow_core/src/env/simulation.rs FinancialParams::default().
T_CRIT_K = 318.15


def run_forced_action_episodes(csv, action_value, n_episodes, days, seed):
    """Runs n_episodes independent random-start episodes with a *fixed*
    action every step (bypassing any policy), and tracks the worst-case
    SoC/thermal excursion and whether the hard bounds were ever violated.
    """
    max_steps = days * 96
    worst_soc = 0.5
    worst_soc_direction = None  # track separately: min (charge-limit risk from discharge? no)
    min_soc_seen = 1.0
    max_soc_seen = 0.0
    max_t_cell_seen = 0.0
    violations = 0
    non_finite = 0
    n_steps_total = 0

    for ep in range(n_episodes):
        env = VoltFlowEnv(csv_path=csv, max_steps=max_steps, seed=seed + ep)
        obs, _ = env.reset(options={"randomize": True})
        action = np.array([action_value], dtype=np.float32)
        for _ in range(max_steps):
            obs, reward, term, trunc, info = env.step(action)
            n_steps_total += 1
            soc = info.get("soc", float("nan"))
            t_cell = info.get("t_cell_k", float("nan"))

            if not (math.isfinite(soc) and math.isfinite(t_cell) and math.isfinite(reward)):
                non_finite += 1
                continue

            min_soc_seen = min(min_soc_seen, soc)
            max_soc_seen = max(max_soc_seen, soc)
            max_t_cell_seen = max(max_t_cell_seen, t_cell)

            if soc < SOC_MIN - 1e-4 or soc > SOC_MAX + 1e-4:
                violations += 1
            if t_cell > T_CRIT_K + 1e-3:
                violations += 1

            if term or trunc:
                break

    return {
        "n_episodes": n_episodes,
        "n_steps_total": n_steps_total,
        "min_soc_seen": min_soc_seen,
        "max_soc_seen": max_soc_seen,
        "max_t_cell_seen": max_t_cell_seen,
        "violations": violations,
        "non_finite": non_finite,
    }


def run_oscillating_episodes(csv, n_episodes, days, seed):
    """Same as run_forced_action_episodes but alternates +1.0/-1.0 every
    step instead of holding a fixed action -- stresses numerical stability
    of the thermal integrator under the most abrupt possible direction
    reversal, not just sustained extremes.
    """
    max_steps = days * 96
    min_soc_seen = 1.0
    max_soc_seen = 0.0
    max_t_cell_seen = 0.0
    violations = 0
    non_finite = 0
    n_steps_total = 0

    for ep in range(n_episodes):
        env = VoltFlowEnv(csv_path=csv, max_steps=max_steps, seed=seed + ep)
        obs, _ = env.reset(options={"randomize": True})
        for step_idx in range(max_steps):
            a = 1.0 if step_idx % 2 == 0 else -1.0
            action = np.array([a], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            n_steps_total += 1
            soc = info.get("soc", float("nan"))
            t_cell = info.get("t_cell_k", float("nan"))

            if not (math.isfinite(soc) and math.isfinite(t_cell) and math.isfinite(reward)):
                non_finite += 1
                continue

            min_soc_seen = min(min_soc_seen, soc)
            max_soc_seen = max(max_soc_seen, soc)
            max_t_cell_seen = max(max_t_cell_seen, t_cell)

            if soc < SOC_MIN - 1e-4 or soc > SOC_MAX + 1e-4:
                violations += 1
            if t_cell > T_CRIT_K + 1e-3:
                violations += 1

            if term or trunc:
                break

    return {
        "n_episodes": n_episodes,
        "n_steps_total": n_steps_total,
        "min_soc_seen": min_soc_seen,
        "max_soc_seen": max_soc_seen,
        "max_t_cell_seen": max_t_cell_seen,
        "violations": violations,
        "non_finite": non_finite,
    }


def run_price_spike_response(csv, ppo_model_path, n_episodes, days, seed, tail_pct=99.0):
    """Runs the trained policy (not forced actions) and checks its behavior
    at the extreme price tail (top `100 - tail_pct` percent of observed
    prices), not just Gate 3's p75. Recovers the *raw* price by re-deriving
    normalization bounds from this run's own observed price range rather
    than hardcoding -50/300 (see verify_gate3.py's denormalize_price for
    the version that hardcodes stale bounds -- percentile grouping is
    invariant to that under affine rescaling, so it doesn't invalidate
    verify_gate3.py's PASS/FAIL, but this script avoids the issue directly
    since it only needs relative ordering, not an absolute EUR/MWh number).
    """
    from stable_baselines3 import PPO

    max_steps = days * 96
    model = PPO.load(ppo_model_path)

    all_price_norm = []
    all_actions = []

    for ep in range(n_episodes):
        env = VoltFlowEnv(csv_path=csv, max_steps=max_steps, seed=seed + ep)
        obs, _ = env.reset(options={"randomize": True})
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            all_price_norm.append(float(obs[4]))
            all_actions.append(float(np.clip(action[0], -1.0, 1.0)))
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

    price_norm = np.array(all_price_norm)
    actions = np.array(all_actions)

    p75 = np.percentile(price_norm, 75.0)
    tail_threshold = np.percentile(price_norm, tail_pct)

    above_p75 = actions[price_norm > p75]
    in_tail = actions[price_norm > tail_threshold]

    return {
        "n_steps": len(price_norm),
        "mean_action_above_p75": float(above_p75.mean()) if len(above_p75) else float("nan"),
        "n_tail_steps": int(len(in_tail)),
        "mean_action_in_tail": float(in_tail.mean()) if len(in_tail) else float("nan"),
        "frac_charging_in_tail": float((in_tail > 0.0).mean()) if len(in_tail) else float("nan"),
        "tail_pct": tail_pct,
    }


def fmt_forced(name, r):
    status = "PASS" if r["violations"] == 0 and r["non_finite"] == 0 else "FAIL"
    lines = [
        f"### {name}: {status}",
        f"- Episodes: {r['n_episodes']} (random-start, sampled across the full dataset)",
        f"- Steps evaluated: {r['n_steps_total']}",
        f"- SoC range observed: [{r['min_soc_seen']:.4f}, {r['max_soc_seen']:.4f}] "
        f"(hard bounds: [{SOC_MIN}, {SOC_MAX}])",
        f"- Max cell temperature observed: {r['max_t_cell_seen']:.2f} K "
        f"(T_crit: {T_CRIT_K} K, margin: {T_CRIT_K - r['max_t_cell_seen']:.2f} K)",
        f"- Hard bound violations: {r['violations']}",
        f"- Non-finite (NaN/Inf) states: {r['non_finite']}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/energy_weather_spain.csv")
    parser.add_argument("--ppo-model", type=str, default=None,
                         help="If given, also runs the price-spike-response scenario "
                              "using this checkpoint. Skipped otherwise.")
    parser.add_argument("--n-episodes", type=int, default=30,
                         help="Random-start episodes per forced-action scenario.")
    parser.add_argument("--days", type=int, default=1,
                         help="Episode length in days for forced-action scenarios "
                              "(short episodes at extreme actions saturate quickly; "
                              "longer episodes cost more compute for similar signal).")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", type=str, default="results/stress_test.md")
    args = parser.parse_args()

    print(f"Running stress test against {args.csv}")
    print(f"({args.n_episodes} episodes x {args.days} day(s) per forced-action scenario)\n")

    discharge = run_forced_action_episodes(
        args.csv, -1.0, args.n_episodes, args.days, args.seed)
    print(fmt_forced("Sustained max discharge (action=-1.0 every step)", discharge))

    charge = run_forced_action_episodes(
        args.csv, 1.0, args.n_episodes, args.days, args.seed + 1000)
    print(fmt_forced("Sustained max charge (action=+1.0 every step)", charge))

    oscillating = run_oscillating_episodes(
        args.csv, args.n_episodes, args.days, args.seed + 2000)
    print(fmt_forced("Oscillating extreme (+1.0/-1.0 alternating every step)", oscillating))

    report_lines = [
        "# VoltFlow Stress Test Results",
        "",
        f"CSV: `{args.csv}`",
        "",
        "Forces adversarial actions through the live simulation loop (not just "
        "unit-testing the clamp functions in isolation) to verify the SoC and "
        "thermal safety envelope holds under worst-case conditions, sampled "
        "across the full dataset rather than one cherry-picked window.",
        "",
        fmt_forced("Sustained max discharge (action=-1.0 every step)", discharge),
        fmt_forced("Sustained max charge (action=+1.0 every step)", charge),
        fmt_forced("Oscillating extreme (+1.0/-1.0 alternating every step)", oscillating),
    ]

    if args.ppo_model:
        spike = run_price_spike_response(
            args.csv, args.ppo_model, args.n_episodes, args.days, args.seed + 3000)
        print(f"### Price spike response (trained policy, {args.ppo_model}): "
              f"{'PASS' if spike['mean_action_in_tail'] <= 0.0 else 'FAIL'}")
        print(f"- Steps evaluated: {spike['n_steps']}")
        print(f"- Mean action above p75 (Gate 3's threshold): "
              f"{spike['mean_action_above_p75']:+.4f}")
        print(f"- Steps in top {100 - spike['tail_pct']:.0f}% price tail: "
              f"{spike['n_tail_steps']}")
        print(f"- Mean action in extreme tail: {spike['mean_action_in_tail']:+.4f}")
        print(f"- Fraction charging in extreme tail: "
              f"{100*spike['frac_charging_in_tail']:.1f}%\n")

        report_lines += [
            f"### Price spike response (trained policy, `{args.ppo_model}`): "
            f"{'PASS' if spike['mean_action_in_tail'] <= 0.0 else 'FAIL'}",
            f"- Steps evaluated: {spike['n_steps']}",
            f"- Mean action above p75 (Gate 3's threshold): "
            f"{spike['mean_action_above_p75']:+.4f}",
            f"- Steps in top {100 - spike['tail_pct']:.0f}% price tail: "
            f"{spike['n_tail_steps']}",
            f"- Mean action in extreme tail: {spike['mean_action_in_tail']:+.4f}",
            f"- Fraction charging in extreme tail: "
            f"{100*spike['frac_charging_in_tail']:.1f}%",
            "",
            "Checks whether the agent's price-threshold behavior (Gate 3) holds at "
            "genuine tail prices, not just the moderate p75 threshold Gate 3 itself "
            "checks -- i.e. does the learned policy generalize, or does it only "
            "work in the price regime it happened to see most during training.",
            "",
        ]
    else:
        print("(--ppo-model not given, skipping price-spike-response scenario)")

    all_passed = (
        discharge["violations"] == 0 and discharge["non_finite"] == 0
        and charge["violations"] == 0 and charge["non_finite"] == 0
        and oscillating["violations"] == 0 and oscillating["non_finite"] == 0
    )

    with open(args.out, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nWritten to {args.out}")

    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()