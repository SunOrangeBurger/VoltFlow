"""Gate 3 verification: does the trained agent learn to halt charging (idle
or discharge) once spot prices exceed the ~75th percentile?

This is a direct behavioral check, distinct from the PnL-based Gate 4
benchmark -- a policy could show strong PnL for reasons other than sensible
price-threshold behavior (e.g. exploiting some other regularity in the
data), so this checks the specific claim in spec Gate 3 directly.

Method:
  1. Load the trained PPO checkpoint.
  2. Step through the given CSV (recommend a held-out eval-fold CSV) with
     the deterministic policy, recording (raw_price, action) at every step.
  3. Recover the raw price from obs[4] (spec 5.1 inverse normalization),
     rather than needing a separate Rust accessor.
  4. Compute the 75th percentile of observed raw prices across the run.
  5. Assert: mean(action | price > p75) <= --idle-threshold (default 0.0,
     i.e. idle or discharging, not charging).

Usage:
    python python/voltflow/scripts/verify_gate3.py \
        --csv data/cv/fold3_eval.csv \
        --ppo-model models/cv/ppo_voltflow_fold3_seed4.zip \
        --episodes 10
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from voltflow.envs.gym_wrapper import VoltFlowEnv

# Must match spec section 5.1's obs[4] normalization exactly:
#   obs4 = (clamp(price, -50, 300) + 50) / 350.0
PRICE_CLAMP_MIN = -50.0
PRICE_CLAMP_MAX = 300.0
PRICE_NORM_SPAN = 350.0


def denormalize_price(obs4: float) -> float:
    return obs4 * PRICE_NORM_SPAN - 50.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                         help="Recommend a held-out eval-fold CSV, e.g. data/cv/fold3_eval.csv")
    parser.add_argument("--ppo-model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10,
                         help="Number of non-overlapping 7-day episodes to pool for the percentile check.")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--percentile", type=float, default=75.0)
    parser.add_argument("--idle-threshold", type=float, default=0.0,
                         help="Mean action above this percentile must be <= this value "
                              "(0.0 = idle-or-discharge, i.e. not net charging).")
    args = parser.parse_args()

    max_steps = args.days * 96
    model = PPO.load(args.ppo_model)

    all_prices = []
    all_actions = []

    for ep in range(args.episodes):
        env = VoltFlowEnv(csv_path=args.csv, max_steps=max_steps, seed=args.seed + ep)
        obs, _ = env.reset(seed=args.seed + ep, options={"randomize": True})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            price = denormalize_price(float(obs[4]))
            all_prices.append(price)
            all_actions.append(float(np.clip(action[0], -1.0, 1.0)))
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

    prices = np.array(all_prices)
    actions = np.array(all_actions)

    threshold_price = np.percentile(prices, args.percentile)
    above_mask = prices > threshold_price
    n_above = above_mask.sum()

    if n_above == 0:
        print(
            f"No steps observed above the {args.percentile}th percentile price "
            f"({threshold_price:.2f}) -- cannot evaluate. Try more --episodes."
        )
        return

    mean_action_above = actions[above_mask].mean()
    mean_action_below = actions[~above_mask].mean()
    frac_charging_above = (actions[above_mask] > 0.0).mean()

    print(f"CSV: {args.csv}")
    print(f"Model: {args.ppo_model}")
    print(f"Steps pooled: {len(prices)} across {args.episodes} episodes")
    print(f"{args.percentile}th percentile price: {threshold_price:.2f} EUR/MWh")
    print(f"Steps above threshold: {n_above} ({100*n_above/len(prices):.1f}% of total)")
    print(f"Mean action when price > p{args.percentile:.0f}: {mean_action_above:+.4f} "
          f"(negative/zero = idle-or-discharge, positive = charging)")
    print(f"Mean action when price <= p{args.percentile:.0f}: {mean_action_below:+.4f}")
    print(f"Fraction of above-threshold steps where agent charges (action > 0): "
          f"{100*frac_charging_above:.1f}%")

    passed = mean_action_above <= args.idle_threshold
    print()
    if passed:
        print(f"GATE 3: PASS -- mean action above p{args.percentile:.0f} "
              f"({mean_action_above:+.4f}) <= threshold ({args.idle_threshold:+.4f})")
    else:
        print(f"GATE 3: FAIL -- mean action above p{args.percentile:.0f} "
              f"({mean_action_above:+.4f}) > threshold ({args.idle_threshold:+.4f})")
        print("The agent is, on average, still charging even at high prices.")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()