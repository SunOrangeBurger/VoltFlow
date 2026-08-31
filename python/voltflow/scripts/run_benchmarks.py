"""Gate 4: runs simulation episodes comparing Heuristic vs. VoltFlow RL on a
given CSV (intended to be a held-out eval-fold CSV, see make_cv_folds.py) and
outputs a Markdown summary table.

Runs `--n-episodes` non-overlapping windows within the given CSV rather than
a single fixed slice, and reports mean +/- std net PnL, so the eval number
isn't itself a single lucky (or unlucky) sample.

Usage:
    python python/voltflow/scripts/run_benchmarks.py \\
        --csv data/cv/fold3_eval.csv \\
        --ppo-model models/ppo_voltflow_fold3_seed1.zip \\
        --seed 123 \\
        --n-episodes 5 \\
        --out benchmark_results_fold3_seed1.md
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from voltflow.envs.gym_wrapper import VoltFlowEnv
from voltflow.models.baselines import ThresholdRuleBaseline, TouHeuristicBaseline, run_episode


class PpoPolicyAdapter:
    """Wraps an SB3 PPO model so it exposes the same `.act(obs)` interface
    as the heuristic baselines."""

    def __init__(self, model):
        self.model = model

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/energy_weather_spain.csv")
    parser.add_argument("--ppo-model", type=str, default="models/ppo_voltflow.zip")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--n-episodes", type=int, default=5,
                         help="Number of non-overlapping windows to evaluate "
                              "within --csv; reports mean +/- std.")
    parser.add_argument("--seed", type=int, default=123,
                         help="Eval env seed. Independent of whatever seed "
                              "the PPO model was trained with.")
    parser.add_argument("--out", type=str, default="benchmark_results.md")
    args = parser.parse_args()

    max_steps = args.days * 96  # 96 steps/day at 15-min resolution

    policies = {
        "Threshold Rule Heuristic": ThresholdRuleBaseline(),
        "TOU Heuristic": TouHeuristicBaseline(),
    }

    if os.path.exists(args.ppo_model):
        from stable_baselines3 import PPO

        model = PPO.load(args.ppo_model)
        policies["VoltFlow RL (PPO)"] = PpoPolicyAdapter(model)
    else:
        print(
            f"WARNING: no PPO model found at {args.ppo_model}. "
            "Run train_ppo.py first for a complete comparison. "
            "Proceeding with heuristics only."
        )

    # Run n_episodes independent windows per policy, each with its own env
    # instance and a distinct seed offset so windows don't overlap/repeat.
    per_policy_runs = {name: [] for name in policies}
    for ep_idx in range(args.n_episodes):
        env = VoltFlowEnv(csv_path=args.csv, max_steps=max_steps, seed=args.seed + ep_idx)
        for name, policy in policies.items():
            per_policy_runs[name].append(run_episode(env, policy, max_steps=max_steps))

    def agg(key, runs):
        vals = np.array([r[key] for r in runs], dtype=np.float64)
        return vals.mean(), vals.std()

    results = {}
    for name, runs in per_policy_runs.items():
        pnl_mean, pnl_std = agg("net_pnl", runs)
        rev_mean, rev_std = agg("total_revenue", runs)
        deg_mean, deg_std = agg("total_degradation_cost", runs)
        soh_mean, _ = agg("final_soh", runs)
        rew_mean, rew_std = agg("total_reward", runs)
        results[name] = {
            "net_pnl_mean": pnl_mean, "net_pnl_std": pnl_std,
            "revenue_mean": rev_mean, "revenue_std": rev_std,
            "deg_mean": deg_mean, "deg_std": deg_std,
            "soh_mean": soh_mean,
            "reward_mean": rew_mean, "reward_std": rew_std,
        }

    lines = [
        f"# VoltFlow Benchmark Results ({args.days}-day episodes, "
        f"n={args.n_episodes}, csv={args.csv})",
        "",
        "| Strategy | Net PnL ($) mean±std | Revenue ($) mean±std | "
        "Degradation ($) mean±std | Final SoH | Reward mean±std |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['net_pnl_mean']:.2f}±{r['net_pnl_std']:.2f} | "
            f"{r['revenue_mean']:.2f}±{r['revenue_std']:.2f} | "
            f"{r['deg_mean']:.2f}±{r['deg_std']:.2f} | {r['soh_mean']:.4f} | "
            f"{r['reward_mean']:.4f}±{r['reward_std']:.4f} |"
        )

    if "VoltFlow RL (PPO)" in results and len(results) > 1:
        best_heuristic = max(
            (v["net_pnl_mean"] for k, v in results.items() if k != "VoltFlow RL (PPO)"),
            default=0.0,
        )
        rl_pnl = results["VoltFlow RL (PPO)"]["net_pnl_mean"]
        if best_heuristic != 0:
            improvement = (rl_pnl - best_heuristic) / abs(best_heuristic) * 100
            lines.append("")
            lines.append(
                f"**RL vs. best heuristic net PnL improvement: {improvement:.1f}%** "
                f"(Gate target: >= 15%, mean over {args.n_episodes} held-out episodes)"
            )

    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()