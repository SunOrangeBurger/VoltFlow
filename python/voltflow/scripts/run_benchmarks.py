"""Gate 4: runs a 7-day simulation comparing Heuristic vs. VoltFlow RL and
outputs a Markdown summary table.

Usage:
    python python/voltflow/scripts/run_benchmarks.py \\
        --csv data/raw/energy_weather_spain.csv \\
        --ppo-model models/ppo_voltflow.zip
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
    parser.add_argument("--out", type=str, default="benchmark_results.md")
    args = parser.parse_args()

    max_steps = args.days * 96  # 96 steps/day at 15-min resolution
    env = VoltFlowEnv(csv_path=args.csv, max_steps=max_steps, seed=123)

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

    results = {}
    for name, policy in policies.items():
        results[name] = run_episode(env, policy, max_steps=max_steps)

    lines = [
        f"# VoltFlow Benchmark Results ({args.days}-day simulation)",
        "",
        "| Strategy | Net PnL ($) | Total Revenue ($) | Degradation Cost ($) | Final SoH | Total Reward |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['net_pnl']:.2f} | {r['total_revenue']:.2f} | "
            f"{r['total_degradation_cost']:.2f} | {r['final_soh']:.4f} | {r['total_reward']:.4f} |"
        )

    if "VoltFlow RL (PPO)" in results and len(results) > 1:
        best_heuristic = max(
            (v["net_pnl"] for k, v in results.items() if k != "VoltFlow RL (PPO)"),
            default=0.0,
        )
        rl_pnl = results["VoltFlow RL (PPO)"]["net_pnl"]
        if best_heuristic != 0:
            improvement = (rl_pnl - best_heuristic) / abs(best_heuristic) * 100
            lines.append("")
            lines.append(
                f"**RL vs. best heuristic net PnL improvement: {improvement:.1f}%** "
                f"(Gate target: >= 15%)"
            )

    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
