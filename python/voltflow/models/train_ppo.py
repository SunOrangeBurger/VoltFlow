"""PPO training pipeline for the VoltFlow BESS arbitrage agent.

Usage:
    python -m voltflow.models.train_ppo --csv data/raw/energy_weather_spain.csv \\
        --timesteps 2000000 --out models/ppo_voltflow

NOTE: 2,000,000 timesteps (spec Phase 4) takes on the order of hours on CPU,
longer with the default MlpPolicy on GPU-less machines. Start with
--timesteps 50000 for a smoke test before committing to the full run.
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from voltflow.envs.gym_wrapper import VoltFlowEnv


def make_env(csv_path: str, max_steps: int, seed: int):
    def _init():
        env = VoltFlowEnv(csv_path=csv_path, max_steps=max_steps, seed=seed)
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train VoltFlow PPO agent")
    parser.add_argument("--csv", type=str, default="data/raw/energy_weather_spain.csv")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models/ppo_voltflow")
    parser.add_argument("--tensorboard-log", type=str, default="logs/tensorboard")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    env_fns = [make_env(args.csv, args.max_steps, args.seed + i) for i in range(args.n_envs)]
    if args.n_envs == 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv(env_fns)
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        vec_env = SubprocVecEnv(env_fns)

    eval_env = Monitor(VoltFlowEnv(csv_path=args.csv, max_steps=args.max_steps, seed=args.seed + 999))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.out) or ".",
        log_path=args.tensorboard_log,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)
    model.save(args.out)
    print(f"Model saved to {args.out}.zip")


if __name__ == "__main__":
    main()
