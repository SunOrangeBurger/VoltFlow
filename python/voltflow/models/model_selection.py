"""Startup model selection: benchmark every available PPO checkpoint against
the heuristic baselines on held-out data, and pick the one with the best net
PnL. Used by the telemetry server so "best model" is chosen empirically each
time the backend starts, rather than hardcoded to a single checkpoint path.

Reuses the same episode-running logic as `scripts/run_benchmarks.py` /
`models/baselines.py` so the number reported here is directly comparable to
the offline benchmark reports in `results/`.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from voltflow.envs.gym_wrapper import VoltFlowEnv
from voltflow.models.baselines import ThresholdRuleBaseline, TouHeuristicBaseline, run_episode

# fold-tagged checkpoints (ppo_voltflow_fold{N}_seed{M}.zip) are evaluated on
# their own fold's held-out eval CSV, since that's the only split each was
# never trained on. Untagged checkpoints (ppo_voltflow.zip, best_model.zip)
# fall back to this CSV.
DEFAULT_EVAL_CSV = "data/raw/energy_weather_spain.csv"
FOLD_EVAL_CSV_TEMPLATE = "data/cv/{fold}_eval.csv"


@dataclass
class CheckpointResult:
    path: str
    label: str
    eval_csv: str
    net_pnl_mean: float
    net_pnl_std: float
    revenue_mean: float
    degradation_mean: float
    best_heuristic_pnl: float
    improvement_pct: float
    error: str | None = None


@dataclass
class SelectionReport:
    results: list[CheckpointResult] = field(default_factory=list)
    winner: CheckpointResult | None = None

    def to_dict(self) -> dict:
        return {
            "winner": self.winner.path if self.winner else None,
            "winner_label": self.winner.label if self.winner else None,
            "winner_net_pnl": self.winner.net_pnl_mean if self.winner else None,
            "winner_improvement_pct": self.winner.improvement_pct if self.winner else None,
            "candidates": [
                {
                    "path": r.path,
                    "label": r.label,
                    "eval_csv": r.eval_csv,
                    "net_pnl_mean": r.net_pnl_mean,
                    "net_pnl_std": r.net_pnl_std,
                    "revenue_mean": r.revenue_mean,
                    "degradation_mean": r.degradation_mean,
                    "best_heuristic_pnl": r.best_heuristic_pnl,
                    "improvement_pct": r.improvement_pct,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def _discover_checkpoints(models_dir: str) -> list[str]:
    patterns = [
        os.path.join(models_dir, "*.zip"),
        os.path.join(models_dir, "cv", "*.zip"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    # de-dupe while preserving order (best_model.zip is duplicated under cv/)
    seen = set()
    unique = []
    for p in paths:
        key = os.path.abspath(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _eval_csv_for_checkpoint(path: str) -> str:
    name = os.path.basename(path)
    for fold in ("fold1", "fold2", "fold3"):
        if fold in name:
            candidate = FOLD_EVAL_CSV_TEMPLATE.format(fold=fold)
            if os.path.exists(candidate):
                return candidate
    return DEFAULT_EVAL_CSV


def _benchmark_heuristics(csv_path: str, max_steps: int, n_episodes: int, seed: int) -> float:
    """Returns the best (highest) mean net PnL among the heuristic baselines
    on this CSV, for use as the comparison bar."""
    heuristics = {
        "Threshold Rule": ThresholdRuleBaseline(),
        "TOU Heuristic": TouHeuristicBaseline(),
    }
    best = float("-inf")
    for policy in heuristics.values():
        pnls = []
        for ep_idx in range(n_episodes):
            env = VoltFlowEnv(csv_path=csv_path, max_steps=max_steps, seed=seed + ep_idx)
            pnls.append(run_episode(env, policy, max_steps=max_steps)["net_pnl"])
        mean_pnl = float(np.mean(pnls))
        best = max(best, mean_pnl)
    return best


class _PpoAdapter:
    def __init__(self, model):
        self.model = model

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def select_best_model(
    models_dir: str = "models",
    max_steps: int = 96,
    n_episodes: int = 3,
    seed: int = 123,
) -> SelectionReport:
    """Benchmarks every checkpoint under `models_dir` (and `models_dir/cv`)
    against the heuristic baselines on held-out fold data, and returns a
    report including the winning checkpoint path.

    Falls back gracefully: if stable_baselines3 isn't importable, or no
    checkpoints are found, or every candidate fails to load, `report.winner`
    is None and the caller should fall back to an idle policy.
    """
    report = SelectionReport()

    try:
        from stable_baselines3 import PPO
    except Exception as e:  # noqa: BLE001
        print(f"VoltFlow model selection: stable_baselines3 unavailable ({e}); skipping.")
        return report

    checkpoints = _discover_checkpoints(models_dir)
    if not checkpoints:
        print(f"VoltFlow model selection: no .zip checkpoints found under {models_dir}/.")
        return report

    print(f"VoltFlow model selection: benchmarking {len(checkpoints)} checkpoint(s)...")

    # Cache heuristic bars per eval CSV so we don't re-run them per checkpoint.
    heuristic_bar_cache: dict[str, float] = {}

    best_result: CheckpointResult | None = None

    for path in checkpoints:
        stem = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(path))
        # Disambiguate same-named checkpoints in different dirs (e.g. both
        # models/best_model.zip and models/cv/best_model.zip exist and are
        # different files).
        label = f"{parent}/{stem}" if parent and parent != os.path.basename(models_dir) else stem
        eval_csv = _eval_csv_for_checkpoint(path)

        if not os.path.exists(eval_csv):
            eval_csv = DEFAULT_EVAL_CSV
        if not os.path.exists(eval_csv):
            result = CheckpointResult(
                path=path, label=label, eval_csv=eval_csv,
                net_pnl_mean=float("nan"), net_pnl_std=float("nan"),
                revenue_mean=float("nan"), degradation_mean=float("nan"),
                best_heuristic_pnl=float("nan"), improvement_pct=float("nan"),
                error=f"eval CSV not found: {eval_csv}",
            )
            report.results.append(result)
            continue

        try:
            model = PPO.load(path)
            adapter = _PpoAdapter(model)

            pnls, revs, degs = [], [], []
            for ep_idx in range(n_episodes):
                env = VoltFlowEnv(csv_path=eval_csv, max_steps=max_steps, seed=seed + ep_idx)
                r = run_episode(env, adapter, max_steps=max_steps)
                pnls.append(r["net_pnl"])
                revs.append(r["total_revenue"])
                degs.append(r["total_degradation_cost"])

            if eval_csv not in heuristic_bar_cache:
                heuristic_bar_cache[eval_csv] = _benchmark_heuristics(
                    eval_csv, max_steps, n_episodes, seed
                )
            best_heuristic = heuristic_bar_cache[eval_csv]

            net_pnl_mean = float(np.mean(pnls))
            net_pnl_std = float(np.std(pnls))
            improvement_pct = (
                (net_pnl_mean - best_heuristic) / abs(best_heuristic) * 100
                if best_heuristic != 0
                else float("nan")
            )

            result = CheckpointResult(
                path=path, label=label, eval_csv=eval_csv,
                net_pnl_mean=net_pnl_mean, net_pnl_std=net_pnl_std,
                revenue_mean=float(np.mean(revs)), degradation_mean=float(np.mean(degs)),
                best_heuristic_pnl=best_heuristic, improvement_pct=improvement_pct,
            )
            print(
                f"  {label}: net_pnl={net_pnl_mean:.2f}±{net_pnl_std:.2f} "
                f"(vs heuristic {best_heuristic:.2f}, {improvement_pct:+.1f}%)"
            )
        except Exception as e:  # noqa: BLE001
            result = CheckpointResult(
                path=path, label=label, eval_csv=eval_csv,
                net_pnl_mean=float("nan"), net_pnl_std=float("nan"),
                revenue_mean=float("nan"), degradation_mean=float("nan"),
                best_heuristic_pnl=float("nan"), improvement_pct=float("nan"),
                error=str(e),
            )
            print(f"  {label}: FAILED to load/evaluate ({e})")

        report.results.append(result)
        if result.error is None and (
            best_result is None or result.net_pnl_mean > best_result.net_pnl_mean
        ):
            best_result = result

    report.winner = best_result
    if best_result is not None:
        print(
            f"VoltFlow model selection: winner = {best_result.label} "
            f"(net_pnl={best_result.net_pnl_mean:.2f}, "
            f"{best_result.improvement_pct:+.1f}% over best heuristic)"
        )
    else:
        print("VoltFlow model selection: no checkpoint could be evaluated; falling back to idle policy.")

    return report