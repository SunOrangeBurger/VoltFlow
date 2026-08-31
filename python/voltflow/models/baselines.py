"""Rule-based & Time-of-Use (TOU) heuristic baselines.

These give the industry-standard comparison point the PPO agent must beat
by >=15% net PnL (spec Phase 4 / Gate 3-adjacent benchmark requirement).
"""

from __future__ import annotations

import numpy as np


class ThresholdRuleBaseline:
    """Simple percentile-threshold arbitrage heuristic.

    Charges full power when spot price is below `charge_pct` percentile of
    the price distribution seen so far in the episode; discharges full power
    above `discharge_pct` percentile; idles otherwise.

    Uses the normalized price observation (index 4) directly, so no external
    price history is required beyond what the env already exposes.
    """

    def __init__(self, charge_pct: float = 0.25, discharge_pct: float = 0.75):
        self.charge_pct = charge_pct
        self.discharge_pct = discharge_pct

    def act(self, obs: np.ndarray) -> np.ndarray:
        price_norm = float(obs[4])  # already normalized to [0, 1]
        soc = float(obs[0])

        if price_norm <= self.charge_pct and soc < 0.9:
            action = 1.0
        elif price_norm >= self.discharge_pct and soc > 0.1:
            action = -1.0
        else:
            action = 0.0
        return np.array([action], dtype=np.float32)


class TouHeuristicBaseline:
    """Classic Time-of-Use tariff heuristic: charge during fixed off-peak
    hours, discharge during fixed on-peak hours, regardless of live price.

    Hour is reconstructed from the sin/cos encoding at obs indices 6/7.
    """

    def __init__(
        self,
        off_peak_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        on_peak_hours: tuple[int, ...] = (17, 18, 19, 20, 21),
    ):
        self.off_peak_hours = set(off_peak_hours)
        self.on_peak_hours = set(on_peak_hours)

    def act(self, obs: np.ndarray) -> np.ndarray:
        sin_h, cos_h = float(obs[6]), float(obs[7])
        angle = np.arctan2(sin_h, cos_h)
        hour = (angle / (2 * np.pi)) * 24.0
        hour = int(round(hour)) % 24
        soc = float(obs[0])

        if hour in self.off_peak_hours and soc < 0.9:
            action = 1.0
        elif hour in self.on_peak_hours and soc > 0.1:
            action = -1.0
        else:
            action = 0.0
        return np.array([action], dtype=np.float32)


def run_episode(env, policy, max_steps: int = 96) -> dict:
    """Runs one episode with a baseline policy object exposing `.act(obs)`.
    Returns a dict of summary metrics matching what run_benchmarks.py needs.
    """
    obs, _ = env.reset(options={"randomize": False})
    total_reward = 0.0
    total_revenue = 0.0
    total_degradation_cost = 0.0
    final_soh = 1.0

    for _ in range(max_steps):
        action = policy.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        total_revenue += info.get("revenue", 0.0)
        total_degradation_cost += info.get("degradation_cost", 0.0)
        final_soh = info.get("soh", final_soh)
        if term or trunc:
            break

    return {
        "total_reward": total_reward,
        "total_revenue": total_revenue,
        "total_degradation_cost": total_degradation_cost,
        "net_pnl": total_revenue - total_degradation_cost,
        "final_soh": final_soh,
    }
