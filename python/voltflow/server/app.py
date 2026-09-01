"""FastAPI WebSocket telemetry server for VoltFlow live dashboard.

At startup, benchmarks every available PPO checkpoint under `models/` (and
`models/cv/`) against the heuristic baselines on held-out fold data, and
loads whichever checkpoint wins on net PnL (see `voltflow.models.model_selection`).
No hardcoded "best" checkpoint path — it's re-decided empirically every time
the server starts, so a new checkpoint dropped into `models/cv/` is picked
up automatically on the next restart.

Then runs three policies in parallel on identical, synchronized episodes
(same CSV window, same starting step) every tick: the selected PPO policy,
the Threshold-Rule heuristic, and the TOU heuristic. This gives the frontend
a live, apples-to-apples revenue/cost/PnL comparison rather than only a
static offline benchmark number.

Run with:
    uvicorn voltflow.server.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voltflow.envs.gym_wrapper import VoltFlowEnv
from voltflow.models.baselines import ThresholdRuleBaseline, TouHeuristicBaseline
from voltflow.models.model_selection import SelectionReport, select_best_model

app = FastAPI(title="VoltFlow Telemetry Server")

CSV_PATH = os.environ.get("VOLTFLOW_CSV_PATH", "data/raw/energy_weather_spain.csv")
MODELS_DIR = os.environ.get("VOLTFLOW_MODELS_DIR", "models")
# Manual override: if set, skip startup benchmarking and load this checkpoint
# directly (useful for fast iteration). Leave unset for normal "pick the best
# model automatically" behavior.
PPO_MODEL_OVERRIDE = os.environ.get("VOLTFLOW_PPO_MODEL")
STEP_INTERVAL_SECONDS = float(os.environ.get("VOLTFLOW_STEP_INTERVAL", "0.5"))
MAX_STEPS = int(os.environ.get("VOLTFLOW_MAX_STEPS", "96"))
SELECTION_EPISODES = int(os.environ.get("VOLTFLOW_SELECTION_EPISODES", "3"))

_connected_clients: set[WebSocket] = set()
_policy = None
_policy_label: Optional[str] = None
_selection_report: Optional[SelectionReport] = None


class _PpoAdapter:
    def __init__(self, model):
        self.model = model

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def _load_policy():
    """Loads the PPO policy to drive the live simulation: either the
    manually-overridden checkpoint, or whichever checkpoint wins the startup
    benchmark sweep. Falls back to an idle policy if neither is available."""
    global _policy, _policy_label, _selection_report

    if PPO_MODEL_OVERRIDE:
        if os.path.exists(PPO_MODEL_OVERRIDE):
            try:
                from stable_baselines3 import PPO

                _policy = _PpoAdapter(PPO.load(PPO_MODEL_OVERRIDE))
                _policy_label = os.path.splitext(os.path.basename(PPO_MODEL_OVERRIDE))[0]
                print(f"VoltFlow server: loaded override checkpoint {PPO_MODEL_OVERRIDE}")
                return
            except Exception as e:  # noqa: BLE001
                print(f"VoltFlow server: failed to load override checkpoint ({e}); falling back to auto-selection.")
        else:
            print(f"VoltFlow server: VOLTFLOW_PPO_MODEL={PPO_MODEL_OVERRIDE} not found; falling back to auto-selection.")

    _selection_report = select_best_model(models_dir=MODELS_DIR, max_steps=MAX_STEPS, n_episodes=SELECTION_EPISODES)
    if _selection_report.winner is not None:
        try:
            from stable_baselines3 import PPO

            _policy = _PpoAdapter(PPO.load(_selection_report.winner.path))
            _policy_label = _selection_report.winner.label
        except Exception as e:  # noqa: BLE001
            print(f"VoltFlow server: failed to load selected checkpoint ({e}); using idle policy.")
            _policy = None
            _policy_label = None
    else:
        _policy = None
        _policy_label = None


class _IdlePolicy:
    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


async def _simulation_loop():
    """Background task: steps three parallel envs (PPO-best, Threshold
    heuristic, TOU heuristic) on identical synchronized episodes forever,
    broadcasting a combined telemetry frame each tick."""
    if not os.path.exists(CSV_PATH):
        print(
            f"VoltFlow server: CSV not found at {CSV_PATH}. "
            "Generate one with generate_synthetic_data.py or download_data.py "
            "before starting the server."
        )
        return

    _load_policy()

    strategies = {
        "ppo": _policy if _policy is not None else _IdlePolicy(),
        "threshold": ThresholdRuleBaseline(),
        "tou": TouHeuristicBaseline(),
    }

    episode_seed = 7

    def make_envs(seed: int):
        envs = {
            name: VoltFlowEnv(csv_path=CSV_PATH, max_steps=MAX_STEPS, seed=seed)
            for name in strategies
        }
        obs = {}
        for name, env in envs.items():
            # options={"randomize": True} on a fixed seed picks the same
            # random start window across all three envs, so they run the
            # exact same slice of the dataset in parallel.
            o, _ = env.reset(options={"randomize": True})
            obs[name] = o
        return envs, obs

    envs, obs = make_envs(episode_seed)
    cumulative = {name: {"pnl": 0.0, "revenue": 0.0, "degradation": 0.0} for name in strategies}
    step_count = 0

    while True:
        frame_by_strategy = {}
        for name, policy in strategies.items():
            action = policy.act(obs[name])
            o, reward, term, trunc, info = envs[name].step(action)
            obs[name] = o

            revenue = info.get("revenue", 0.0)
            degradation = info.get("degradation_cost", 0.0)
            cumulative[name]["revenue"] += revenue
            cumulative[name]["degradation"] += degradation
            cumulative[name]["pnl"] += revenue - degradation

            frame_by_strategy[name] = {
                "soc": info.get("soc"),
                "soh": info.get("soh"),
                "t_cell_c": info.get("t_cell_k", 273.15) - 273.15,
                "price": info.get("price"),
                "revenue": revenue,
                "degradation_cost": degradation,
                "thermal_penalty": info.get("thermal_penalty"),
                "thermal_interlock_active": info.get("thermal_interlock_active"),
                "cumulative_revenue": cumulative[name]["revenue"],
                "cumulative_degradation": cumulative[name]["degradation"],
                "cumulative_pnl": cumulative[name]["pnl"],
                "reward": reward,
                "term": term,
                "trunc": trunc,
            }

        step_count += 1

        best_heuristic_pnl = max(
            cumulative["threshold"]["pnl"], cumulative["tou"]["pnl"]
        )
        ppo_pnl = cumulative["ppo"]["pnl"]
        live_improvement_pct = (
            (ppo_pnl - best_heuristic_pnl) / abs(best_heuristic_pnl) * 100
            if best_heuristic_pnl != 0
            else None
        )

        payload = {
            "step": step_count,
            "policy_label": _policy_label or "idle (no PPO checkpoint loaded)",
            # top-level fields mirror the PPO strategy for backward
            # compatibility with any client only reading flat fields.
            **{k: v for k, v in frame_by_strategy["ppo"].items() if k not in ("term", "trunc")},
            "strategies": frame_by_strategy,
            "live_improvement_pct": live_improvement_pct,
            "startup_selection": _selection_report.to_dict() if _selection_report else None,
        }
        await _broadcast(payload)

        any_done = any(frame_by_strategy[name]["term"] or frame_by_strategy[name]["trunc"] for name in strategies)
        if any_done:
            episode_seed += 1
            envs, obs = make_envs(episode_seed)
            cumulative = {name: {"pnl": 0.0, "revenue": 0.0, "degradation": 0.0} for name in strategies}
            step_count = 0

        await asyncio.sleep(STEP_INTERVAL_SECONDS)


async def _broadcast(payload: dict):
    if not _connected_clients:
        return
    message = json.dumps(payload)
    stale = set()
    for client in _connected_clients:
        try:
            await client.send_text(message)
        except Exception:  # noqa: BLE001
            stale.add(client)
    _connected_clients.difference_update(stale)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_simulation_loop())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "csv_path": CSV_PATH,
        "ppo_loaded": _policy is not None,
        "policy_label": _policy_label,
        "startup_selection": _selection_report.to_dict() if _selection_report else None,
    }


@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    _connected_clients.add(websocket)
    try:
        while True:
            # Keep the connection open; clients don't need to send anything.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _connected_clients.discard(websocket)