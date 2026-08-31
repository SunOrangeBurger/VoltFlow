"""FastAPI WebSocket telemetry server for VoltFlow live dashboard.

Runs a live simulation episode and streams state to any connected WebSocket
clients (the Next.js dashboard). Optionally drives the episode with a
trained PPO policy; otherwise uses a random/idle policy for smoke testing.

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

app = FastAPI(title="VoltFlow Telemetry Server")

CSV_PATH = os.environ.get("VOLTFLOW_CSV_PATH", "data/raw/energy_weather_spain.csv")
PPO_MODEL_PATH = os.environ.get("VOLTFLOW_PPO_MODEL", "models/ppo_voltflow.zip")
STEP_INTERVAL_SECONDS = float(os.environ.get("VOLTFLOW_STEP_INTERVAL", "0.5"))
MAX_STEPS = int(os.environ.get("VOLTFLOW_MAX_STEPS", "96"))

_connected_clients: set[WebSocket] = set()
_policy = None


def _load_policy():
    global _policy
    if _policy is not None:
        return _policy
    if os.path.exists(PPO_MODEL_PATH):
        try:
            from stable_baselines3 import PPO

            _policy = PPO.load(PPO_MODEL_PATH)
        except Exception as e:  # noqa: BLE001 - degrade gracefully to idle policy
            print(f"VoltFlow server: failed to load PPO model ({e}); using idle policy.")
            _policy = None
    return _policy


async def _simulation_loop():
    """Background task: steps the env forever, broadcasting telemetry."""
    if not os.path.exists(CSV_PATH):
        print(
            f"VoltFlow server: CSV not found at {CSV_PATH}. "
            "Generate one with generate_synthetic_data.py or download_data.py "
            "before starting the server."
        )
        return

    env = VoltFlowEnv(csv_path=CSV_PATH, max_steps=MAX_STEPS, seed=7)
    obs, _ = env.reset(options={"randomize": True})
    policy = _load_policy()
    cumulative_pnl = 0.0
    step_count = 0

    while True:
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.array([0.0], dtype=np.float32)  # idle policy fallback

        obs, reward, term, trunc, info = env.step(action)
        cumulative_pnl += info.get("revenue", 0.0) - info.get("degradation_cost", 0.0)
        step_count += 1

        payload = {
            "step": step_count,
            "soc": info.get("soc"),
            "soh": info.get("soh"),
            "t_cell_k": info.get("t_cell_k"),
            "t_cell_c": info.get("t_cell_k", 273.15) - 273.15,
            "price": info.get("price"),
            "revenue": info.get("revenue"),
            "degradation_cost": info.get("degradation_cost"),
            "thermal_penalty": info.get("thermal_penalty"),
            "cumulative_pnl": cumulative_pnl,
            "reward": reward,
        }
        await _broadcast(payload)

        if term or trunc:
            obs, _ = env.reset(options={"randomize": True})
            cumulative_pnl = 0.0
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
    return {"status": "ok", "csv_path": CSV_PATH, "ppo_loaded": _policy is not None}


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
