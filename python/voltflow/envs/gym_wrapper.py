import gymnasium as gym
from gymnasium import spaces
import numpy as np
import voltflow_core


class VoltFlowEnv(gym.Env):
    """Gymnasium adapter around the Rust `voltflow_core.RustBessEnv`.

    Observation: 8-element normalized Box, see TECHNICAL.md section 5.1.
    Action: 1-element continuous Box in [-1.0, 1.0], see TECHNICAL.md section 5.2.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path: str, max_steps: int = 96, seed: int = 42):
        # 96 steps = 24 hours at 15-minute intervals.
        super().__init__()
        self._rust_env = voltflow_core.RustBessEnv(csv_path, max_steps, seed)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self._last_info = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        randomize = True
        if options is not None:
            randomize = options.get("randomize", True)
        obs = self._rust_env.reset(randomize)
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        act = float(np.clip(action[0], -1.0, 1.0))
        obs, reward, term, trunc, info = self._rust_env.step(act)
        self._last_info = info
        return np.array(obs, dtype=np.float32), float(reward), term, trunc, info

    def get_state(self):
        """Non-stepping state accessor, used by the telemetry server."""
        return self._rust_env.get_state()
