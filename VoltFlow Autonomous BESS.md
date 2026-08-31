# VoltFlow: Autonomous BESS Arbitrage & Degradation Management System
## Technical Specification & Agent Implementation Blueprint

---

## 1. Executive System Overview & Scope

**VoltFlow** is an ultra-high-throughput Industrial Battery Energy Storage System (BESS) simulation and real-time dispatch optimization platform. 

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VOLTFLOW PLATFORM                             │
├──────────────────────────────┬──────────────────────────────────────────┤
│ Rust Core Engine             │ Non-linear battery electrochemistry,     │
│ (`voltflow_core`)            │ thermal lumped-capacitance physics,      │
│                              │ semi-empirical degradation, & stochastic │
│                              │ weather/pricing domain randomization.    │
├──────────────────────────────┼──────────────────────────────────────────┤
│ PyO3 / Maturin Bridge        │ Zero-copy C-ABI Gymnasium environment    │
│ (`voltflow_gym`)             │ executing at 5,000,000+ steps/second.    │
├──────────────────────────────┼──────────────────────────────────────────┤
│ AI Optimization Layer        │ PPO/SAC with hard thermodynamic safety   │
│ (`voltflow_ai`)              │ projections, benchmarked against         │
│                              │ industry-standard heuristics (Rule/TOU). │
├──────────────────────────────┼──────────────────────────────────────────┤
│ Telemetry & Web Dashboard    │ FastAPI WebSocket telemetry streaming    │
│ (`voltflow_ui`)              │ to a Next.js / Tailwind / Recharts UI.   │
└──────────────────────────────┴──────────────────────────────────────────┘
```

---

## 2. Locked Tech Stack & Dependency Matrix

The agent **must use exact versions** to prevent ABI breakage, dependency divergence, or PyO3 compilation errors.

### 2.1. Rust Toolchain & Dependencies (`Cargo.toml`)
* **Rust Edition:** `2021` (Stable 1.78+)
* **Crate Dependencies:**
  ```toml
  [package]
  name = "voltflow_core"
  version = "0.1.0"
  edition = "2021"

  [lib]
  name = "voltflow_core"
  crate-type = ["cdylib", "rlib"]

  [dependencies]
  pyo3 = { version = "0.22.2", features = ["extension-module"] }
  numpy = "0.22.0"
  serde = { version = "1.0.208", features = ["derive"] }
  serde_json = "1.0.125"
  csv = "1.3.0"
  rand = "0.8.5"
  rand_distr = "0.4.3"
  rayon = "1.10.0"

  [profile.release]
  opt-level = 3
  lto = "fat"
  codegen-units = 1
  panic = "abort"
  ```

### 2.2. Python & RL Toolchain (`pyproject.toml` / `requirements.txt`)
* **Python Version:** `3.10` or `3.11`
* **Core Packages:**
  ```text
  maturin>=1.7.0,<2.0.0
  gymnasium==0.29.1
  stable-baselines3==2.3.2
  torch==2.4.0
  numpy>=1.26.4,<2.0.0
  pandas==2.2.2
  fastapi==0.112.0
  uvicorn[standard]==0.30.5
  websockets==12.0
  requests==2.32.3
  pydantic==2.8.2
  ```

### 2.3. Frontend Dashboard Toolchain
* **Framework:** Next.js 14 (App Router) + TypeScript + Tailwind CSS
* **UI Components & Charts:** `lucide-react`, `recharts`, `clsx`, `tailwind-merge`

---

## 3. Project Directory Architecture

```
voltflow/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── data/
│   └── raw/
│       └── energy_weather_spain.csv
├── crates/
│   └── voltflow_core/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs              # PyO3 module bindings
│           ├── battery/
│           │   ├── mod.rs
│           │   ├── cell.rs         # Cell chemistry & SoC coulomb counting
│           │   ├── thermal.rs      # Lumped-capacitance heat dissipation
│           │   └── degradation.rs  # Rainflow cycle & calendar aging
│           ├── env/
│           │   ├── mod.rs
│           │   ├── simulation.rs   # Core Gym step / reset state machine
│           │   └── stochastic.rs   # Ornstein-Uhlenbeck noise & price shocks
│           └── data/
│               ├── mod.rs
│               └── loader.rs       # CSV ingestion into contiguous memory
├── python/
│   └── voltflow/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── gym_wrapper.py      # Gymnasium API adapter
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train_ppo.py        # PPO pipeline with SB3
│       │   └── baselines.py        # Rule-based & TOU heuristic baselines
│       ├── server/
│       │   ├── __init__.py
│       │   └── app.py              # FastAPI WebSocket server
│       └── scripts/
│           ├── download_data.py    # Fetches real grid/weather datasets
│           └── run_benchmarks.py   # Side-by-side performance evaluation
└── ui/
    ├── package.json
    ├── tsconfig.json
    ├── tailwind.config.js
    └── src/
        ├── app/
        │   ├── layout.tsx
        │   └── page.tsx            # Live telemetry dashboard
        ├── components/
        │   ├── MetricsGrid.tsx
        │   ├── LiveChart.tsx
        │   └── BatteryGauge.tsx
        └── hooks/
            └── useSimulationSocket.ts
```

---

## 4. Mathematical & Physical Modeling Specifications

The AI agent **must strictly implement these exact formulas** within the Rust core:

### 4.1. Coulomb Counting & Inverter Efficiency
For a time-step $\Delta t = 0.25\text{ hours}$ ($15\text{ minutes}$):

$$\text{Effective Power: } P_{\text{eff}} = \begin{cases} P_{\text{action}} \cdot \eta_{\text{inverter}} & \text{if } P_{\text{action}} > 0 \text{ (Charging)} \\ \frac{P_{\text{action}}}{\eta_{\text{inverter}}} & \text{if } P_{\text{action}} < 0 \text{ (Discharging)} \end{cases}$$

$$\Delta SoC = \frac{P_{\text{eff}} \cdot \Delta t}{E_{\text{nominal}} \cdot SoH}$$

$$SoC_{t+1} = \text{clamp}(SoC_t + \Delta SoC, SoC_{\min}, SoC_{\max})$$

*Default Parameters:* $E_{\text{nominal}} = 1000.0\text{ kWh}$ ($1\text{ MWh}$), $P_{\max} = 500.0\text{ kW}$, $\eta_{\text{inverter}} = 0.96$, $SoC_{\min} = 0.05$, $SoC_{\max} = 0.95$.

### 4.2. Thermal Dynamics (Lumped-Capacitance Model)
$$Q_{\text{gen}} = I^2 \cdot R_{\text{internal}} = \left(\frac{P_{\text{eff}}}{V_{\text{nominal}}}\right)^2 \cdot R_{\text{internal}}$$

$$\frac{dT_{\text{cell}}}{dt} = \frac{Q_{\text{gen}} - h \cdot A \cdot (T_{\text{cell}} - T_{\text{ambient}})}{C_{\text{thermal}}}$$

*Default Parameters:* $V_{\text{nominal}} = 800.0\text{ V}$, $R_{\text{internal}} = 0.015\text{ }\Omega$, $C_{\text{thermal}} = 15,000.0\text{ J/K}$, $h \cdot A = 25.0\text{ W/K}$.

### 4.3. Non-Linear Battery Degradation (Cycle + Calendar Aging)
Total State of Health ($SoH \in [0.0, 1.0]$) loss per step:

$$\Delta SoH_{\text{total}} = \Delta SoH_{\text{cycle}} + \Delta SoH_{\text{calendar}}$$

$$\Delta SoH_{\text{cycle}} = \gamma_{\text{cycle}} \cdot \left(\frac{|P_{\text{eff}}|}{P_{\max}}\right)^{1.3} \cdot e^{\frac{T_{\text{cell}} - 298.15}{20.0}} \cdot \Delta t$$

$$\Delta SoH_{\text{calendar}} = \gamma_{\text{cal}} \cdot e^{k_{\text{temp}}(T_{\text{cell}} - 298.15)} \cdot (SoC_t)^{0.5} \cdot \Delta t$$

*Default Parameters:* $\gamma_{\text{cycle}} = 1.2 \times 10^{-6}$, $\gamma_{\text{cal}} = 4.0 \times 10^{-7}$, $k_{\text{temp}} = 0.03$.

### 4.4. Financial Ledger & Reward Function
$$\text{Revenue}_t = \left(P_{\text{discharged}} \cdot \lambda_t\right) - \left(P_{\text{charged}} \cdot \lambda_t\right)$$

$$\text{Degradation Penalty: } C_{\text{deg}} = \Delta SoH_{\text{total}} \cdot \text{AssetReplacementCost}$$

$$\text{Thermal Violation Penalty: } C_{\text{thermal}} = \begin{cases} \kappa \cdot (T_{\text{cell}} - T_{\text{crit}})^2 & \text{if } T_{\text{cell}} > T_{\text{crit}} \\ 0 & \text{otherwise} \end{cases}$$

$$\mathcal{R}_t = \frac{\text{Revenue}_t - C_{\text{deg}} - C_{\text{thermal}}}{\text{RewardNormalizationScale}}$$

*Default Parameters:* $\text{AssetReplacementCost} = \$150,000$, $T_{\text{crit}} = 318.15\text{ K}$ ($45^\circ\text{C}$), $\kappa = 50.0$, $\text{RewardNormalizationScale} = 100.0$.

---

## 5. Vector Spaces & Observation Specification

### 5.1. Observation Vector (8 Continuous Features)
Normalized to $[-1.0, 1.0]$ or $[0.0, 1.0]$ for network stability:

| Index | Feature | Unit / Raw Range | Normalization Formula |
| :--- | :--- | :--- | :--- |
| `0` | $SoC_t$ | $[0.0, 1.0]$ | $SoC_t$ |
| `1` | $SoH_t$ | $[0.7, 1.0]$ | $(SoH_t - 0.7) / 0.3$ |
| `2` | $T_{\text{cell}}$ | $[273.15, 333.15]\text{ K}$ | $(T_{\text{cell}} - 273.15) / 60.0$ |
| `3` | $T_{\text{ambient}}$ | $[263.15, 318.15]\text{ K}$ | $(T_{\text{ambient}} - 263.15) / 55.0$ |
| `4` | $\lambda_t$ (Spot Price) | $[-50.0, 300.0]\text{ \$/MWh}$ | $(\text{clamp}(\lambda_t, -50, 300) + 50) / 350.0$ |
| `5` | $\lambda_{t+1}$ (1-Step Ahead) | $[-50.0, 300.0]\text{ \$/MWh}$ | $(\text{clamp}(\lambda_{t+1}, -50, 300) + 50) / 350.0$ |
| `6` | $\sin(\text{Hour Angle})$ | $[-1.0, 1.0]$ | $\sin(2\pi \cdot \text{hour} / 24.0)$ |
| `7` | $\cos(\text{Hour Angle})$ | $[-1.0, 1.0]$ | $\cos(2\pi \cdot \text{hour} / 24.0)$ |

### 5.2. Action Space
* Continuous 1D Box: $a_t \in [-1.0, 1.0]$ where:
  * $[-1.0, 0.0) \rightarrow$ Discharge power scale ($a_t \cdot P_{\max}$).
  * $(0.0, 1.0] \rightarrow$ Charge power scale ($a_t \cdot P_{\max}$).
  * $0.0 \rightarrow$ Idle.

---

## 6. Implementation Code Signatures

### 6.1. Rust Core & PyO3 Module (`crates/voltflow_core/src/lib.rs`)

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
pub struct RustBessEnv {
    // Internal state variables
    soc: f32,
    soh: f32,
    temp_cell: f32,
    current_step: usize,
    max_steps: usize,
    // Configuration parameters
    nominal_energy_kwh: f32,
    max_power_kw: f32,
    // Loaded market time series
    prices: Vec<f32>,
    ambient_temps: Vec<f32>,
    solar_irradiance: Vec<f32>,
}

#[pymethods]
impl RustBessEnv {
    #[new]
    pub fn new(csv_path: &str, max_steps: usize) -> PyResult<Self> {
        // Loads CSV into memory, sets initial state
        // Must handle File IO and parsing gracefully
        todo!()
    }

    pub fn reset(&mut self, randomize: bool) -> PyResult<Vec<f32>> {
        // Resets SoC, thermal state, randomizes starting index if true
        // Returns 8-element observation Vec<f32>
        todo!()
    }

    pub fn step(&mut self, action: f32) -> PyResult<(Vec<f32>, f32, bool, bool, PyObject)> {
        // Computes:
        // 1. Coulomb counting & Inverter losses
        // 2. Heat generation & thermal dissipation
        // 3. Degradation costs
        // 4. Arbitrage cash flow
        // Returns: (observation, reward, terminated, truncated, info_dict)
        todo!()
    }
}

#[pymodule]
fn voltflow_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBessEnv>()?;
    Ok(())
}
```

### 6.2. Python Gymnasium Wrapper (`python/voltflow/envs/gym_wrapper.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import voltflow_core

class VoltFlowEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path: str, max_steps: int = 96): # 96 steps = 24 hours at 15m intervals
        super().__init__()
        self._rust_env = voltflow_core.RustBessEnv(csv_path, max_steps)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        randomize = options.get("randomize", True) if options else True
        obs = self._rust_env.reset(randomize)
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        act = float(np.clip(action[0], -1.0, 1.0))
        obs, reward, term, trunc, info = self._rust_env.step(act)
        return np.array(obs, dtype=np.float32), float(reward), term, trunc, info
```

---

## 7. Step-by-Step AI Agent Execution Plan

The autonomous coding agent must execute the build sequentially in **5 strict phases**:

```
PHASE 1: Data Acquisition & Validation
  └─ Download Spain / CAISO dataset → Validate columns → Save to `data/raw/`

PHASE 2: Rust Core Engine (`voltflow_core`)
  └─ Implement battery electrochemistry, degradation, thermal ODE, & PyO3 bindings
  └─ Test: `cargo test` verifying zero memory leaks and >1M steps/sec

PHASE 3: Maturin Build & Gymnasium Environment
  └─ Run `maturin develop --release` → Implement `VoltFlowEnv`
  └─ Test: `python -m pytest` with random actions

PHASE 4: Baseline Models & PPO Training Pipeline
  └─ Implement Rule-Based / TOU Heuristic
  └─ Train PPO agent using Stable-Baselines3 (2,000,000 steps)
  └─ Benchmark: Verify RL beats Heuristic by ≥15% net PnL

PHASE 5: Telemetry Server & Interactive Web UI
  └─ FastAPI WebSocket streaming endpoint
  └─ Next.js/Tailwind frontend showing live PnL, SoC, and battery health
```

### Milestone Verification Gates

* **Gate 1 (Rust Speed):** A Rust benchmark test (`benches/sim_benchmark.rs`) must verify execution speed $> 2,000,000\text{ steps/second}$ across 4 parallel threads.
* **Gate 2 (Electrochemical Sanity):** Discharging at $1.0\text{ C}$ for 1 hour must drop SoC by exactly $(1.0 \times \eta_{\text{inverter}}) / 1.0 \approx 96\%$.
* **Gate 3 (Model Convergence):** The PPO agent must learn to halt charging when spot prices exceed the mean 75th percentile price.
* **Gate 4 (End-to-End Execution):** Executing `python python/voltflow/scripts/run_benchmarks.py` must run a 7-day simulation test and output a Markdown summary table comparing **Heuristic vs. VoltFlow RL**.

---

## 8. Agent Instructions & Prompt Directives

```text
YOU ARE AN AUTONOMOUS SYSTEMS & MACHINE LEARNING CODING AGENT.
YOUR OBJECTIVE IS TO IMPLEMENT THE VOLTFLOW PLATFORM AS SPECIFIED.

STRICT OPERATIONAL RULES:
1. NO STUBBING OR MOCK CODE: Write complete, production-ready Rust and Python code. Implement every mathematical equation completely.
2. MEMORY EFFICIENCY: In Rust, avoid allocating memory inside the `step()` hot loop. Pre-allocate all buffers, vectors, and state structs inside `RustBessEnv::new()`.
3. NUMERICAL STABILITY: Wrap all floating-point divisions against zero guards (e.g., `(val).max(1e-6)`). Always clamp SoC and temperatures to physical boundaries.
4. ERROR HANDLING: Do not use `unwrap()` in production paths. Propagate errors gracefully using `PyResult` or explicit `Result<T, E>`.
5. ZERO-COPY MATURIN COMPLIANCE: Ensure `crates/voltflow_core/Cargo.toml` and root `pyproject.toml` correctly cross-reference each other so that `maturin develop --release` builds cleanly in any standard Python 3.10/3.11 virtual environment.

BEGIN BY CREATING THE DIRECTORY STRUCTURE, INITIALIZING CARGO WORKSPACES, AND PROCEEDING THROUGH PHASES 1 TO 5.
```
