# VoltFlow: Autonomous BESS Arbitrage & Degradation Management System

A high-throughput industrial Battery Energy Storage System (BESS) simulation and real-time dispatch optimization platform combining Rust electrochemistry simulation, PyO3-powered Gymnasium environments, PPO reinforcement learning, and live telemetry dashboard.

## Project Status

**✅ All four specification gates cleared with comprehensive validation**

- **Gate 1 (Throughput):** >2M steps/sec across 4 threads (Rust benchmark)
- **Gate 2 (Electrochemical sanity):** Verified discharge/charge physics
- **Gate 3 (Behavioral verification):** Agent learns price-threshold arbitrage (PASS)
- **Gate 4 (Performance):** PPO beats heuristics by ≥15% net PnL across all 15 CV runs

**Current checkpoint:** `models/cv/ppo_voltflow_fold3_seed4.zip` (best fold3 PnL, Gate 3 verified)

**Documentation:**
- **[TECHNICAL.md](./TECHNICAL.md)** - Complete mathematical specification & implementation details
- **[STATUS.md](./STATUS.md)** - Project progress, known deviations, and remaining tasks
- **[results/README.md](./results/README.md)** - Benchmark results and cross-validation summary

## Quick Start

### 1. Prerequisites
- Rust stable 1.78+ (`rustup install stable`)
- Python 3.10 or 3.11 (not 3.12+)
- Node.js 18+ (for dashboard)

### 2. Dataset Setup

Use the Kaggle ["Energy Consumption, Generation, Prices and Weather" (Spain)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) dataset:

```bash
# Option A: Real data (recommended)
uv pip install kaggle pandas
python python/voltflow/scripts/download_data.py  # Requires Kaggle API token

# Option B: Synthetic placeholder (testing only)
python python/voltflow/scripts/generate_synthetic_data.py --days 90
```

### 3. Build & Test

```bash
# Rust core engine
cd crates/voltflow_core
cargo test        # 17/17 tests passing
cargo bench       # Gate 1 verification

# Python environment
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install maturin
export PYO3_PYTHON=$(pwd)/.venv/bin/python3  # If PyO3 version mismatch
maturin develop --release

# Smoke test
python -c "
from voltflow.envs.gym_wrapper import VoltFlowEnv
env = VoltFlowEnv('data/raw/energy_weather_spain.csv', max_steps=96)
obs, _ = env.reset()
print('obs shape:', obs.shape)
obs, reward, term, trunc, info = env.step([0.5])
print('reward:', reward, 'info:', info)
"
```

### 4. Train & Benchmark

```bash
# Full PPO training (expect hours on CPU)
python -m voltflow.models.train_ppo --timesteps 2000000 --n-envs 4 \
    --out models/ppo_voltflow

# Cross-validation sweep (3 folds × 5 seeds)
python python/voltflow/scripts/run_cv_sweep.py \
    --csv data/raw/energy_weather_spain.csv

# Single benchmark
python python/voltflow/scripts/run_benchmarks.py \
    --csv data/raw/energy_weather_spain.csv \
    --ppo-model models/cv/ppo_voltflow_fold3_seed4.zip
```

### 5. Live Dashboard

```bash
# Terminal 1: Telemetry backend
export VOLTFLOW_CSV_PATH=data/raw/energy_weather_spain.csv
export VOLTFLOW_PPO_MODEL=models/cv/ppo_voltflow_fold3_seed4.zip
uvicorn voltflow.server.app:app --reload --port 8000 --app-dir python

# Terminal 2: Frontend
cd ui
npm install
npm run dev
```

Open http://localhost:3000 for live telemetry dashboard.

## Project Structure

```
voltflow/
├── Cargo.toml                  # Rust workspace root
├── pyproject.toml              # maturin build config
├── requirements.txt
├── README.md                   # This file
├── TECHNICAL.md                # Mathematical specification & implementation
├── STATUS.md                   # Progress & remaining tasks
├── data/
│   └── raw/
│       └── energy_weather_spain.csv   # Market/weather dataset
├── crates/voltflow_core/       # Rust simulation engine
├── python/voltflow/            # Python package (env, training, server)
├── ui/                         # Next.js dashboard (Tailwind + Recharts)
└── results/                    # Benchmark results & CV summaries
```

## Key Results

**Walk-forward Cross-Validation (3 folds × 5 seeds = 15 runs):**
- **100% success rate** - PPO beats both heuristics on every run
- **Worst case:** +136.8% improvement over best heuristic
- **Best case:** +476.4% improvement over best heuristic
- **Overall mean:** $359.44 ± $55.05 net PnL

**Gate 3 Verification:**
- Mean action above 75th percentile price: -0.9676 (near-max discharge)
- Mean action below 75th percentile: -0.2885  
- Only 0.3% of above-threshold steps show any charging

## License & Citation

This project implements the VoltFlow specification for autonomous BESS management. For commercial use or academic citation, please refer to the technical documentation.

---
*See [TECHNICAL.md](./TECHNICAL.md) for complete mathematical specification and implementation details.*