# VoltFlow: Autonomous BESS Arbitrage & Degradation Management System

A high-throughput industrial Battery Energy Storage System (BESS) simulation and real-time dispatch optimization platform combining Rust electrochemistry simulation, PyO3-powered Gymnasium environments, PPO reinforcement learning, and live telemetry dashboard.

## Project Status

**✅ All four specification gates cleared**

- **Gate 1 (Throughput):** target >2M steps/sec — measured **35.9M steps/sec** (4 threads, 1000 steps each, parallel) / **22.6M steps/sec** (single-threaded), `cargo bench` release profile, 2026-09-01. Note: 4 threads gave ~1.6x speedup over single-thread, not ~4x — target is still cleared by a wide margin, but the sub-linear scaling hasn't been profiled and is an open item (see STATUS.md).
- **Gate 2 (Electrochemical sanity):** Verified discharge/charge physics
- **Gate 3 (Behavioral verification):** Agent learns price-threshold arbitrage (PASS)
- **Gate 4 (Performance):** PPO beats heuristics by ≥15% net PnL across all 15 CV runs

**Model checkpoints are not committed to this repo** (`models/` is gitignored by design — see [Training a Checkpoint](#4-train--benchmark) below to reproduce). Two example checkpoints from earlier single-seed runs are tracked for quick smoke-testing: `models/best_model.zip` and `models/ppo_voltflow.zip`. The CV-selected best policy (`ppo_voltflow_fold3_seed4`, referenced in [results/README.md](./results/README.md)) is reproducible via the CV sweep command below but is not itself included in the repo.

**Documentation:**
- **[TECHNICAL.md](./TECHNICAL.md)** - Complete mathematical specification & implementation details
- **[STATUS.md](./STATUS.md)** - Project progress, known deviations, and remaining tasks
- **[results/README.md](./results/README.md)** - Benchmark results and cross-validation summary

## Quick Start

### 1. Prerequisites
- Rust stable 1.78+ (`rustup install stable`)
- Python 3.10 or 3.11 (**not 3.12+ — this applies to `cargo test`/`cargo bench` too**, not just `maturin develop`, since PyO3 0.22.x will fail to build against a system Python 3.12+ interpreter even for pure-Rust test runs. See step 3.)
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
# Set up a 3.11 venv FIRST — needed even for plain `cargo test`/`cargo bench`,
# since this crate is in the same Cargo workspace as the PyO3 bindings.
uv venv --python 3.11 .venv
source .venv/bin/activate
export PYO3_PYTHON=$(pwd)/.venv/bin/python3   # run this from the repo root

# Rust core engine
cd crates/voltflow_core
cargo test        # 28/28 tests passing
cargo bench       # Gate 1 verification — expect ~20-35M steps/sec depending on hardware

# Python environment
uv pip install -r ../../requirements.txt
uv pip install maturin
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

# Cross-validation sweep (3 folds × 5 seeds) — reproduces the CV-best checkpoint
# referenced in results/README.md; writes to models/cv/, which is gitignored
python python/voltflow/scripts/run_cv_sweep.py \
    --csv data/raw/energy_weather_spain.csv

# Single benchmark against a checkpoint you've trained
python python/voltflow/scripts/run_benchmarks.py \
    --csv data/raw/energy_weather_spain.csv \
    --ppo-model models/cv/ppo_voltflow_fold3_seed4.zip
```

### 5. Live Dashboard

```bash
# Terminal 1: Telemetry backend
# Use a checkpoint you've trained (step 4). For a quick smoke-test without
# training, the tracked models/ppo_voltflow.zip works but is a single-seed
# run, not the CV-validated best policy.
export VOLTFLOW_CSV_PATH=data/raw/energy_weather_spain.csv
export VOLTFLOW_PPO_MODEL=models/ppo_voltflow.zip
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