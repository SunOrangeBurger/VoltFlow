# VoltFlow

Autonomous BESS Arbitrage & Degradation Management System. See
`progress.md` for what's done, `to_be_done.md` for what's left.

## Directory structure

Keep this layout exactly — the Rust crate, Python package, and Next.js app
all reference each other by relative path.

```
voltflow/
├── Cargo.toml                  # Rust workspace root
├── pyproject.toml              # maturin build config (points at crate)
├── requirements.txt
├── progress.md
├── to_be_done.md
├── data/raw/
│   └── energy_weather_spain.csv   # <-- put your dataset here (see below)
├── crates/voltflow_core/        # Rust simulation engine
├── python/voltflow/             # Python package (gym env, training, server)
└── ui/                          # Next.js dashboard
```

## 1. Dataset

**Use the Kaggle "Energy Consumption, Generation, Prices and Weather"
(Spain) dataset only** — not CityLearn (wrong shape: multi-building
demand-response, not single-BESS arbitrage).

https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

Two ways to populate `data/raw/energy_weather_spain.csv`:

**Option A — real data (recommended before real training):**
```bash
uv pip install kaggle pandas
# Place your Kaggle API token at ~/.kaggle/kaggle.json first
python python/voltflow/scripts/download_data.py
```

**Option B — synthetic placeholder (for testing the pipeline only):**
```bash
python python/voltflow/scripts/generate_synthetic_data.py --days 90
```
A synthetic CSV is already included in this delivery so you can test the
pipeline immediately without Kaggle access. Swap in the real data before
trusting any training results.

Either way the loader expects these columns (see
`crates/voltflow_core/src/data/loader.rs` for accepted aliases):
`timestamp, price_eur_mwh, ambient_temp_c, solar_irradiance`

## 2. Rust core

Requires Rust stable 1.78+ (`rustup install stable`).

```bash
cd crates/voltflow_core
cargo test        # unit tests — confirmed passing (17/17) on reference hardware
cargo bench       # Gate 1: verify >2M steps/sec across 4 threads
cd ../..
```

If `cargo test` fails with something like `the configured Python
interpreter version (3.14) is newer than PyO3's maximum supported version`,
your system's default `python3` on `PATH` is newer than pyo3 supports, and
cargo picked that up instead of the venv from step 3. Set up the venv
first (step 3 below), then:
```bash
export PYO3_PYTHON=$(pwd)/.venv/bin/python3
cargo clean
cargo test
```
Keep `PYO3_PYTHON` exported for the rest of your session.

## 3. Python environment + maturin build (using `uv`)

Requires Python 3.10 or 3.11 (not 3.12+ — some pinned deps may lack wheels
for newer Python versions).

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate the venv:
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python3 --version                # confirm this really is 3.11
```
If you don't have a 3.11 interpreter available at all, `uv` can fetch one:
```bash
uv python install 3.11
```

Install dependencies and build the Rust extension into this venv:
```bash
uv pip install -r requirements.txt
uv pip install maturin

# If you hit the PyO3/Python-version error from step 2, set this first:
export PYO3_PYTHON=$(pwd)/.venv/bin/python3

maturin develop --release
```

Smoke test the environment:
```bash
python -c "
from voltflow.envs.gym_wrapper import VoltFlowEnv
env = VoltFlowEnv('data/raw/energy_weather_spain.csv', max_steps=96)
obs, _ = env.reset()
print('obs shape:', obs.shape)
obs, reward, term, trunc, info = env.step([0.5])
print('reward:', reward, 'info:', info)
"
```

## 4. Train

```bash
# Smoke test first (a few minutes):
python -m voltflow.models.train_ppo --timesteps 50000 --n-envs 2

# Full run (spec Phase 4 target — expect hours on CPU):
python -m voltflow.models.train_ppo --timesteps 2000000 --n-envs 4 \
    --out models/ppo_voltflow
```

## 5. Benchmark (Gate 4)

```bash
python python/voltflow/scripts/run_benchmarks.py \
    --csv data/raw/energy_weather_spain.csv \
    --ppo-model models/ppo_voltflow.zip
```
Writes `benchmark_results.md` with a Heuristic-vs-RL PnL comparison table.

## 6. Live dashboard

Terminal 1 — telemetry backend (from repo root, venv activated):
```bash
export VOLTFLOW_CSV_PATH=data/raw/energy_weather_spain.csv
export VOLTFLOW_PPO_MODEL=models/ppo_voltflow.zip   # optional, falls back to idle policy
uvicorn voltflow.server.app:app --reload --port 8000 --app-dir python
```
Note the `--app-dir python` — the `voltflow` package lives under
`python/voltflow/`, not the repo root, so uvicorn needs to be told where to
find it (alternatively, `pip install -e .` the package once and drop this
flag).

Terminal 2 — frontend:
```bash
cd ui
npm install
npm run dev
```
Open http://localhost:3000 — should show a live-updating dashboard once
both are running.

## Locked versions

See `Cargo.toml` / `crates/voltflow_core/Cargo.toml` / `requirements.txt`
for exact pinned versions — don't drift from these without checking ABI
compatibility (PyO3 in particular is version-sensitive).