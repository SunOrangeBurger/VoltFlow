#!/usr/bin/env bash
# VoltFlow dashboard startup dock.
#
# Starts the telemetry backend (FastAPI/uvicorn) and the Next.js frontend
# together, waits for both to come up, opens the dashboard in your default
# browser, and tears both down cleanly on Ctrl-C.
#
# Prerequisites (see README.md for the full one-time setup):
#   - Python venv created and activated, `uv pip install -r requirements.txt`
#   - `maturin develop --release` already run (voltflow_core importable)
#   - `cd ui && npm install` already run
#
# Usage:
#   ./start_dashboard.sh
#
# Env overrides (all optional, same as the backend itself accepts):
#   VOLTFLOW_CSV_PATH, VOLTFLOW_MODELS_DIR, VOLTFLOW_PPO_MODEL,
#   VOLTFLOW_STEP_INTERVAL, VOLTFLOW_MAX_STEPS, VOLTFLOW_SELECTION_EPISODES

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BACKEND_PORT="${VOLTFLOW_BACKEND_PORT:-8000}"
FRONTEND_PORT="${VOLTFLOW_FRONTEND_PORT:-3000}"
DASHBOARD_URL="http://localhost:${FRONTEND_PORT}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down VoltFlow dashboard..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- Sanity checks -----------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "WARNING: no Python virtual environment appears active."
  echo "  Run 'source .venv/bin/activate' first (see README.md step 3)."
fi

if ! python3 -c "import voltflow_core" >/dev/null 2>&1; then
  echo "WARNING: 'import voltflow_core' failed — the Rust extension may not"
  echo "  be built yet. Run 'maturin develop --release' first (README.md step 2/3)."
fi

if [[ ! -d "ui/node_modules" ]]; then
  echo "WARNING: ui/node_modules missing — run 'cd ui && npm install' first."
fi

if [[ ! -f "${VOLTFLOW_CSV_PATH:-data/raw/energy_weather_spain.csv}" ]]; then
  echo "WARNING: dataset CSV not found. Run generate_synthetic_data.py or"
  echo "  download_data.py first (README.md step 1)."
fi

# --- Start backend -------------------------------------------------------
echo "Starting VoltFlow telemetry backend on port ${BACKEND_PORT}..."
echo "  (this benchmarks every checkpoint under models/ to pick the best one — may take a bit)"
uvicorn voltflow.server.app:app --port "${BACKEND_PORT}" --app-dir python \
  > /tmp/voltflow_backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend health check to respond.
echo -n "Waiting for backend"
for _ in $(seq 1 120); do
  if curl -sf "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo " — up."
    break
  fi
  echo -n "."
  sleep 1
done
if ! curl -sf "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
  echo ""
  echo "ERROR: backend did not come up within 120s. Check /tmp/voltflow_backend.log:"
  tail -n 40 /tmp/voltflow_backend.log || true
  exit 1
fi

# --- Start frontend --------------------------------------------------------
echo "Starting VoltFlow frontend on port ${FRONTEND_PORT}..."
(cd ui && NEXT_PUBLIC_VOLTFLOW_WS_URL="ws://localhost:${BACKEND_PORT}/ws/telemetry" \
  npm run dev -- --port "${FRONTEND_PORT}" > /tmp/voltflow_frontend.log 2>&1) &
FRONTEND_PID=$!

echo -n "Waiting for frontend"
for _ in $(seq 1 60); do
  if curl -sf "$DASHBOARD_URL" >/dev/null 2>&1; then
    echo " — up."
    break
  fi
  echo -n "."
  sleep 1
done

# --- Open the dashboard -----------------------------------------------------
echo ""
echo "VoltFlow dashboard ready at ${DASHBOARD_URL}"
echo "  Backend health:  http://localhost:${BACKEND_PORT}/health"
echo "  Backend logs:    /tmp/voltflow_backend.log"
echo "  Frontend logs:   /tmp/voltflow_frontend.log"
echo ""

if command -v open >/dev/null 2>&1; then
  open "$DASHBOARD_URL"          # macOS
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$DASHBOARD_URL"      # Linux
elif command -v wslview >/dev/null 2>&1; then
  wslview "$DASHBOARD_URL"       # WSL
else
  echo "Open ${DASHBOARD_URL} in your browser manually."
fi

echo "Press Ctrl-C to stop both the backend and frontend."
wait