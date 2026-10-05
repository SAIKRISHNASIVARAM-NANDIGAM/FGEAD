#!/usr/bin/env bash
# ==============================================================================
# FGEAD - Production Startup Script (Linux / Cloud VM)
# Launches both FastAPI Backend and Streamlit Dashboard in Production Mode
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "          FGEAD - Production Stack Launcher"
echo "  Feature Graph Explainable Anomaly Detection Platform"
echo "======================================================================"

# 1. Load Environment Variables
if [ -f .env ]; then
    echo "[INFO] Loading configuration from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "[WARNING] .env not found, using .env.example defaults..."
    if [ -f .env.example ]; then
        export $(grep -v '^#' .env.example | xargs)
    fi
fi

export FGEAD_ENV="PRODUCTION"
export FGEAD_SERVER_HOST="${FGEAD_SERVER_HOST:-0.0.0.0}"
export FGEAD_SERVER_PORT="${FGEAD_SERVER_PORT:-8000}"
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-8501}"

# 2. Activate Python Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[WARNING] No virtual environment detected. Running with system python3..."
fi

# 3. Create required directories
mkdir -p data checkpoints logs

echo "[INFO] Starting FastAPI Backend on ${FGEAD_SERVER_HOST}:${FGEAD_SERVER_PORT}..."
uvicorn api.main:app --host "${FGEAD_SERVER_HOST}" --port "${FGEAD_SERVER_PORT}" --workers 2 &
API_PID=$!

echo "[INFO] Starting Streamlit Dashboard on port ${STREAMLIT_SERVER_PORT}..."
streamlit run app/streamlit_app.py \
    --server.port "${STREAMLIT_SERVER_PORT}" \
    --server.address "${FGEAD_SERVER_HOST}" \
    --server.headless true \
    --browser.gatherUsageStats false &
DASH_PID=$!

cleanup() {
    echo ""
    echo "[INFO] Shutting down FGEAD production processes..."
    kill "$API_PID" "$DASH_PID" 2>/dev/null || true
    wait "$API_PID" "$DASH_PID" 2>/dev/null || true
    echo "[INFO] All services stopped."
}

trap cleanup SIGINT SIGTERM EXIT

echo "[SUCCESS] FGEAD Stack is running."
echo "          - API Gateway: http://${FGEAD_SERVER_HOST}:${FGEAD_SERVER_PORT}"
echo "          - Dashboard:   http://${FGEAD_SERVER_HOST}:${STREAMLIT_SERVER_PORT}"
echo "          - Health:      http://${FGEAD_SERVER_HOST}:${FGEAD_SERVER_PORT}/health"
echo "          - API Docs:    http://${FGEAD_SERVER_HOST}:${FGEAD_SERVER_PORT}/docs"
echo "Press Ctrl+C to terminate."

wait
