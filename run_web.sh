#!/usr/bin/env bash
#
# run_web.sh — launch the FP32 Diffusion web server (API + frontend).
#
# Ports (chosen to avoid conflicts with Ollama, OpenWebUI, monitoring):
#   Backend (FastAPI):  8765  (override with SD_API_PORT)
#   Frontend (Vite dev): 5174  (override with SD_FRONT_PORT)
#
# Usage:
#   ./run_web.sh              # production: serves built React from web/dist
#   ./run_web.sh --dev        # dev mode: runs Vite + API separately
#   ./run_web.sh --build      # build frontend, then run production
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_PORT="${SD_API_PORT:-8765}"
FRONT_PORT="${SD_FRONT_PORT:-5174}"
MODE="${1:-prod}"

# Activate venv if present
if [ -f ~/diffusion-env/bin/activate ]; then
    source ~/diffusion-env/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

install_api_deps() {
    pip install -q fastapi "uvicorn[standard]" python-multipart 2>/dev/null || true
}

if [ "$MODE" = "--build" ]; then
    echo "=== Building React frontend ==="
    cd web
    npm install
    npm run build
    cd "$SCRIPT_DIR"
    echo "=== Starting production server (port $API_PORT) ==="
    install_api_deps
    export SD_API_PORT="$API_PORT"
    exec python api_server.py

elif [ "$MODE" = "--dev" ]; then
    echo "=== Starting DEV mode ==="
    echo "  API      → http://0.0.0.0:$API_PORT"
    echo "  Frontend → http://0.0.0.0:$FRONT_PORT"
    install_api_deps

    # Start API server in background
    export SD_API_PORT="$API_PORT"
    python api_server.py &
    API_PID=$!

    # Start Vite dev server
    cd web
    if [ ! -d node_modules ]; then
        npm install
    fi
    npx vite --port "$FRONT_PORT" --host 0.0.0.0 &
    VITE_PID=$!

    # Trap exit to kill both
    trap "kill $API_PID $VITE_PID 2>/dev/null || true" EXIT
    wait

else
    # Production mode
    if [ ! -d web/dist ]; then
        echo "Frontend not built. Run: ./run_web.sh --build"
        echo "Or use dev mode: ./run_web.sh --dev"
        exit 1
    fi
    echo "=== Starting production server (port $API_PORT) ==="
    echo "  Open http://localhost:$API_PORT in your browser"
    install_api_deps
    export SD_API_PORT="$API_PORT"
    exec python api_server.py
fi