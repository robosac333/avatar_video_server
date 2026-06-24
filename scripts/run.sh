#!/usr/bin/env bash
# Run by post_start.sh on every pod boot.
# Also callable manually: bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export WORKSPACE="${WORKSPACE:-/workspace}"

echo "==> Starting FastAPI inference server on :8000"
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Give the API a moment to bind before the UI tries to reach it
sleep 4

echo "==> Serving HTML UI on :8501"
cd ui
python3 -m http.server 8501 &
UI_PID=$!

echo "==> Avatar Studio running"
echo "    API : http://0.0.0.0:8000"
echo "    UI  : http://0.0.0.0:8501/avatar_studio.html"

# If either process dies, exit so the caller knows something broke
wait -n "$API_PID" "$UI_PID"
echo "==> A process exited. Shutting down."
kill "$API_PID" "$UI_PID" 2>/dev/null || true
