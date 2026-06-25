#!/usr/bin/env bash
# Start the inference server and the Streamlit UI side by side on the pod.
#   bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export WORKSPACE="${WORKSPACE:-/workspace}"

echo "==> Starting FastAPI inference server on :8000"
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# give the API a moment to bind
sleep 3

echo "==> Starting Streamlit UI on :8501"
SERVER_URL="http://localhost:8000" \
  streamlit run ui/streamlit_app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true &
UI_PID=$!

echo "==> Up. API pid=$API_PID  UI pid=$UI_PID"
echo "    Expose ports 8000 (API) and 8501 (UI) via the RunPod HTTP proxy."

# If either process dies, take the whole thing down so the pod doesn't sit
# half-broken.
wait -n "$API_PID" "$UI_PID"
echo "A process exited; shutting down the other."
kill "$API_PID" "$UI_PID" 2>/dev/null || true
