#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# Launch HunyuanVideo-Avatar's official Gradio UI — single GPU, inside the venv.
#
# Activates the isolated venv first so the right transformers/diffusers are used,
# then runs the single-GPU backend (no torchrun, no 8-GPU crash) + Gradio front.
#
# Usage:  bash launch_ui.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/repos/HunyuanVideo-Avatar}"
VENV="$WORKSPACE/hunyuan-venv"

# Activate the isolated environment
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "✓ venv active: $(which python3)"

cd "$REPO"
export PYTHONPATH=./
export MODEL_BASE=./weights
export CPU_OFFLOAD=1
export DISABLE_SP=1                 # disable multi-GPU sequence parallelism
export CUDA_VISIBLE_DEVICES=0       # single GPU

CKPT="./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"

echo "==> Starting single-GPU inference backend on :8080 (model load ~2-4 min)"
python3 hymm_gradio/flask_audio.py \
    --input 'assets/test.csv' \
    --ckpt "$CKPT" \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --use-fp8 \
    --cpu-offload \
    --infer-min &
BACKEND_PID=$!

sleep 8
echo "==> Starting Gradio frontend"
python3 hymm_gradio/gradio_audio.py &
UI_PID=$!

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Watch for:  Running on public URL: https://xxxxx.gradio.live"
echo "  Or open:    https://YOURPODID-8080.proxy.runpod.net"
echo "════════════════════════════════════════════════════════"

wait -n "$BACKEND_PID" "$UI_PID"
kill "$BACKEND_PID" "$UI_PID" 2>/dev/null || true