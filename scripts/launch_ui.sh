#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# Launch the official HunyuanVideo-Avatar Gradio web UI in SINGLE-GPU mode.
#
# The repo's stock run_gradio.sh hardcodes 8 GPUs (--nproc_per_node=8) which
# crashes on a 1-GPU pod with "invalid device ordinal". This launches the
# single-GPU flask backend + gradio frontend directly instead.
#
# Usage:  bash launch_ui.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="${REPO:-/workspace/repos/HunyuanVideo-Avatar}"
cd "$REPO"

export PYTHONPATH=./
export MODEL_BASE=./weights
export CPU_OFFLOAD=1
export DISABLE_SP=1                 # disable multi-GPU sequence parallelism
export CUDA_VISIBLE_DEVICES=0       # pin to the single GPU

CKPT="./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"

echo "==> Starting single-GPU inference backend (flask) on :8080"
# Single process, single GPU — NO torchrun (that's what caused the 8-GPU crash)
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

# Give the backend time to load the 80GB model into GPU memory
echo "==> Backend loading model (2-4 min)…"
sleep 8

echo "==> Starting Gradio frontend"
python3 hymm_gradio/gradio_audio.py &
UI_PID=$!

echo ""
echo "════════════════════════════════════════════════════════"
echo "  UI starting. Watch for a line like:"
echo "    Running on public URL: https://xxxxx.gradio.live"
echo "  Open that link — it's your working interface."
echo ""
echo "  Or expose port 8080 on RunPod and open:"
echo "    https://YOURPODID-8080.proxy.runpod.net"
echo "════════════════════════════════════════════════════════"

wait -n "$BACKEND_PID" "$UI_PID"
kill "$BACKEND_PID" "$UI_PID" 2>/dev/null || true