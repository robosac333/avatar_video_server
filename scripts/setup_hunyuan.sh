#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# CLEAN venv-based setup for HunyuanVideo-Avatar on a single-GPU RunPod pod.
#
# Why a venv: the RunPod base image ships its own transformers/huggingface-hub/
# diffusers that fight with the exact versions Hunyuan needs. A virtualenv gives
# Hunyuan its own isolated packages so nothing conflicts with the system. This
# is the robust fix for the dependency errors hit on the bare system.
#
# The venv lives on the NETWORK VOLUME (/workspace/hunyuan-venv), so it SURVIVES
# pod restarts — you only run this heavy install once.
#
# Versions are Tencent's own tested set: transformers 4.41.2 + diffusers 0.33.0.
#
# Usage:  bash setup_hunyuan_venv.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS="$WORKSPACE/models/HunyuanVideo-Avatar"
REPO="$WORKSPACE/repos/HunyuanVideo-Avatar"
VENV="$WORKSPACE/hunyuan-venv"

echo "════════════════════════════════════════════════════════"
echo "  HunyuanVideo-Avatar — clean venv setup"
echo "════════════════════════════════════════════════════════"

# ── 0. Weights must already be on the volume ─────────────────────────────────
CKPT="$MODELS/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
if [ ! -f "$CKPT" ]; then
  echo "✗ Weights not found at: $CKPT"
  echo "  Download first: hf download tencent/HunyuanVideo-Avatar --local-dir $MODELS"
  exit 1
fi
echo "✓ Weights found on volume"

# ── 1. System deps (these are fine to touch — not Python) ────────────────────
echo "==> Installing ffmpeg + git + python venv tooling"
apt-get update -qq && apt-get install -y -qq ffmpeg git git-lfs python3-venv python3-pip >/dev/null 2>&1
git lfs install >/dev/null 2>&1 || true

# ── 2. Clone the repo ────────────────────────────────────────────────────────
mkdir -p "$WORKSPACE/repos"
if [ ! -d "$REPO" ]; then
  echo "==> Cloning HunyuanVideo-Avatar"
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git "$REPO"
else
  echo "✓ Repo already cloned"
fi

# ── 3. Create the isolated venv on the volume ────────────────────────────────
# --system-site-packages lets the venv SEE the base image's CUDA-built torch
# (so we don't re-download 3GB of GPU torch) while still installing our OWN
# transformers/diffusers ON TOP, isolated from the system ones.
if [ ! -d "$VENV" ]; then
  echo "==> Creating venv at $VENV (inherits system torch)"
  python3 -m venv --system-site-packages "$VENV"
else
  echo "✓ venv already exists"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "✓ venv active: $(which python3)"

python3 -m pip install --quiet --upgrade pip

# ── 4. Install Hunyuan's exact tested deps INTO the venv ─────────────────────
echo "==> Installing repo requirements into venv"
pip install --quiet -r "$REPO/requirements.txt" || true

echo "==> Pinning Tencent's tested versions (transformers 4.41.2 / diffusers 0.33.0)"
pip install --quiet \
    "transformers==4.41.2" \
    "diffusers==0.33.0" \
    "huggingface-hub>=0.23,<0.25" \
    "tokenizers>=0.19,<0.20" \
    "gradio==3.39.0" \
    flask loguru

# FlashAttention — optional speedup; never blocks the run
pip install --quiet ninja 2>/dev/null || true
pip install --quiet flash-attn --no-build-isolation 2>/dev/null \
    || echo "   (flash-attn skipped — model still runs)"

# ── 5. Link weights into the repo ────────────────────────────────────────────
echo "==> Linking weights into repo"
mkdir -p "$REPO/weights"
rm -rf "$REPO/weights/ckpts"
ln -s "$MODELS/ckpts" "$REPO/weights/ckpts"
if [ -f "$REPO/weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt" ]; then
  echo "✓ Weights linked"
else
  echo "✗ Weight link failed"; exit 1
fi

# ── 6. Verify the imports that previously failed ─────────────────────────────
echo "==> Verifying imports inside venv"
python3 -c "from diffusers.hooks import apply_group_offloading; print('✓ diffusers.hooks OK')"
python3 -c "import transformers; print('✓ transformers', transformers.__version__)"
python3 -c "import torch; print('✓ torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Clean setup complete. venv: $VENV"
echo "  Launch the UI with:  bash $WORKSPACE/launch_ui.sh"
echo "════════════════════════════════════════════════════════"