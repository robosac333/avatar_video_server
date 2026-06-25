#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# Clean venv-based setup for HunyuanVideo-Avatar — single GPU RunPod pod.
#
# Uses the repo's own requirements.txt as the base, then applies one targeted
# patch to fix a known conflict between diffusers==0.33.0 and transformers>=4.50
# (FLAX_WEIGHTS_NAME was removed from transformers but diffusers still imports it).
#
# The venv lives on the NETWORK VOLUME so it survives pod restarts.
# Run this ONCE. Subsequent starts just call launch_ui.sh.
#
# Usage:  bash setup_hunyuan_venv.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS="$WORKSPACE/models/HunyuanVideo-Avatar"
REPO="$WORKSPACE/repos/HunyuanVideo-Avatar"
VENV="$WORKSPACE/hunyuan-venv"

echo "════════════════════════════════════════════════════════"
echo "  HunyuanVideo-Avatar — venv setup"
echo "════════════════════════════════════════════════════════"

# ── 0. Weights must already be on the volume ─────────────────────────────────
CKPT="$MODELS/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
if [ ! -f "$CKPT" ]; then
  echo "✗ Weights not found at: $CKPT"
  echo "  Run: hf download tencent/HunyuanVideo-Avatar --local-dir $MODELS"
  exit 1
fi
echo "✓ Weights found"

# ── 1. System deps ───────────────────────────────────────────────────────────
echo "==> System deps"
apt-get update -qq && apt-get install -y -qq \
    ffmpeg git git-lfs python3-venv python3-pip >/dev/null 2>&1
git lfs install >/dev/null 2>&1 || true

# ── 2. Clone repo ────────────────────────────────────────────────────────────
mkdir -p "$WORKSPACE/repos"
if [ ! -d "$REPO" ]; then
  echo "==> Cloning HunyuanVideo-Avatar"
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git "$REPO"
else
  echo "✓ Repo already cloned"
fi

# ── 3. Create venv on the volume (survives pod restarts) ─────────────────────
# --system-site-packages reuses the base image's CUDA-compiled torch
# so we don't re-download 3GB of GPU wheels.
if [ ! -d "$VENV" ]; then
  echo "==> Creating venv at $VENV"
  python3 -m venv --system-site-packages "$VENV"
else
  echo "✓ venv exists"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "✓ venv active: $(which python3)"
pip install --quiet --upgrade pip

# ── 4. Install repo's own requirements (the ground truth) ────────────────────
echo "==> Installing repo requirements (this takes a few minutes)"
pip install --quiet -r "$REPO/requirements.txt"

# ── 5. Patch the ONE known conflict ──────────────────────────────────────────
# diffusers==0.33.0 still imports FLAX_WEIGHTS_NAME from transformers.utils,
# but transformers>=4.50 removed it. We add it back inside the venv only —
# no system files are touched.
echo "==> Patching FLAX_WEIGHTS_NAME into venv transformers"
python3 -c "
import transformers.utils as tu
if not hasattr(tu, 'FLAX_WEIGHTS_NAME'):
    with open(tu.__file__, 'a') as f:
        f.write('\nFLAX_WEIGHTS_NAME = \"flax_model.msgpack\"\n')
    print('   patched')
else:
    print('   already present — no action needed')
"

# ── 6. Link weights into the repo ────────────────────────────────────────────
echo "==> Linking weights"
mkdir -p "$REPO/weights"
rm -rf "$REPO/weights/ckpts"
ln -sf "$MODELS/ckpts" "$REPO/weights/ckpts"
if [ -f "$REPO/weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt" ]; then
  echo "✓ Weights linked"
else
  echo "✗ Weight link failed"; exit 1
fi

# ── 7. Verify every import that previously failed ────────────────────────────
echo "==> Verifying imports"
python3 -c "from diffusers.hooks import apply_group_offloading; print('✓ diffusers.hooks')"
python3 -c "from transformers.utils import FLAX_WEIGHTS_NAME; print('✓ FLAX_WEIGHTS_NAME')"
python3 -c "import torch; print('✓ torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import gradio; print('✓ gradio', gradio.__version__)"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Setup complete."
echo "  Launch:  bash $WORKSPACE/launch_ui.sh"
echo "════════════════════════════════════════════════════════"