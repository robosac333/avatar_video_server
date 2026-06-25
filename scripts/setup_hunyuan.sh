#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# ONE-SHOT SETUP for HunyuanVideo-Avatar on a single-GPU RunPod pod.
#
# Run this ONCE on a fresh pod after the model weights are on the network
# volume. It encodes every fix discovered during setup:
#   - clones the official repo
#   - installs the exact transformers/diffusers versions that work together
#   - restores FLAX_WEIGHTS_NAME (removed in newer transformers)
#   - links weights from the volume into the repo at ./weights
#   - leaves you ready to launch the official single-GPU Gradio server
#
# Usage:  bash setup_hunyuan.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS="$WORKSPACE/models/HunyuanVideo-Avatar"
REPO="$WORKSPACE/repos/HunyuanVideo-Avatar"

echo "════════════════════════════════════════════════════════"
echo "  HunyuanVideo-Avatar single-GPU setup"
echo "════════════════════════════════════════════════════════"

# ── 0. Sanity: weights must already be downloaded ────────────────────────────
CKPT="$MODELS/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
if [ ! -f "$CKPT" ]; then
  echo "✗ Weights not found at: $CKPT"
  echo "  Download them first:"
  echo "    hf download tencent/HunyuanVideo-Avatar --local-dir $MODELS"
  exit 1
fi
echo "✓ Weights found on volume"

# ── 1. System deps ───────────────────────────────────────────────────────────
echo "==> Installing ffmpeg + git"
apt-get update -qq && apt-get install -y -qq ffmpeg git git-lfs >/dev/null 2>&1
git lfs install >/dev/null 2>&1 || true

# ── 2. Clone the official repo ───────────────────────────────────────────────
mkdir -p "$WORKSPACE/repos"
if [ ! -d "$REPO" ]; then
  echo "==> Cloning HunyuanVideo-Avatar"
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git "$REPO"
else
  echo "✓ Repo already cloned"
fi

# ── 3. Install the EXACT working dependency versions ─────────────────────────
# These specific versions resolve the two import errors hit during setup:
#   - diffusers 0.33.0 has diffusers.hooks (needed by sample_gpu_poor.py)
#   - transformers 4.45.2 + the FLAX patch below satisfies the old import
echo "==> Installing Python deps (this takes a few minutes)"
pip install --quiet --break-system-packages -r "$REPO/requirements.txt" || true
pip install --quiet --break-system-packages \
    "transformers==4.45.2" \
    "diffusers==0.33.0" \
    "tokenizers>=0.20,<0.21" \
    gradio flask
# FlashAttention — speeds up the transformer; optional, won't fail the script
pip install --quiet --break-system-packages flash-attn --no-build-isolation 2>/dev/null \
    || echo "   (flash-attn skipped — model still runs)"

# ── 4. Restore FLAX_WEIGHTS_NAME removed from newer transformers ─────────────
echo "==> Patching transformers (restore FLAX_WEIGHTS_NAME)"
python3 -c "
import transformers.utils as tu
if not hasattr(tu, 'FLAX_WEIGHTS_NAME'):
    with open(tu.__file__, 'a') as f:
        f.write('\nFLAX_WEIGHTS_NAME = \"flax_model.msgpack\"\n')
    print('   patched')
else:
    print('   already present')
"

# ── 5. Link weights into the repo at ./weights/ckpts ─────────────────────────
echo "==> Linking weights into repo"
mkdir -p "$REPO/weights"
# Remove any stale link/dir from earlier attempts, then link the ckpts folder
rm -rf "$REPO/weights/ckpts"
ln -s "$MODELS/ckpts" "$REPO/weights/ckpts"
if [ -f "$REPO/weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt" ]; then
  echo "✓ Weights linked and resolve correctly"
else
  echo "✗ Weight link failed — check $MODELS/ckpts"
  exit 1
fi

# ── 6. Verify the two import fixes ───────────────────────────────────────────
echo "==> Verifying imports"
python3 -c "from diffusers.hooks import apply_group_offloading; print('✓ diffusers.hooks OK')"
python3 -c "from transformers.utils import FLAX_WEIGHTS_NAME; print('✓ FLAX_WEIGHTS_NAME OK')"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Setup complete."
echo ""
echo "  Launch the web UI with:"
echo "      bash $WORKSPACE/launch_ui.sh"
echo ""
echo "  Then open the gradio.live link it prints, or expose"
echo "  port 8080 on RunPod and open the proxy URL."
echo "════════════════════════════════════════════════════════"