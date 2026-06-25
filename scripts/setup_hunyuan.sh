#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# HunyuanVideo-Avatar — complete, self-contained single-GPU setup.
#
# Run ONCE on a fresh pod. One command, no follow-up patching. Encodes every
# fix discovered: isolated venv, clean huggingface_hub, repo requirements,
# FLAX patch, weight linking, full verification.
#
# Key design choice: the venv does NOT inherit system site-packages. The base
# image ships a BROKEN huggingface_hub (1.20.1, invalid metadata) that poisons
# every transformers import. We build a clean venv and install our own torch
# stack into it so nothing inherited can break us. This costs ~3GB extra in the
# venv but it lives on the network volume and only installs once.
#
# Usage:  bash setup_hunyuan.sh
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS="$WORKSPACE/models/HunyuanVideo-Avatar"
REPO="$WORKSPACE/repos/HunyuanVideo-Avatar"
VENV="$WORKSPACE/hunyuan-venv"

echo "════════════════════════════════════════════════════════"
echo "  HunyuanVideo-Avatar — full clean setup"
echo "════════════════════════════════════════════════════════"

# ── 0. Weights must already be on the volume ─────────────────────────────────
CKPT="$MODELS/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
if [ ! -f "$CKPT" ]; then
  echo "✗ Weights not found at: $CKPT"
  echo "  Run first: hf download tencent/HunyuanVideo-Avatar --local-dir $MODELS"
  exit 1
fi
echo "✓ Weights found"

# ── 1. System deps ───────────────────────────────────────────────────────────
echo "==> System deps (ffmpeg, git, venv tooling)"
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

# ── 3. Build a FRESH, ISOLATED venv (no system-site-packages) ────────────────
# We deliberately do NOT inherit system packages — that's what kept breaking us.
if [ -d "$VENV" ]; then
  echo "==> Removing old venv for a clean build"
  rm -rf "$VENV"
fi
echo "==> Creating isolated venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "✓ venv active: $(which python3)"
pip install --quiet --upgrade pip wheel setuptools

# ── 4. Install the torch stack FIRST, matched to the pod's CUDA 12.4 ─────────
# echo "==> Installing PyTorch (CUDA 12.4 build) — a few minutes"
# pip install --quiet torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
#     --index-url https://download.pytorch.org/whl/cu124

# ── 5. Install a KNOWN-GOOD huggingface_hub BEFORE transformers ──────────────
# This is the version the repo's transformers expects; installing it first
# prevents the broken inherited one from ever being imported.
echo "==> Installing huggingface_hub 0.24.7"
pip install --quiet "huggingface-hub==0.24.7"

# ── 6. Install the repo's own requirements (ground truth) ────────────────────
echo "==> Installing repo requirements"
pip install --quiet -r "$REPO/requirements.txt"

# ── 7. Re-pin the deps that the repo's loose constraints can drift on ────────
# requirements.txt says transformers>=4.50 + diffusers==0.33.0, which conflict
# on FLAX_WEIGHTS_NAME. We pin a known-compatible pair and re-assert hub.
echo "==> Pinning compatible transformers / diffusers / hub"
pip install --quiet \
    "transformers==4.50.0" \
    "diffusers==0.33.0" \
    "huggingface-hub==0.24.7" \
    "tokenizers>=0.21,<0.22"

# ── 8. The single real patch: restore FLAX_WEIGHTS_NAME for diffusers 0.33 ───
echo "==> Patching FLAX_WEIGHTS_NAME into venv transformers"
python3 - <<'PY'
import transformers.utils as tu
if not hasattr(tu, "FLAX_WEIGHTS_NAME"):
    with open(tu.__file__, "a") as f:
        f.write('\nFLAX_WEIGHTS_NAME = "flax_model.msgpack"\n')
    print("   patched")
else:
    print("   already present")
PY

# ── 9. FlashAttention (optional speedup; never blocks) ───────────────────────
echo "==> FlashAttention (optional)"
pip install --quiet ninja 2>/dev/null || true
pip install --quiet flash-attn --no-build-isolation 2>/dev/null \
    || echo "   (skipped — model still runs without it)"

# ── 10. Link weights into the repo ───────────────────────────────────────────
echo "==> Linking weights"
mkdir -p "$REPO/weights"
rm -rf "$REPO/weights/ckpts"
ln -sf "$MODELS/ckpts" "$REPO/weights/ckpts"
[ -f "$REPO/weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt" ] \
    && echo "✓ Weights linked" || { echo "✗ Weight link failed"; exit 1; }

# ── 11. Verify EVERYTHING that previously failed, in one place ───────────────
echo "==> Verifying full import chain"
python3 - <<'PY'
import torch;            print("✓ torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
import transformers;     print("✓ transformers", transformers.__version__)
import diffusers;        print("✓ diffusers", diffusers.__version__)
import huggingface_hub;  print("✓ huggingface_hub", huggingface_hub.__version__)
from transformers.utils import FLAX_WEIGHTS_NAME;       print("✓ FLAX_WEIGHTS_NAME")
from diffusers.hooks import apply_group_offloading;     print("✓ diffusers.hooks")
import gradio;           print("✓ gradio", gradio.__version__)
PY

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Setup complete — every import verified."
echo "  Launch the UI:  bash $WORKSPACE/launch_ui.sh"
echo "════════════════════════════════════════════════════════"