#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup on a fresh RunPod pod. Clones the official model repos and
# wires the MultiTalk weights into the Wan base checkpoint the way the repo
# expects. Run AFTER download_models.sh.
#
#   bash scripts/setup_runpod.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"
REPOS_DIR="${REPOS_DIR:-$WORKSPACE/repos}"
mkdir -p "$REPOS_DIR"

echo "==> System deps (ffmpeg for audio probing/merging)"
apt-get update -qq && apt-get install -y -qq ffmpeg git git-lfs
git lfs install

# ── MultiTalk repo ───────────────────────────────────────────────────────────
if [ ! -d "$REPOS_DIR/MultiTalk" ]; then
  echo "==> Cloning MultiTalk"
  git clone https://github.com/MeiGen-AI/MultiTalk.git "$REPOS_DIR/MultiTalk"
fi

echo "==> Installing MultiTalk python deps"
pip install -q -r "$REPOS_DIR/MultiTalk/requirements.txt" || true
# FlashAttention 2 — big speedup on the transformer blocks. Prebuilt wheel.
pip install -q flash-attn --no-build-isolation || \
  echo "   (flash-attn wheel failed; model still runs, just slower)"

# The repo expects the MultiTalk weight file linked inside the Wan base dir.
WAN="$MODELS_DIR/Wan2.1-I2V-14B-480P"
MT="$MODELS_DIR/MeiGen-MultiTalk"
if [ -d "$WAN" ] && [ -d "$MT" ]; then
  echo "==> Linking MultiTalk weights into Wan base checkpoint"
  # keep an untouched copy of the base index, then expose MultiTalk files
  [ -f "$WAN/diffusion_pytorch_model.safetensors.index.json" ] && \
    cp -n "$WAN/diffusion_pytorch_model.safetensors.index.json" \
          "$WAN/diffusion_pytorch_model.safetensors.index.json.bak" || true
  for f in "$MT"/*; do
    ln -sf "$f" "$WAN/$(basename "$f")"
  done
fi

# ── HunyuanVideo-Avatar repo ─────────────────────────────────────────────────
if [ ! -d "$REPOS_DIR/HunyuanVideo-Avatar" ]; then
  echo "==> Cloning HunyuanVideo-Avatar"
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git \
            "$REPOS_DIR/HunyuanVideo-Avatar"
fi
echo "==> Installing HunyuanVideo-Avatar python deps"
pip install -q -r "$REPOS_DIR/HunyuanVideo-Avatar/requirements.txt" || true

echo "==> Setup complete."
echo "    Repos:   $REPOS_DIR"
echo "    Weights: $MODELS_DIR"
echo "    Next:    start the server (see README 'Running')."
