#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Download all model weights onto the network volume. Run ONCE per volume.
# Total footprint is large (~80–90 GB), so make sure the volume has room.
#
#   bash scripts/download_models.sh            # downloads everything
#   MODELS_DIR=/workspace/models bash scripts/download_models.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "==> Installing huggingface_hub CLI"
pip install -q "huggingface_hub[cli]"

# Faster, resumable downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q hf_transfer

dl () {  # dl <repo_id> <local_dir>
  echo "==> $1  ->  $MODELS_DIR/$2"
  hf download "$1" --local-dir "$MODELS_DIR/$2" --exclude "*.git*"
}

# ── MultiTalk stack (3 pieces) ───────────────────────────────────────────────
# 1) Wan 2.1 image-to-video base (the diffusion backbone)
dl "Wan-AI/Wan2.1-I2V-14B-480P"            "Wan2.1-I2V-14B-480P"
# 2) Audio encoder MultiTalk uses to read speech
dl "TencentGameMate/chinese-wav2vec2-base" "chinese-wav2vec2-base"
# 3) The MultiTalk weights themselves (the audio->motion adapter)
dl "MeiGen-AI/MeiGen-MultiTalk"            "MeiGen-MultiTalk"

# ── HunyuanVideo-Avatar (single repo, large) ─────────────────────────────────
dl "tencent/HunyuanVideo-Avatar"          "HunyuanVideo-Avatar"

echo "==> All weights downloaded into $MODELS_DIR"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
