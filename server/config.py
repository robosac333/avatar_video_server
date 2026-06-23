"""
Central configuration. Everything path- or tunable-related lives here so the
rest of the code never hardcodes a directory. Override any value with an
environment variable of the same name (see .env.example).
"""
import os
from pathlib import Path

# ── Base directories ────────────────────────────────────────────────────────
# On a RunPod pod, attach a Network Volume at /workspace so weights survive
# pod restarts. Everything heavy lives under there.
WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace"))

# Model weights (downloaded once onto the network volume)
MODELS_DIR = Path(os.getenv("MODELS_DIR", WORKSPACE / "models"))

# Where the cloned official repos live
REPOS_DIR = Path(os.getenv("REPOS_DIR", WORKSPACE / "repos"))

# Runtime scratch: uploads in, finished videos out, per-job working dirs
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", WORKSPACE / "runtime"))
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
JOB_DIR = RUNTIME_DIR / "jobs"

for _d in (UPLOAD_DIR, OUTPUT_DIR, JOB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Weight subpaths (must match download_models.sh) ─────────────────────────
WAN_BASE_CKPT = MODELS_DIR / "Wan2.1-I2V-14B-480P"
WAV2VEC_DIR = MODELS_DIR / "chinese-wav2vec2-base"
MULTITALK_CKPT = MODELS_DIR / "MeiGen-MultiTalk"
HUNYUAN_CKPT = MODELS_DIR / "HunyuanVideo-Avatar"

# ── Repo locations (cloned by setup_runpod.sh) ──────────────────────────────
MULTITALK_REPO = REPOS_DIR / "MultiTalk"
HUNYUAN_REPO = REPOS_DIR / "HunyuanVideo-Avatar"

# ── Inference tunables ──────────────────────────────────────────────────────
# Low-VRAM mode lets the 14B MultiTalk model fit on a single 24GB card (RTX 4090)
# by not keeping all DiT params resident. Set to "0" for max offload (slowest,
# smallest footprint); raise it on an A100/H100 with VRAM to spare for speed.
NUM_PERSISTENT_PARAM_IN_DIT = os.getenv("NUM_PERSISTENT_PARAM_IN_DIT", "0")

# TeaCache skips redundant diffusion steps -> big speedup, tiny quality cost.
ENABLE_TEACACHE = os.getenv("ENABLE_TEACACHE", "true").lower() == "true"
TEACACHE_THRESHOLD = os.getenv("TEACACHE_THRESHOLD", "0.3")

# Sampling steps. Fewer = faster. 40 is the repo default; 20–25 is fine for reels.
SAMPLE_STEPS = int(os.getenv("SAMPLE_STEPS", "25"))

# Default output resolution for MultiTalk: 480 or 720. 480 ~2x faster.
DEFAULT_RESOLUTION = os.getenv("DEFAULT_RESOLUTION", "480")

# Which GPU to pin to (single-GPU pod -> "0")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Hard ceiling on audio length we'll accept (seconds). Protects you from a
# 10-minute file silently costing a fortune in compute time.
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "90"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
