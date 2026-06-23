# ─────────────────────────────────────────────────────────────────────────────
# Optional: a Docker image for the pod. You don't strictly need this — you can
# run setup_runpod.sh on RunPod's stock PyTorch template instead. But baking an
# image makes pod starts reproducible and faster.
#
# Build & push:
#   docker build -t YOURUSER/avatar-server:latest .
#   docker push YOURUSER/avatar-server:latest
# Then point your RunPod pod template at that image.
# ─────────────────────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    WORKSPACE=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/* && git lfs install

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY ui ./ui
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

# Weights and repos live on the mounted network volume at /workspace, NOT in
# the image (they're too big and change independently). setup_runpod.sh +
# download_models.sh populate the volume the first time.
EXPOSE 8000 8501
CMD ["bash", "scripts/run.sh"]
