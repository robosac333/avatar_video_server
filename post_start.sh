#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# /post_start.sh — RunPod calls this automatically after the pod boots.
# DO NOT rename or move this file on the pod — it must live at /post_start.sh
#
# To install: cp /workspace/avatar_video_server/post_start.sh /post_start.sh
# ─────────────────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
APP="/workspace/avatar_video_server"

echo "==> [post_start] Installing Python deps"
pip install --quiet --break-system-packages -r "$APP/requirements.txt"

echo "==> [post_start] Launching Avatar Studio"
bash "$APP/scripts/run.sh" &

echo "==> [post_start] Done — servers starting in background"
