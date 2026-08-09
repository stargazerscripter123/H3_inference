#!/bin/bash
# Launch headless ComfyUI for MiniMax H3.
# Usage: launch_comfy.sh <gpu_index> [extra comfy args...]
set -euo pipefail
GPU="${1:?need gpu index}"
PORT="${2:-8188}"
shift 2 2>/dev/null || shift || true
BASE="$HOME/data/dropbox/CV/h3"
ENVPY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"

if curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" > /dev/null 2>&1; then
  echo "ComfyUI already running on :$PORT"
  exit 0
fi

cd "$BASE/ComfyUI"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$ENVPY" main.py \
  --listen 127.0.0.1 --port "$PORT" \
  --output-directory "$BASE/outputs" \
  --input-directory "$BASE/inputs" \
  "$@" > "$BASE/logs/comfyui.log" 2>&1 < /dev/null &
echo "launched pid $! on GPU $GPU"

for i in $(seq 1 60); do
  sleep 5
  if curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" > /dev/null 2>&1; then
    echo "READY after $((i*5))s"
    exit 0
  fi
  if ! kill -0 "$!" 2>/dev/null; then
    echo "DIED — tail of log:"; tail -30 "$BASE/logs/comfyui.log"; exit 1
  fi
done
echo "NOT_READY after 300s — tail of log:"; tail -30 "$BASE/logs/comfyui.log"; exit 1
