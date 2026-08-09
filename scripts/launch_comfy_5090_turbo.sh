#!/bin/bash
# 5090 turbo worker: GPU0, port 8189, speedup flags per doc/speedup_5090.md.
# Baseline worker (GPU1:8188) is a separate process and is never touched here.
set -euo pipefail
BASE="$HOME/data/dropbox/CV/h3"
ENVPY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
PORT=8189
if curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" > /dev/null 2>&1; then
  echo "turbo worker already running on :$PORT"
  exit 0
fi
cd "$BASE/ComfyUI"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 setsid nohup "$ENVPY" main.py \
  --listen 127.0.0.1 --port "$PORT" \
  --preview-method none \
  --async-offload 2 \
  --reserve-vram 1.5 \
  --output-directory "$BASE/outputs" --input-directory "$BASE/inputs" \
  > "$BASE/logs/comfyui_turbo.log" 2>&1 < /dev/null &
echo "launched $! on GPU0 :$PORT"
for i in $(seq 1 36); do
  sleep 5
  curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" > /dev/null 2>&1 && { echo "READY after $((i*5))s"; exit 0; }
done
echo "NOT_READY — log tail:"; tail -20 "$BASE/logs/comfyui_turbo.log"; exit 1
