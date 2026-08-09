#!/bin/bash
# 6000a 4-GPU launcher for BF16 sharded inference (port 8288)
set -euo pipefail
BASE="$HOME/data/dropbox/CV/h3"
ENVPY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
LIBDIR="$HOME/miniconda3/envs/h3_comfy_NV_py312/lib/python3.12/site-packages/nvidia/cu13/lib"
if curl -s -m 3 "http://127.0.0.1:8288/system_stats" > /dev/null 2>&1; then echo "already running"; exit 0; fi
cd "$BASE/ComfyUI"
LD_LIBRARY_PATH="$LIBDIR" CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup "$ENVPY" main.py \
  --listen 127.0.0.1 --port 8288 \
  --output-directory "$BASE/outputs" --input-directory "$BASE/inputs" \
  > "$BASE/logs/comfyui.log" 2>&1 < /dev/null &
echo "launched $!"
