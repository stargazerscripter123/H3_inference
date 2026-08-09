#!/bin/bash
# Authenticated resume download of the FP8 pruned DiT (5090).
set -u
dst=~/data/dropbox/CV/h3/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
want=20958205608
TOKEN=$(cat ~/.cache/huggingface/token)
for i in $(seq 1 40); do
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  [ "$have" = "$want" ] && break
  curl -sSL --fail -C - -H "Authorization: Bearer $TOKEN" \
    --retry 3 --retry-delay 3 --connect-timeout 15 \
    --speed-limit 500000 --speed-time 60 \
    -o "$dst" \
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
  sleep 5
done
have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
if [ "$have" = "$want" ]; then echo FP8_DOWNLOAD_DONE; else echo "FP8_DOWNLOAD_FAILED have=$have"; fi
