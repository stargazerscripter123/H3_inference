#!/bin/bash
# Plan-2 BF16 weights for 6000a. Waits for the smoke-test download to finish
# first so it never competes with it for bandwidth.
set -uo pipefail
BASE="$HOME/data/dropbox/CV/h3"
MODELS="$BASE/ComfyUI/models"
# BF16 files live on the second NVMe (602G free) to spare the 93%-full root disk;
# symlinked back into ComfyUI/models below.
STORE="/home/isaac/Data/h3_weights"
HF="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
mkdir -p "$STORE/diffusion_models" "$STORE/text_encoders"

# wait until the smoke-critical download script is done
while pgrep -f "download_h3.sh" > /dev/null; do sleep 60; done
echo "smoke download finished, starting BF16 downloads at $(date)"

fail=0
for entry in \
  "diffusion_models/minimax_h3_fl2va_bf16.safetensors|66280487368" \
  "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors|51506295256"; do
  path="${entry%%|*}"; want="${entry##*|}"
  dst="$STORE/$path"
  ln -sfn "$dst" "$MODELS/$path"
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ]; then echo "SKIP (complete) $path"; continue; fi
  echo "DOWNLOAD $path ($want bytes, have $have)"
  ok=0
  for attempt in $(seq 1 40); do
    curl -sSL --fail -C - --retry 5 --retry-delay 5 --connect-timeout 20 -o "$dst" "$HF/$path" && ok=1 && break
    echo "  retry $attempt for $path"; sleep 10
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$ok" = "1" ] && [ "$have" = "$want" ]; then echo "OK $path"; else echo "FAIL $path (have $have want $want)"; fail=1; fi
done
df -h /home/isaac/Data | tail -1
df -h "$HOME" | tail -1
[ "$fail" = "0" ] && echo "BF16_DOWNLOAD_DONE $(hostname)" || echo "BF16_DOWNLOAD_FAILED $(hostname)"
