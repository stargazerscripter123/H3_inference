#!/bin/bash
# Authenticated single-instance FL2VA download (6000a). No kill logic here —
# caller must ensure no other instance is running.
set -u
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin:$PATH"
DEST=/home/isaac/Data/h3_weights/MiniMax-H3
# clear stale locks from previous killed instances
find "$DEST/.cache" -name "*.lock" -delete 2>/dev/null || true
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/**" --local-dir "$DEST" 2>&1
echo "HF_FL2VA_DOWNLOAD_EXIT=$?"
du -sh "$DEST"
