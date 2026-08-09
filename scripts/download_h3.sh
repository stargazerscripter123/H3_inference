#!/bin/bash
# MiniMax H3 weight downloader — curl resumable, size-verified.
# Usage: download_h3.sh <profile: 6000a|5090>
set -uo pipefail
PROFILE="${1:?need profile 6000a|5090}"
BASE="$HOME/data/dropbox/CV/h3"
MODELS="$BASE/ComfyUI/models"
HF="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"

mkdir -p "$MODELS/diffusion_models" "$MODELS/text_encoders" "$MODELS/vae"

# path|bytes
COMMON="
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors|20970379616
vae/minimax_h3_video_vae_fp16.safetensors|5207808496
vae/minimax_h3_audio_vae_fp32.safetensors|605254808
"
TE_INT8="text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|27141342152"
TE_NVFP4="text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|15687142551"

case "$PROFILE" in
  6000a) LIST="$COMMON $TE_INT8" ;;
  5090)  LIST="$COMMON $TE_NVFP4 $TE_INT8" ;;
  *) echo "unknown profile $PROFILE"; exit 1 ;;
esac

fail=0
for entry in $LIST; do
  path="${entry%%|*}"; want="${entry##*|}"
  dst="$MODELS/$path"
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ]; then echo "SKIP (complete) $path"; continue; fi
  echo "DOWNLOAD $path ($want bytes, have $have)"
  ok=0
  for attempt in $(seq 1 20); do
    curl -sSL --fail -C - --retry 5 --retry-delay 5 --connect-timeout 20 -o "$dst" "$HF/$path" && ok=1 && break
    echo "  retry $attempt for $path"; sleep 10
  done
  have=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$ok" = "1" ] && [ "$have" = "$want" ]; then
    echo "OK $path"
  else
    echo "FAIL $path (have $have want $want)"; fail=1
  fi
done

df -h "$HOME" | tail -1
if [ "$fail" = "0" ]; then echo "DOWNLOAD_ALL_DONE $(hostname)"; else echo "DOWNLOAD_HAD_FAILURES $(hostname)"; fi
