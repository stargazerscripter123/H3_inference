#!/bin/bash
# Smoke: vLLM turbo (merged ckpt) NFE sweep on the ToS standard pair, seed 0.
# Usage: smoke_vllm_turbo.sh <variant-tag e.g. bf16|fp8> [nfe ...]
set -u
TAG="${1:?need variant tag}"; shift
NFES=("${@:-4}")
[ $# -eq 0 ] && NFES=(4 6 8)
BASE="$HOME/data/dropbox/CV/h3"
PY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
LOG="$BASE/logs/vllm_server.log"
cd "$BASE"
for NFE in "${NFES[@]}"; do
  OUTDIR="outputs/turbo_vllm_${TAG}_nfe${NFE}"
  mkdir -p "$OUTDIR"
  MARK=$(wc -l < "$LOG")
  echo "=== NFE=$NFE -> $OUTDIR ==="
  "$PY" scripts/run_fl2va_vllm.py \
    --first inputs/first_864x480.png --last inputs/last_864x480.png \
    --prompt-file workflows/turbo_bench_prompt.txt \
    --nfe "$NFE" --seed 0 --width 864 --height 480 \
    --out "$OUTDIR/tos_seed0.mp4" || exit 1
  echo "--- server log since request (denoise evidence) ---"
  tail -n +"$MARK" "$LOG" | grep -iE "denois|step|sigma|it/s|s/it" | tail -6
done
