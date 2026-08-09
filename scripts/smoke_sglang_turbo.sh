#!/bin/bash
# Smoke: SGLang turbo NFE sweep on the ToS standard pair, seed 0, 768P.
# Usage: smoke_sglang_turbo.sh <tag e.g. merged|dynamic> [nfe ...]
set -u
TAG="${1:?need tag}"; shift
NFES=("${@}")
[ ${#NFES[@]} -eq 0 ] && NFES=(4 6 8)
BASE="$HOME/data/dropbox/CV/h3"
PY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
LOG="$BASE/logs/sglang_server.log"
cd "$BASE"
for NFE in "${NFES[@]}"; do
  OUTDIR="outputs/turbo_sgl_${TAG}_nfe${NFE}"
  mkdir -p "$OUTDIR"
  MARK=$(wc -l < "$LOG")
  echo "=== NFE=$NFE -> $OUTDIR ==="
  "$PY" scripts/run_fl2va_sglang.py \
    --first "$BASE/inputs/first_864x480.png" --last "$BASE/inputs/last_864x480.png" \
    --prompt-file "$BASE/workflows/turbo_bench_prompt.txt" \
    --nfe "$NFE" --seed 0 --short-edge 768 \
    --out "$BASE/$OUTDIR/tos_seed0.mp4" || exit 1
  echo "--- server log since request ---"
  tail -n +"$MARK" "$LOG" | grep -iE "denois|steps=|it/s|s/it|Pixel data|seconds" | tail -5 | cut -c1-160
done
