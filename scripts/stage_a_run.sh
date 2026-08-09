#!/bin/bash
# Stage A batch: run all 8 ToS cases against the CURRENTLY ACTIVE backend.
# Usage: stage_a_run.sh <engine: vllm|sglang> <tag> <nfe> [seed]
#   -> outputs/stageA_<engine>_<tag>_nfe<NFE>/<case>_seed<seed>.mp4
# Case list: sniper(bench pair) bridge scope shout holo sky fight robot
set -u
ENGINE="${1:?vllm|sglang}"; TAG="${2:?tag}"; NFE="${3:?nfe}"; SEED="${4:-0}"
BASE="$HOME/data/dropbox/CV/h3"
PY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
OUTDIR="$BASE/outputs/stageA_${ENGINE}_${TAG}_nfe${NFE}"
mkdir -p "$OUTDIR"
CASES=(sniper bridge scope shout holo sky fight robot)
for CASE in "${CASES[@]}"; do
  if [ "$CASE" = "sniper" ]; then
    FIRST="$BASE/inputs/first_864x480.png"; LAST="$BASE/inputs/last_864x480.png"
    PROMPT="$BASE/workflows/turbo_bench_prompt.txt"
  else
    FIRST="$BASE/inputs/stage_a_final/${CASE}_first.png"
    LAST="$BASE/inputs/stage_a_final/${CASE}_last.png"
    PROMPT="$BASE/workflows/stage_a/${CASE}.txt"
  fi
  OUT="$OUTDIR/${CASE}_seed${SEED}.mp4"
  [ -f "$OUT" ] && { echo "skip existing $OUT"; continue; }
  echo "=== $CASE nfe=$NFE seed=$SEED ==="
  if [ "$ENGINE" = "vllm" ]; then
    "$PY" "$BASE/scripts/run_fl2va_vllm.py" \
      --first "$FIRST" --last "$LAST" --prompt-file "$PROMPT" \
      --nfe "$NFE" --seed "$SEED" --width 864 --height 480 \
      --out "$OUT" || echo "FAILED $CASE"
  else
    "$PY" "$BASE/scripts/run_fl2va_sglang.py" \
      --first "$FIRST" --last "$LAST" --prompt-file "$PROMPT" \
      --nfe "$NFE" --seed "$SEED" --short-edge 768 \
      --out "$OUT" || echo "FAILED $CASE"
  fi
done
echo "STAGE_A_BATCH_DONE $ENGINE $TAG nfe=$NFE seed=$SEED"
