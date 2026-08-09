#!/bin/bash
# One probe run + frame extraction for TP2 round-2 experiments.
# usage: tp2_r2_probe.sh <tag> [seed] [steps]
set -u
TAG="${1:?need tag}"; SEED="${2:-0}"; STEPS="${3:-12}"
BASE="$HOME/data/dropbox/CV/h3"
OUT="$BASE/outputs/vllm_tp2_r2/${TAG}.mp4"
"$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python" "$BASE/scripts/run_fl2va_vllm.py" \
  --first "$BASE/inputs/first_864x480.png" --last "$BASE/inputs/last_864x480.png" \
  --prompt-file "$BASE/workflows/smoke_prompt.txt" \
  --seed "$SEED" --steps "$STEPS" --width 864 --height 480 \
  --out "$OUT"
rc=$?
if [ $rc -eq 0 ] && [ -f "$OUT" ]; then
  ffmpeg -y -loglevel error -i "$OUT" -ss 2 -frames:v 1 "$BASE/outputs/vllm_tp2_r2/${TAG}_f2s.png"
  ffmpeg -y -loglevel error -i "$OUT" -frames:v 1 "$BASE/outputs/vllm_tp2_r2/${TAG}_f0.png"
  echo "FRAMES_OK ${TAG}"
fi
exit $rc
