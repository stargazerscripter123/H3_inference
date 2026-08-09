#!/bin/bash
# probe.sh <tag> [seed] [nfe]
set -u
TAG="${1:?}"; SEED="${2:-0}"; NFE="${3:-6}"
BASE=/workspace/h3
mkdir -p "$BASE/outputs/bench"
OUT="$BASE/outputs/bench/${TAG}.mp4"
"$BASE/env/bin/python" "$BASE/scripts/run_fl2va_vllm.py" \
  --first "$BASE/inputs/first_864x480.png" --last "$BASE/inputs/last_864x480.png" \
  --prompt-file "$BASE/workflows/smoke_prompt.txt" \
  --seed "$SEED" --nfe "$NFE" --width 864 --height 480 --out "$OUT"
rc=$?
[ $rc -eq 0 ] && ffmpeg -y -loglevel error -i "$OUT" -ss 2 -frames:v 1 "$BASE/outputs/bench/${TAG}_f2s.png" && echo FRAME_OK
exit $rc
