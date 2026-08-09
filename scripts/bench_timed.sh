#!/bin/bash
# Formal perf protocol: 1 exact-shape warmup (not recorded) + N timed runs,
# FIXED seed/prompt/shape (engines have no ComfyUI-style graph cache; verified
# by distinct wall times across identical requests).
# Usage: bench_timed.sh <engine: vllm|sglang> <label> <nfe> [timed_n=3]
set -u
ENGINE="${1:?}"; LABEL="${2:?}"; NFE="${3:?}"; N="${4:-3}"
BASE="$HOME/data/dropbox/CV/h3"
PY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
OUTDIR="$BASE/outputs/bench_${LABEL}_nfe${NFE}"
mkdir -p "$OUTDIR"
run_one() { # $1 = tag
  local OUT="$OUTDIR/$1.mp4"
  if [ "$ENGINE" = "vllm" ]; then
    "$PY" "$BASE/scripts/run_fl2va_vllm.py" \
      --first "$BASE/inputs/first_864x480.png" --last "$BASE/inputs/last_864x480.png" \
      --prompt-file "$BASE/workflows/turbo_bench_prompt.txt" \
      --nfe "$NFE" --seed 0 --width 864 --height 480 --out "$OUT"
  else
    "$PY" "$BASE/scripts/run_fl2va_sglang.py" \
      --first "$BASE/inputs/first_864x480.png" --last "$BASE/inputs/last_864x480.png" \
      --prompt-file "$BASE/workflows/turbo_bench_prompt.txt" \
      --nfe "$NFE" --seed 0 --short-edge 768 --out "$OUT"
  fi
}
echo "### bench $LABEL nfe=$NFE: warmup"
run_one warmup
for i in $(seq 1 "$N"); do
  echo "### bench $LABEL nfe=$NFE: timed$i"
  run_one "timed$i"
done
echo "### walls (median of timed):"
for f in "$OUTDIR"/timed*.mp4.manifest.json; do
  grep -oE '"wall_s": [0-9.]+' "$f"
done
echo "BENCH_DONE $LABEL nfe=$NFE"
