#!/bin/bash
# Turbo benchmark driver (runs ON popos-5090 against the turbo worker :8189).
# Usage: bench_turbo.sh <tag> <dit_file> <steps> <thresh> <mode>
#   mode=quality : seed 0, canonical prompt (output kept for quality compare; also serves as warmup)
#   mode=timedN  : seed N + prompt suffix -> defeats ComfyUI node cache so the
#                  full TE-encode + denoise + decode path re-executes (true warm latency)
set -u
BASE=~/data/dropbox/CV/h3
PY=~/miniconda3/envs/h3_comfy_NV_py312/bin/python
TAG=$1; DIT=$2; STEPS=$3; THRESH=$4; MODE=$5

PROMPT=$(cat "$BASE/workflows/smoke_prompt.txt")
SEED=0
if [[ "$MODE" == timed* ]]; then
  N=${MODE#timed}
  SEED=$N
  PROMPT="$PROMPT (timing variation $N)"
fi

OUT=$($PY "$BASE/scripts/run_fl2va.py" \
  --server http://127.0.0.1:8189 \
  --dit "$DIT" \
  --te qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --first first_864x480.png --last last_864x480.png \
  --prompt "$PROMPT" \
  --width 864 --height 480 --length 124 --steps "$STEPS" --seed "$SEED" \
  --teacache-thresh "$THRESH" \
  --gpu-index 0 --prefix "h3_bench/${TAG}_${MODE}" 2>&1 | tail -1)

SIT=$(grep -oE "${STEPS}/${STEPS} \[[^]]*, *[0-9.]+s/it\]" "$BASE/logs/comfyui_turbo.log" | tail -1 | grep -oE "[0-9.]+s/it" | tail -1)
echo "$TAG $MODE | dit=$DIT steps=$STEPS thresh=$THRESH | $OUT | ${SIT:-n/a}" | tee -a "$BASE/logs/bench_results.log"
