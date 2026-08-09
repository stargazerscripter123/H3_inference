#!/bin/bash
# ckpt850 vs v4s600ema A/B via SGLang dynamic LoRA hot-swap (/v1/set_lora).
# Motion-heavy cases only (fight, scope, bridge), NFE 4 and 6, 768P, seed 0.
# Requires the sglang-lora flavor to be up (dynamic mode, v4s600ema loaded).
set -u
BASE="$HOME/data/dropbox/CV/h3"
PY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
LORAS="/home/isaac/Data/h3_weights/loras"
set_lora() { # $1 = nickname, $2 = path
  curl -s -m 600 -X POST http://127.0.0.1:30010/v1/set_lora \
    -H "Content-Type: application/json" \
    -d "{\"lora_nickname\": \"$1\", \"lora_path\": \"$2\", \"strength\": 1.0, \"merge_mode\": \"dynamic\"}"
  echo
}
run_case() { # $1 = ckpt tag, $2 = case, $3 = nfe
  local OUTDIR="$BASE/outputs/stageA_sgl_dyn_${1}_nfe${3}"
  mkdir -p "$OUTDIR"
  local OUT="$OUTDIR/${2}_seed0.mp4"
  [ -f "$OUT" ] && { echo "skip $OUT"; return; }
  echo "=== $1 $2 nfe=$3 ==="
  "$PY" "$BASE/scripts/run_fl2va_sglang.py" \
    --first "$BASE/inputs/stage_a_final/${2}_first.png" \
    --last "$BASE/inputs/stage_a_final/${2}_last.png" \
    --prompt-file "$BASE/workflows/stage_a/${2}.txt" \
    --nfe "$3" --seed 0 --short-edge 768 --out "$OUT" || echo "FAILED $1 $2 $3"
}
echo "--- ckpt v4s600ema (already loaded at startup) ---"
for NFE in 4 6; do for C in fight scope bridge; do run_case v4s600 "$C" "$NFE"; done; done
echo "--- hot-swap to ckpt850 ---"
set_lora turbo850 "$LORAS/minimax_h3_turbo_4step_ema_ckpt850.safetensors"
for NFE in 4 6; do for C in fight scope bridge; do run_case ck850 "$C" "$NFE"; done; done
echo "--- swap back to v4s600ema ---"
set_lora turbo "$LORAS/minimax_h3_turbo_v4_step600_ema.safetensors"
echo "AB_CKPT850_DONE"
