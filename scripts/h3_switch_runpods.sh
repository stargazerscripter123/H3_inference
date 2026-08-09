#!/bin/bash
# RunPods 4x5090 switcher for H3 serving (doc/TP4_5090.md route).
#
# Model / precision / topology are three ORTHOGONAL dimensions, because on this
# box they trade off independently:
#   - no GPU P2P at all (nvidia-smi topo -p2p r/w = CNS everywhere), so every
#     NCCL collective is host-staged and ring all-reduce traffic per GPU
#     (~2(N-1)/N x S) dominates -> TP4 is not automatically fastest.
#   - DiT weights per card: BF16 TP4 16.5G | BF16 TP2 33G (does NOT fit 32G)
#                           FP8  TP4  8.3G | FP8  TP2 16.5G
#
#   h3_switch_runpods.sh <original|turbo> <bf16|fp8> <tp4|tp2u2> [resident]
#   h3_switch_runpods.sh stop
set -u
WHICH="${1:?need original|turbo|stop}"
B=/workspace/h3
VENV="$B/env/bin"
RUN="$B/run"; mkdir -p "$RUN" "$B/logs"

stop_server() {
  local pf="$RUN/vllm.pid"
  if [ -f "$pf" ]; then
    local pg; pg=$(cat "$pf")
    kill -- -"$pg" 2>/dev/null || kill "$pg" 2>/dev/null || true
    sleep 4; kill -9 -- -"$pg" 2>/dev/null || true
    rm -f "$pf"
  fi
  pgrep -f '[v]llm serve' >/dev/null && { pkill -9 -f '[v]llm serve'; sleep 3; } || true
  echo 'server stopped'
}
[ "$WHICH" = stop ] && { stop_server; exit 0; }

PREC="${2:?need bf16|fp8}"
TOPO="${3:?need tp4|tp2u2}"
RES_IN="${4:-}"

case "$WHICH" in
  original) MODEL="$B/base/MiniMax-H3/FL2VA" ;;
  turbo)    MODEL="$B/merged/MiniMax-H3-Turbo-v4s600ema/FL2VA"
            [ -f "$B/merged/MiniMax-H3-Turbo-v4s600ema/.complete" ] \
              || { echo 'merged checkpoint incomplete'; exit 1; } ;;
  *) echo "unknown model $WHICH"; exit 1 ;;
esac
case "$PREC" in
  bf16) QUANT=(); DEFRES=40 ;;
  fp8)  QUANT=(--quantization fp8); DEFRES=50 ;;
  *) echo "unknown precision $PREC"; exit 1 ;;
esac
case "$TOPO" in
  tp4)   TP=4; USP=1 ;;
  tp2u2) TP=2; USP=2 ;;   # TP pairs land on (0,1)(2,3) = one NUMA node each
  *) echo "unknown topology $TOPO"; exit 1 ;;
esac
RES="${RES_IN:-$DEFRES}"
LABEL="${WHICH}-${PREC}-${TOPO}-r${RES}"

stop_server
cd "$B"
setsid nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  VLLM_DISABLED_KERNELS="${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}" \
  "$VENV/vllm" serve "$MODEL" \
    --omni --host 127.0.0.1 --port 8091 --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size "$TP" --text-encoder-tp-size "${H3_TE_TP:-4}" \
    --usp "$USP" --ring 1 \
    "${QUANT[@]}" \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "$RES" \
    --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
    --diffusion-attention-backend "${VLLM_ATTN:-CUDNN_ATTN}" \
    --enforce-eager \
    > "$B/logs/vllm_${LABEL}.log" 2>&1 < /dev/null &
echo $! > "$RUN/vllm.pid"
echo "vllm($LABEL) launched pgid $(cat "$RUN/vllm.pid") log=logs/vllm_${LABEL}.log"
