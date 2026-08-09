#!/bin/bash
# Round-2 TP2-noise cross-validation matrix on 6000a (sm_89, 48G cards).
# Shadow-loads PR#5910 @ b18eeff2 via PYTHONPATH; production env untouched.
# usage: tp2_r2_6000a.sh {tp1|tp2cw|tp2cutlass|p2comp}
set -u
MODE="${1:?need tp1|tp2cw|tp2cutlass|p2comp}"
BASE="$HOME/data/dropbox/CV/h3"
MODEL="/home/isaac/Data/h3_weights/MiniMax-H3/FL2VA"
VLM_BIN="$HOME/miniconda3/envs/h3_vllm_NV_py312/bin"
PRDIR="$BASE/src/vllm-omni-pr5910"
RES="${VLLM_DLO_RESIDENT:-20}"
QUANT_ARGS=(--quantization fp8)
KERNELS=""
case "$MODE" in
  tp1)       DEV=0;   NG=1; TP=1; VAE_ARGS=(--vae-use-tiling); KERNELS="CutlassFP8ScaledMMLinearKernel" ;;
  tp2cw)     DEV=0,1; NG=2; TP=2; VAE_ARGS=(--vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling); KERNELS="CutlassFP8ScaledMMLinearKernel" ;;
  tp2cutlass) DEV=0,1; NG=2; TP=2; VAE_ARGS=(--vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling); KERNELS="" ;;
  p2comp)    DEV=0,1; NG=2; TP=2; VAE_ARGS=(--vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling); KERNELS="CutlassFP8ScaledMMLinearKernel"; QUANT_ARGS=(--diffusion-quantization-config "{\"transformer\": {\"method\": \"fp8\"}}") ;;
  *) echo "unknown mode"; exit 1 ;;
esac
cd "$BASE"
setsid nohup env CUDA_VISIBLE_DEVICES="$DEV" \
  PYTHONPATH="$PRDIR" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  VLLM_DISABLED_KERNELS="$KERNELS" \
  VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}" VLLM_R2_DUMP="${VLLM_R2_DUMP:-0}" \
  "$VLM_BIN/vllm" serve "$MODEL" \
    --omni --host 127.0.0.1 --port 8091 --trust-remote-code \
    --num-gpus "$NG" --tensor-parallel-size "$TP" --text-encoder-tp-size "$TP" \
    --usp 1 --ring 1 \
    "${QUANT_ARGS[@]}" \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "$RES" \
    "${VAE_ARGS[@]}" \
    --diffusion-attention-backend TORCH_SDPA \
    --enforce-eager \
    > "$BASE/logs/vllm_r2_${MODE}.log" 2>&1 < /dev/null &
echo $! > "$BASE/run/vllm_r2.pid"
echo "r2 $MODE launched pgid $(cat "$BASE/run/vllm_r2.pid")"
