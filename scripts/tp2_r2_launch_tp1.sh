#!/bin/bash
# TP1 control experiment for TP2 noise localization (round-2).
# Identical to h3_switch_5090.sh vllm EXCEPT: 1 GPU, TP1, TE-TP1, VAE parallel off.
set -u
BASE="$HOME/data/dropbox/CV/h3"
MODEL_ROOT="$BASE/models_official/MiniMax-H3"
VLM_BIN="$HOME/miniconda3/envs/h3_vllm_NV_py312/bin"
RES="${VLLM_DLO_RESIDENT:-20}"
cd "$BASE"
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  VLLM_DISABLED_KERNELS="${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}" \
  VLLM_TEST_FORCE_FP8_MARLIN="${VLLM_FORCE_MARLIN:-0}" \
  VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}" \
  "$VLM_BIN/vllm" serve "$MODEL_ROOT/FL2VA" \
    --omni --host 127.0.0.1 --port 8091 --trust-remote-code \
    --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
    --usp 1 --ring 1 \
    --quantization fp8 \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "$RES" \
    --vae-use-tiling \
    --diffusion-attention-backend TORCH_SDPA \
    --enforce-eager \
    > "$BASE/logs/vllm_server_tp1.log" 2>&1 < /dev/null &
echo $! > "$BASE/run/vllm.pid"
# 与 h3_switch_5090.sh 共享 :8091: 必须声明自己的变体,否则下一次
# vllm-turbo 会看到陈旧 variant + 端口健康而空操作,静默服错 checkpoint。
printf %s tp1-base > "$BASE/run/vllm.variant"
printf %s "$MODEL_ROOT/FL2VA" > "$BASE/run/vllm.model"
echo "vllm TP1 launched pgid $(cat "$BASE/run/vllm.pid")"
