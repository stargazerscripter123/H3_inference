#!/bin/bash
# P2 component isolation: DiT FP8 + TE BF16 (component quantization config).
# Same as h3_switch_5090.sh vllm EXCEPT --quantization fp8 replaced by component config.
set -u
BASE="$HOME/data/dropbox/CV/h3"
MODEL_ROOT="$BASE/models_official/MiniMax-H3"
VLM_BIN="$HOME/miniconda3/envs/h3_vllm_NV_py312/bin"
RES="${VLLM_DLO_RESIDENT:-4}"
cd "$BASE"
setsid nohup env CUDA_VISIBLE_DEVICES=0,1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  VLLM_DISABLED_KERNELS="${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}" \
  VLLM_BATCH_INVARIANT=0 \
  "$VLM_BIN/vllm" serve "$MODEL_ROOT/FL2VA" \
    --omni --host 127.0.0.1 --port 8091 --trust-remote-code \
    --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
    --usp 1 --ring 1 \
    --diffusion-quantization-config "{\"transformer\": {\"method\": \"fp8\"}}" \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "$RES" \
    --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
    --diffusion-attention-backend TORCH_SDPA \
    --enforce-eager \
    > "$BASE/logs/vllm_server_p2.log" 2>&1 < /dev/null &
echo $! > "$BASE/run/vllm.pid"
# 与 h3_switch_5090.sh 共享 :8091: 必须声明自己的变体,否则下一次
# vllm-turbo 会看到陈旧 variant + 端口健康而空操作,静默服错 checkpoint。
printf %s p2-base > "$BASE/run/vllm.variant"
printf %s "$MODEL_ROOT/FL2VA" > "$BASE/run/vllm.model"
echo "vllm P2 (DiT FP8 / TE BF16) launched pgid $(cat "$BASE/run/vllm.pid")"
