#!/bin/bash
# Bootstrap a fresh RunPods 4x5090 pod for MiniMax-H3 Turbo TP4 serving.
# Everything persistent lives on /workspace (survives pod restart); / is ephemeral.
# Usage: run ON the pod:  bash setup_runpods.sh <HF_TOKEN>
# After it finishes:      bash /workspace/h3/scripts/h3_switch_runpods.sh bf16
# (Scripts h3_switch_runpods.sh / probe.sh / run_fl2va_vllm.py / merge_turbo_lora.py
#  are pushed from the Mac project by h3_generate.py deployment docs — see
#  claude_history/09_runpods_tp4/FINAL.md.)
set -euo pipefail
TOKEN="${1:?need HF token}"
B=/workspace/h3
PR_HEAD=1a9b9c2c13c97763567405f43c5fee994c43ab13   # vllm-omni PR#5910
# CRITICAL: the pod's driver (570.x) only speaks CUDA 12.8. PyPI's vllm 0.26.0
# pulls torch 2.11.0+cu130 (CUDA *13*) and dies with "driver too old (found
# version 12080)". vLLM ships a +cu129 wheel per release; CUDA 12.9 is still
# major 12, so minor-version compatibility runs it on a 12.8 driver.
# Check `nvidia-smi` first — on a driver >=580 pod the plain `vllm==0.26.0` works.
VLLM_WHL=https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
TORCH_INDEX=https://download.pytorch.org/whl/cu129

mkdir -p "$B"/{base,merged,loras,outputs,logs,scripts,inputs,workflows,run} /root/hf-cache
echo -n "$TOKEN" > /root/hf-cache/token

python3 -m venv "$B/env"
"$B/env/bin/pip" install -q --upgrade pip
"$B/env/bin/pip" install -q "huggingface_hub[hf_transfer]"

# weights (resumable). Only FL2VA; Ref2VA excluded by design (storage plan, TP4 note §10).
env HF_HOME=/root/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 "$B/env/bin/hf" download \
  MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir "$B/base/MiniMax-H3" &
env HF_HOME=/root/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 "$B/env/bin/hf" download \
  larryvrh/MiniMax-H3-Turbo-Lora --revision afc0346 \
  --include "minimax_h3_turbo_v4_step600_ema.safetensors" \
  --local-dir "$B/loras/MiniMax-H3-Turbo-Lora" &

"$B/env/bin/pip" install "$VLLM_WHL" --extra-index-url "$TORCH_INDEX"
"$B/env/bin/pip" install setuptools_scm wheel aenum safetensors
"$B/env/bin/python" -c 'import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, torch.version.cuda)'
cd "$B" && { [ -d src-vllm-omni ] || git clone https://github.com/vllm-project/vllm-omni.git src-vllm-omni; }
cd src-vllm-omni && git fetch origin pull/5910/head:pr5910 && git checkout "$PR_HEAD"
# If this head still repoints resident DLO weights with .view() (stride bug, see
# claude_history/08_5090_tp2/FINAL.md), apply scripts/pr5910_resident_stride_fix.patch.
grep -q "as_strided" vllm_omni/diffusion/offloader/distributed_layerwise_backend.py \
  || echo "WARNING: check resident-group repoint for the stride bug before serving"
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 "$B/env/bin/pip" install -e . --no-build-isolation

wait  # downloads
# merge Turbo LoRA (deterministic; verify provenance hashes against 6000a manifest)
"$B/env/bin/pip" install -q safetensors
"$B/env/bin/python" "$B/scripts/merge_turbo_lora.py" \
  --lora "$B/loras/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors" \
  --base "$B/base/MiniMax-H3/FL2VA" \
  --dst "$B/merged/MiniMax-H3-Turbo-v4s600ema" \
  --comfy-single /nonexistent --lora-revision afc0346
echo "setup complete — launch with: bash $B/scripts/h3_switch_runpods.sh bf16"
