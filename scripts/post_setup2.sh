#!/bin/bash
# Finish RunPods setup after weights are on disk: editable install of the
# vllm-omni PR, stride patch, Turbo LoRA merge.
set -u
B=/workspace/h3
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log 'installing build deps + editable vllm-omni'
"$B/env/bin/pip" install -q setuptools_scm wheel aenum safetensors || { log 'FATAL: build deps'; exit 1; }
cd "$B/src-vllm-omni" || { log 'FATAL: src missing'; exit 1; }
log "head: $(git log --oneline -1)"

if sed -n '/class PinnedResidentLayerGroup/,/def offload/p' \
     vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -q as_strided; then
  log 'stride fix already upstream - skipping patch'
else
  git apply "$B/scripts/pr5910_resident_stride_fix.patch" \
    && log 'stride patch applied' || { log 'FATAL: patch failed'; exit 1; }
fi

SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 "$B/env/bin/pip" install -e . --no-build-isolation \
  > "$B/logs/install_omni.log" 2>&1 || { log 'FATAL: editable install'; tail -15 "$B/logs/install_omni.log"; exit 1; }
log 'editable install ok'

"$B/env/bin/python" -c 'import vllm, vllm_omni; print("IMPORT_OK", vllm.__version__, vllm_omni.__file__)' || exit 1

log 'merging Turbo LoRA'
"$B/env/bin/python" "$B/scripts/merge_turbo_lora.py" \
  --lora "$B/loras/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors" \
  --base "$B/base/MiniMax-H3/FL2VA" \
  --dst "$B/merged/MiniMax-H3-Turbo-v4s600ema" \
  --comfy-single /nonexistent --lora-revision afc0346 || { log 'FATAL: merge'; exit 1; }
log 'POST_SETUP2_COMPLETE'
