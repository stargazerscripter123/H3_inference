#!/bin/bash
# Runs after downloads+pip finish: apply stride patch, verify env, merge Turbo LoRA.
set -u
B=/workspace/h3
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for base download..."
while pgrep -f "h[f] download MiniMaxAI" >/dev/null; do sleep 20; done
SH=$(ls $B/base/MiniMax-H3/FL2VA/transformer/*.safetensors 2>/dev/null | wc -l)
log "base download done: $(du -sBG $B/base | cut -f1), transformer shards=$SH"
[ "$SH" -eq 13 ] || { log "FATAL: expected 13 transformer shards, got $SH"; exit 1; }

log "waiting for pip install..."
while pgrep -f "pip install" >/dev/null; do sleep 20; done
log "pip done"

cd $B/src-vllm-omni || { log "FATAL: src missing"; exit 1; }
log "head: $(git log --oneline -1)"
if grep -q "as_strided" vllm_omni/diffusion/offloader/distributed_layerwise_backend.py \
   && sed -n "/class PinnedResidentLayerGroup/,/def offload/p" vllm_omni/diffusion/offloader/distributed_layerwise_backend.py | grep -q as_strided; then
  log "stride fix already upstream — skipping patch"
else
  git apply $B/scripts/pr5910_resident_stride_fix.patch && log "stride patch applied" || { log "FATAL: patch failed"; exit 1; }
fi

$B/env/bin/python -c "import vllm, vllm_omni; print(\"IMPORT_OK\", vllm.__version__, vllm_omni.__file__)" || exit 1

log "merging Turbo LoRA..."
$B/env/bin/pip install -q safetensors
$B/env/bin/python $B/scripts/merge_turbo_lora.py \
  --lora $B/loras/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  --base $B/base/MiniMax-H3/FL2VA \
  --dst $B/merged/MiniMax-H3-Turbo-v4s600ema \
  --comfy-single /nonexistent --lora-revision afc0346 || { log "FATAL: merge failed"; exit 1; }
log "POST_SETUP_COMPLETE"
