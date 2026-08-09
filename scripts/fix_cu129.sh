#!/bin/bash
# The pod's driver is 570.195.03 => CUDA 12.8. PyPI vllm 0.26.0 pulls torch
# 2.11.0+cu130 (CUDA *13*), which no 12.x driver can load ("driver too old,
# found version 12080"). vLLM publishes a +cu129 wheel per release; CUDA 12.9 is
# still CUDA major 12, so minor-version compatibility lets it run on a 12.8
# driver. Build a parallel venv, verify, then swap it in atomically.
set -u
B=/workspace/h3
NEW="$B/env129"
WHL=https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
log(){ echo "[$(date +%H:%M:%S)] $*"; }

rm -rf "$NEW"
python3 -m venv "$NEW"
"$NEW/bin/pip" install -q --upgrade pip

log "installing vllm 0.26.0+cu129"
"$NEW/bin/pip" install "$WHL" --extra-index-url https://download.pytorch.org/whl/cu129 \
  > "$B/logs/install_cu129.log" 2>&1 || { log "FATAL: vllm cu129"; tail -15 "$B/logs/install_cu129.log"; exit 1; }

CUDA_V=$("$NEW/bin/python" -c 'import torch; print(torch.version.cuda)')
log "torch cuda = $CUDA_V"
case "$CUDA_V" in
  12.*) ;;
  *) log "FATAL: expected CUDA 12.x, got $CUDA_V"; exit 1 ;;
esac

"$NEW/bin/python" - <<'PYEOF' || { echo "FATAL: torch cannot init CUDA"; exit 1; }
import torch
assert torch.cuda.is_available(), "cuda not available"
print("GPUS", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PYEOF

log "installing vllm-omni editable + deps"
"$NEW/bin/pip" install -q setuptools_scm wheel aenum safetensors huggingface_hub || exit 1
cd "$B/src-vllm-omni" || exit 1
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0 "$NEW/bin/pip" install -e . --no-build-isolation \
  >> "$B/logs/install_cu129.log" 2>&1 || { log "FATAL: omni editable"; tail -15 "$B/logs/install_cu129.log"; exit 1; }

"$NEW/bin/python" - <<'PYEOF' || exit 1
import vllm, vllm_omni, torch
print("IMPORT_OK vllm", vllm.__version__, "torch", torch.__version__)
PYEOF

log "swapping venv"
[ -d "$B/env" ] && mv "$B/env" "$B/env_cu130_unusable"
mv "$NEW" "$B/env"
"$B/env/bin/python" -c 'import vllm_omni, torch; print("FINAL_OK torch", torch.__version__, torch.version.cuda)'
log "FIX_CU129_COMPLETE"
