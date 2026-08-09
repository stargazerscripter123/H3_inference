#!/bin/bash
# 5090 backend installs — mirrors the 6000a recipe with all known fixes baked in.
set -u
SGL_PY=$HOME/miniconda3/envs/h3_sglang_NV_py312/bin/python
VLM_PY=$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python
SRC=$HOME/data/dropbox/CV/h3/src

# wait for prerequisites (envs + src rsync)
for i in $(seq 1 60); do
  [ -x "$SGL_PY" ] && [ -x "$VLM_PY" ] && [ -d "$SRC/sglang/python" ] && [ -d "$SRC/vllm-omni" ] && break
  sleep 20
done
[ -x "$SGL_PY" ] || { echo "PREREQ_TIMEOUT"; exit 1; }

echo "=== [vllm] vllm==0.26.0 + vllm-omni editable ==="
$VLM_PY -m pip install --no-input -q uv
$VLM_PY -m uv pip install --python "$VLM_PY" "vllm==0.26.0" --torch-backend=auto 2>&1 | tail -2
cd "$SRC/vllm-omni"
$VLM_PY -m uv pip install --python "$VLM_PY" -e . 2>&1 | tail -2
$VLM_PY - <<'EOF'
import vllm, vllm_omni, pathlib
print("vllm", vllm.__version__, "| vllm_omni", getattr(vllm_omni, "__version__", "?"))
print("minimax_h3:", (pathlib.Path(vllm_omni.__file__).parent / "diffusion/models/minimax_h3").exists())
import torch
print("torch", torch.__version__, "| sm:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
EOF
echo VLLM_5090_DONE

echo "=== [sglang] editable + diffusion ==="
$SGL_PY -m pip install --no-input -q uv ninja nvidia-cuda-nvcc-cu13 nvidia-cuda-runtime-cu13
cd "$SRC/sglang"
SGLANG_BUILD_RUST_EXTS=none $SGL_PY -m uv pip install --python "$SGL_PY" -e "python[diffusion]" --prerelease=allow 2>&1 | tail -2
LIB=$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13/lib
[ -f "$LIB/libcudart.so.13" ] && ln -sfn "$LIB/libcudart.so.13" "$LIB/libcudart.so"
$SGL_PY - <<'EOF'
import sglang
from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3
print("sglang", sglang.__version__, "minimax_h3 OK")
EOF
echo SGLANG_5090_DONE
