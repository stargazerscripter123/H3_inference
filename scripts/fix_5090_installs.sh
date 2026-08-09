#!/bin/bash
# Repair round: the combined pip line failed on nvcc sdist builds and took
# uv/ninja down with it, so nothing after installed. Split every step.
set -u
SGL_PY=$HOME/miniconda3/envs/h3_sglang_NV_py312/bin/python
VLM_PY=$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python
SRC=$HOME/data/dropbox/CV/h3/src

echo "=== [vllm] complete the editable install (deps like aenum) ==="
$VLM_PY -m pip install --no-input -q uv
cd "$SRC/vllm-omni"
$VLM_PY -m uv pip install --python "$VLM_PY" -e . 2>&1 | tail -3
$VLM_PY - <<'EOF'
import vllm, vllm_omni, pathlib, torch
print("vllm", vllm.__version__, "| vllm_omni", getattr(vllm_omni, "__version__", "?"))
print("minimax_h3:", (pathlib.Path(vllm_omni.__file__).parent / "diffusion/models/minimax_h3").exists())
print("torch", torch.__version__, "| sm:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
EOF
echo VLLM_FIX_DONE

echo "=== [sglang] step-by-step ==="
$SGL_PY -m pip install --no-input -q uv ninja
cd "$SRC/sglang"
SGLANG_BUILD_RUST_EXTS=none $SGL_PY -m uv pip install --python "$SGL_PY" -e "python[diffusion]" --prerelease=allow 2>&1 | tail -3
echo "--- nvcc wheels (binary only) ---"
$SGL_PY -m pip install --no-input -q --only-binary :all: nvidia-cuda-nvcc-cu13 nvidia-cuda-runtime-cu13 \
  && echo "nvcc wheels OK" || echo "NVCC_WHEELS_FAILED (will rsync from 6000a)"
LIB=$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13/lib
[ -f "$LIB/libcudart.so.13" ] && ln -sfn "$LIB/libcudart.so.13" "$LIB/libcudart.so"
$SGL_PY - <<'EOF'
import sglang
from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3
print("sglang", sglang.__version__, "minimax_h3 OK")
EOF
ls $HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc 2>/dev/null && echo "nvcc present" || echo "nvcc MISSING"
echo SGLANG_FIX_DONE
