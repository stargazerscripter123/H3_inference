#!/bin/bash
# Install SGLang (diffusion) and vLLM-Omni into their dedicated envs on 6000a.
# Per official docs:
#   sglang:   uv pip install "sglang[diffusion]" --prerelease=allow
#   vllm-omni: try PyPI 0.26.0 first; fall back to source (main) if the
#              minimax_h3 model is absent from the wheel.
set -u
SGL_PY=$HOME/miniconda3/envs/h3_sglang_NV_py312/bin/python
VLM_PY=$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python

echo "=== [sglang] install ==="
$SGL_PY -m pip install --no-input -q uv
$SGL_PY -m uv pip install --python "$SGL_PY" "sglang[diffusion]" --prerelease=allow 2>&1 | tail -3
echo "--- sglang sanity ---"
$SGL_PY - <<'EOF'
import importlib, importlib.util
import sglang
print("sglang", sglang.__version__)
spec = importlib.util.find_spec("sglang.multimodal_gen")
print("multimodal_gen present:", spec is not None)
try:
    from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3
    print("minimax_h3 pipeline config: OK")
except Exception as e:
    print("minimax_h3 pipeline config FAIL:", e)
EOF
echo "SGLANG_INSTALL_SECTION_DONE"

echo "=== [vllm-omni] install ==="
$VLM_PY -m pip install --no-input -q uv
$VLM_PY -m uv pip install --python "$VLM_PY" vllm-omni==0.26.0 2>&1 | tail -3
echo "--- vllm-omni sanity ---"
$VLM_PY - <<'EOF'
import importlib.util, pathlib
import vllm_omni
print("vllm_omni", getattr(vllm_omni, "__version__", "?"))
root = pathlib.Path(vllm_omni.__file__).parent
hits = list(root.rglob("*minimax*"))
print("minimax files in wheel:", len(hits))
for h in hits[:5]:
    print("  ", h.relative_to(root))
EOF
echo "VLLM_INSTALL_SECTION_DONE"
