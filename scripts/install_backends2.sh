#!/bin/bash
# Round 2: install vllm==0.26.0 (pairs with vllm-omni 0.26.0) and SGLang from
# source main (H3 support not in the 0.5.16 wheel).
set -u
SGL_PY=$HOME/miniconda3/envs/h3_sglang_NV_py312/bin/python
VLM_PY=$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/python
SRC=$HOME/data/dropbox/CV/h3/src
mkdir -p "$SRC"

echo "=== [vllm] install vllm==0.26.0 ==="
$VLM_PY -m uv pip install --python "$VLM_PY" "vllm==0.26.0" --torch-backend=auto 2>&1 | tail -3
echo "--- vllm sanity ---"
$VLM_PY - <<'EOF'
import vllm, vllm_omni, pathlib
print("vllm", vllm.__version__, "| vllm_omni", getattr(vllm_omni, "__version__", "?"))
root = pathlib.Path(vllm_omni.__file__).parent
print("minimax_h3 present:", (root / "diffusion/models/minimax_h3").exists())
EOF
echo "--- vllm serve flags ---"
$HOME/miniconda3/envs/h3_vllm_NV_py312/bin/vllm serve --help 2>/dev/null | grep -oE -- "--(omni|usp|ring|tensor-parallel-size|text-encoder-tp-size|vae-patch-parallel-size|num-gpus|quantization|diffusion-attention-backend)[^ ]*" | sort -u | head -12
echo "VLLM_ROUND2_DONE"

echo "=== [sglang] source install (main) ==="
if [ ! -d "$SRC/sglang" ]; then
  git clone -q --depth 1 https://github.com/sgl-project/sglang.git "$SRC/sglang"
fi
cd "$SRC/sglang" && git rev-parse HEAD
$SGL_PY -m uv pip install --python "$SGL_PY" -e "python[diffusion]" --prerelease=allow 2>&1 | tail -3
echo "--- sglang sanity ---"
$SGL_PY - <<'EOF'
import sglang
print("sglang", sglang.__version__)
from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3
print("minimax_h3 pipeline config: OK")
EOF
echo "--- sglang serve exists ---"
ls $HOME/miniconda3/envs/h3_sglang_NV_py312/bin/ | grep -E "^sglang" | head -3
echo "SGLANG_ROUND2_DONE"
