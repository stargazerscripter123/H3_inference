#!/bin/bash
# MiniMax H3 deployment bootstrap — conda env + ComfyUI pinned checkout
set -euo pipefail
export PATH="$HOME/miniconda3/bin:$PATH"

BASE="$HOME/data/dropbox/CV/h3"
PIN=57500fc5bc92566a63f2046824f522cd55c335ca
ENVNAME=h3_comfy_NV_py312
ENVPY="$HOME/miniconda3/envs/$ENVNAME/bin/python"

echo "=== [1/5] dirs ==="
mkdir -p "$BASE"/{inputs,outputs,workflows,scripts,logs}

echo "=== [2/5] conda env ==="
if [ ! -x "$ENVPY" ]; then
  conda create -y -n "$ENVNAME" python=3.12
fi
"$ENVPY" --version

echo "=== [3/5] torch cu130 ==="
"$ENVPY" -m pip install --no-input torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "=== [4/5] ComfyUI @ $PIN ==="
if [ ! -d "$BASE/ComfyUI/.git" ]; then
  mkdir -p "$BASE/ComfyUI"
  cd "$BASE/ComfyUI"
  git init -q
  git remote add origin https://github.com/comfyanonymous/ComfyUI.git 2>/dev/null || true
  git fetch --depth 1 origin "$PIN"
  git checkout -q FETCH_HEAD
else
  cd "$BASE/ComfyUI"
  git fetch --depth 1 origin "$PIN" && git checkout -q FETCH_HEAD
fi
git log --oneline -1

echo "=== [5/5] python deps ==="
"$ENVPY" -m pip install --no-input -r "$BASE/ComfyUI/requirements.txt"
"$ENVPY" -m pip install --no-input "huggingface_hub[cli]" hf_transfer

echo "=== sanity ==="
"$ENVPY" - <<'EOF'
import torch, importlib
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "devcount", torch.cuda.device_count())
for m in ("comfy_kitchen", "comfy_aimdo", "av", "huggingface_hub"):
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "FAIL", e)
EOF
echo "SETUP_DONE $(hostname)"
