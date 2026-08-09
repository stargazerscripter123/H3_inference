#!/bin/bash
# 5090 ComfyUI worker for the Turbo-LoRA pipeline (profile comfy-int8-turbo-1c).
# GPU0, port 8190, 独立日志。与 :8188/:8189 同为单卡 worker,但 host RAM 只有 125G、
# 每个 worker 常驻 ~45G,所以三个不能共存 —— 由 h3_switch_5090.sh comfy-tlora 保证互斥。
set -u
BASE="$HOME/data/dropbox/CV/h3"
ENVPY="$HOME/miniconda3/envs/h3_comfy_NV_py312/bin/python"
PORT=8190

if curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  echo "comfy tlora worker already up on :$PORT"; exit 0
fi

cd "$BASE/ComfyUI"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 setsid nohup "$ENVPY" main.py \
  --listen 127.0.0.1 --port "$PORT" \
  --preview-method none \
  --async-offload 2 \
  --reserve-vram 1.5 \
  --output-directory "$BASE/outputs" \
  --input-directory "$BASE/inputs" \
  > "$BASE/logs/comfyui_tlora.log" 2>&1 < /dev/null &

for i in $(seq 1 36); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
    echo "comfy tlora worker up on :$PORT (GPU0)"; exit 0
  fi
  kill -0 "$!" 2>/dev/null || { echo "worker died, log tail:"; tail -30 "$BASE/logs/comfyui_tlora.log"; exit 1; }
  sleep 5
done
echo "timeout waiting for :$PORT"; tail -30 "$BASE/logs/comfyui_tlora.log"; exit 1
