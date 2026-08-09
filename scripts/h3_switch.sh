#!/bin/bash
# 6000a pipeline switcher — the pipelines share 4 GPUs, only one hot at a time.
#   h3_switch.sh baseline            -> stop sglang+vllm; ComfyUI (:8288) stays resident
#   h3_switch.sh sglang              -> free ComfyUI models, stop vllm, start sglang (:30010, FP8)
#   h3_switch.sh vllm [bf16|fp8]     -> free ComfyUI models, stop sglang, start vllm (:8091)
#   h3_switch.sh vllm-turbo [bf16|fp8] -> vllm serving the Turbo-LoRA-merged checkpoint
#   h3_switch.sh sglang-turbo        -> sglang serving the Turbo-LoRA-merged checkpoint (FP8)
#   h3_switch.sh sglang-lora         -> sglang base + runtime dynamic Turbo LoRA (research/AB)
# Turbo modes refuse to start unless the merged dir carries .complete (merge verified).
set -u
TARGET="${1:?need baseline|sglang|vllm|vllm-turbo|sglang-turbo|sglang-lora}"
VARIANT="${2:-}"   # vllm/vllm-turbo: bf16(default)|fp8
BASE="$HOME/data/dropbox/CV/h3"
RUN="$BASE/run"; mkdir -p "$RUN"
MODEL_ROOT=/home/isaac/Data/h3_weights/MiniMax-H3
TURBO_ROOT=/home/isaac/Data/h3_weights/MiniMax-H3-Turbo-v4s600ema
# sglang resolves the native H3 pipeline by the path BASENAME (registry short-name
# match must equal "minimax-h3"), so it gets an alias dir whose leaf is MiniMax-H3.
TURBO_ROOT_SGL=/home/isaac/Data/h3_weights/turbo_v4s600ema_alias/MiniMax-H3
TURBO_LORA=/home/isaac/Data/h3_weights/loras/minimax_h3_turbo_v4_step600_ema.safetensors
SGL_BIN="$HOME/miniconda3/envs/h3_sglang_NV_py312/bin"
VLM_BIN="$HOME/miniconda3/envs/h3_vllm_NV_py312/bin"

free_comfy() {
  curl -s -m 10 -X POST http://127.0.0.1:8288/free \
    -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}' >/dev/null 2>&1 || true
}

stop_one() { # $1 = name
  local pf="$RUN/$1.pid"
  if [ -f "$pf" ]; then
    local pg; pg=$(cat "$pf")
    kill -- -"$pg" 2>/dev/null || kill "$pg" 2>/dev/null || true
    sleep 3
    kill -9 -- -"$pg" 2>/dev/null || true
    rm -f "$pf"
    echo "stopped $1"
  fi
  rm -f "$RUN/$1.variant"   # 别留下过期的变体声明
}

healthy() { curl -s -m 3 "$1" >/dev/null 2>&1; }

port_busy() { ss -ltn 2>/dev/null | grep -qE "[^0-9]${1}[[:space:]]"; }

require_complete() { # $1 = merged model root
  if [ ! -f "$1/.complete" ]; then
    echo "REFUSED: $1 lacks .complete (merge unfinished or unverified)"; exit 2
  fi
}

start_sglang() { # $1 = model path, $2 = variant label, rest = extra serve args
  local model_path="$1" want="$2"; shift 2
  local have=""; [ -f "$RUN/sglang.variant" ] && have=$(cat "$RUN/sglang.variant")
  if healthy "http://127.0.0.1:30010/health" || healthy "http://127.0.0.1:30010/v1/models"; then
    if [ "$have" = "$want" ]; then echo "sglang already up ($have)"; return 0; fi
    echo "sglang variant mismatch ($have -> $want), restarting"; stop_one sglang
  else
    stop_one sglang   # 端口未响应但进程可能半死: 先清干净,避免撞端口
  fi
  if port_busy 30010; then
    echo "REFUSED: :30010 仍被占用(未被 run/sglang.pid 跟踪的残留进程?); 先查 ss -ltnp | grep 30010" >&2
    exit 1
  fi
  cd "$BASE"
  local CUDAHOME="$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13"
  setsid nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_HOME="$CUDAHOME" PATH="$SGL_BIN:$CUDAHOME/bin:$PATH" \
    LIBRARY_PATH="$CUDAHOME/lib" LD_LIBRARY_PATH="$CUDAHOME/lib:${LD_LIBRARY_PATH:-}" \
    RUNAI_STREAMER_MEMORY_LIMIT=8589934592 \
    NINJA_STATUS="[ninja] " \
    "$SGL_BIN/sglang" serve \
      --model-path "$model_path" \
      --model-variant fl2va \
      --num-gpus 4 \
      --tp-size 4 \
      --ulysses-degree 1 \
      --performance-mode speed \
      --enable-torch-compile false \
      --quantization fp8 \
      --host 127.0.0.1 --port 30010 \
      "$@" \
      > "$BASE/logs/sglang_server.log" 2>&1 < /dev/null 9>&- &
  echo $! > "$RUN/sglang.pid"
  # setsid/nohup 即使 exec 失败也返回 0: 先确认进程活着再落变体声明
  sleep 2
  if ! kill -0 "$(cat "$RUN/sglang.pid" 2>/dev/null)" 2>/dev/null; then
    echo "启动失败: sglang 进程在 2s 内退出。日志末尾:" >&2
    tail -5 "$BASE/logs/sglang_server.log" >&2
    rm -f "$RUN/sglang.pid"
    exit 1
  fi
  printf '%s' "$want" > "$RUN/sglang.variant"
  echo "sglang launched pgid $(cat "$RUN/sglang.pid") variant=$want — cold start takes minutes (weight load)"
}

start_vllm() { # $1 = model root, $2 = variant label (bf16|fp8|turbo-bf16|turbo-fp8)
  local model_root="$1" want="$2"
  local have=""; [ -f "$RUN/vllm.variant" ] && have=$(cat "$RUN/vllm.variant")
  if healthy "http://127.0.0.1:8091/health"; then
    if [ "$have" = "$want" ]; then echo "vllm already up ($have)"; return 0; fi
    echo "vllm variant mismatch ($have -> $want), restarting"; stop_one vllm
  else
    stop_one vllm   # 端口未响应但进程可能半死: 先清干净,避免撞端口
  fi
  if port_busy 8091; then
    echo "REFUSED: :8091 仍被占用(未被 run/vllm.pid 跟踪的残留进程?); 先查 ss -ltnp | grep 8091" >&2
    exit 1
  fi
  local QUANT=""
  case "$want" in *fp8) QUANT="--quantization fp8" ;; esac
  cd "$BASE"
  setsid nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    "$VLM_BIN/vllm" serve "$model_root/FL2VA" \
      --omni \
      --host 127.0.0.1 --port 8091 \
      --trust-remote-code \
      --num-gpus 4 \
      --tensor-parallel-size 4 \
      --usp 1 --ring 1 \
      --text-encoder-tp-size 4 \
      --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
      --diffusion-attention-backend FLASH_ATTN \
      $QUANT \
      > "$BASE/logs/vllm_server.log" 2>&1 < /dev/null 9>&- &
  echo $! > "$RUN/vllm.pid"
  # setsid/nohup 即使 exec 失败也返回 0: 先确认进程活着再落变体声明
  sleep 2
  if ! kill -0 "$(cat "$RUN/vllm.pid" 2>/dev/null)" 2>/dev/null; then
    echo "启动失败: vllm 进程在 2s 内退出。日志末尾:" >&2
    tail -5 "$BASE/logs/vllm_server.log" >&2
    rm -f "$RUN/vllm.pid"
    exit 1
  fi
  printf '%s' "$want" > "$RUN/vllm.variant"
  echo "vllm launched pgid $(cat "$RUN/vllm.pid") variant=$want — cold start takes minutes (weight load)"
}

# 两个 Claude session 共用这台机器,并发切换会互相踩(A 删了 pid 文件、
# B 的 stop_one 变空操作 -> 两个服务抢同一端口)。全局串行化。
exec 9>"$RUN/.switch.lock"
flock -w 600 9 || { echo "另一个切换正在进行中(等待 600s 超时);稍后再试" >&2; exit 1; }

case "$TARGET" in
  baseline)
    stop_one sglang; stop_one vllm
    echo "baseline active (ComfyUI :8288 resident)" ;;
  sglang)
    free_comfy; stop_one vllm; start_sglang "MiniMaxAI/MiniMax-H3" "base" ;;
  sglang-turbo)
    require_complete "$TURBO_ROOT"
    mkdir -p "$(dirname "$TURBO_ROOT_SGL")"
    [ -e "$TURBO_ROOT_SGL" ] || ln -s "$TURBO_ROOT" "$TURBO_ROOT_SGL"
    free_comfy; stop_one vllm; start_sglang "$TURBO_ROOT_SGL" "turbo" ;;
  sglang-lora)
    free_comfy; stop_one vllm
    start_sglang "MiniMaxAI/MiniMax-H3" "lora" \
      --lora-path "$TURBO_LORA" --lora-nickname turbo \
      --lora-scale 1.0 --lora-merge-mode dynamic ;;
  vllm)
    free_comfy; stop_one sglang; start_vllm "$MODEL_ROOT" "${VARIANT:-bf16}" ;;
  vllm-turbo)
    require_complete "$TURBO_ROOT"
    free_comfy; stop_one sglang; start_vllm "$TURBO_ROOT" "turbo-${VARIANT:-bf16}" ;;
  *) echo "unknown target $TARGET"; exit 1 ;;
esac
