#!/bin/bash
# 5090 pipeline switcher. Serving backends need BOTH GPUs and most of host RAM,
# so switching to vllm/sglang KILLS the two ComfyUI workers (frees ~90GB RAM);
# switching back to comfy restarts them (cold reload ~1min).
#   h3_switch_5090.sh comfy      -> stop serving backends; start both ComfyUI workers
#   h3_switch_5090.sh vllm       -> stop comfy workers + sglang; start vLLM base   (:8091, TP2 FP8+DLO)
#   h3_switch_5090.sh vllm-turbo -> stop comfy workers + sglang; start vLLM merged Turbo-LoRA (:8091)
#   h3_switch_5090.sh sglang     -> stop comfy workers + vllm; start SGLang (:30010, TP2 FP8)
#
# vllm 与 vllm-turbo 共用 :8091 但服务不同的 checkpoint,因此用 run/vllm.variant
# 记录当前变体: 互切时必须重启,不能因为"端口是健康的"就复用(否则会静默地
# 继续用上一个 checkpoint 出片,而延迟几乎相同、单帧也难分辨)。
# 所服务的权重路径同时写入 run/vllm.model 与新日志头,便于事后追溯。
set -u
if [ "${H3_SWITCH_LIB:-0}" = 1 ]; then
  TARGET="__lib__"          # 被 test 脚本 source 时只加载函数,不执行分派
else
  TARGET="${1:?need comfy|comfy-tlora|vllm|vllm-turbo|sglang}"
fi
BASE="$HOME/data/dropbox/CV/h3"
RUN="$BASE/run"; mkdir -p "$RUN"
MODEL_ROOT=$HOME/data/dropbox/CV/h3/models_official/MiniMax-H3
SGL_BIN="$HOME/miniconda3/envs/h3_sglang_NV_py312/bin"
VLM_BIN="$HOME/miniconda3/envs/h3_vllm_NV_py312/bin"

healthy() { curl -s -m 3 "$1" >/dev/null 2>&1; }

port_busy() { # $1 = port
  ss -ltn 2>/dev/null | grep -qE "[^0-9]${1}[[:space:]]"
}

rotate_log() { # $1 = logfile — 保留历史,不再每次启动截断(否则无法追溯是哪个 ckpt)
  local lf="$1"
  [ -s "$lf" ] && mv "$lf" "${lf%.log}.$(date +%Y%m%d_%H%M%S).log"
  ls -1t "${lf%.log}."*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
}

stop_comfy_workers() { # $1 = 端口 ERE(默认 8188|8189|8190 全停);方括号防自匹配
  local PORTS="${1:-81(88|89|90)}"
  local PAT="[m]ain.py --listen 127.0.0.1 --port $PORTS"
  for p in $(pgrep -f "$PAT"); do kill "$p" 2>/dev/null; done
  sleep 3
  for p in $(pgrep -f "$PAT"); do kill -9 "$p" 2>/dev/null; done
  echo "comfy workers stopped ($PORTS)"
}

stop_one() {
  local pf="$RUN/$1.pid"
  if [ -f "$pf" ]; then
    local pg; pg=$(cat "$pf")
    kill -- -"$pg" 2>/dev/null || kill "$pg" 2>/dev/null || true
    sleep 3
    kill -9 -- -"$pg" 2>/dev/null || true
    rm -f "$pf"
    echo "stopped $1"
  fi
  rm -f "$RUN/$1.variant" "$RUN/$1.model"   # 别留下过期的变体声明
}

_launch_vllm() { # $1 = model root, $2 = label, $3 = resident layers (test 时被 stub)
  local MROOT="$1" LABEL="$2" RES="$3"
  local LOG="$BASE/logs/vllm_server.log"
  rotate_log "$LOG"
  local SRC="$BASE/src/vllm-omni-pr5910"
  # ★ 判据必须**限定在 PinnedResidentLayerGroup 类内**。
  # 裸的 `grep as_strided <整个文件>` 恒真、毫无鉴别力:未打补丁的原文件在
  # _shard_and_pin(:299) 与 prefetch_layer(:412) 本来就有两处 as_strided,
  # 补丁加的那处在 PinnedResidentLayerGroup.load()(:630)。
  # 2026-08-09 实测反向应用补丁做对照:裸 grep 对"打了/没打"都报 APPLIED。
  # 也不能用 git diff —— 补丁已提交时 worktree 干净,会反过来误判成"没打"。
  # 不打这个补丁,FP8+DLO 不报错、**静默产出纯噪声**。
  local DLO="$SRC/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py"
  local STRIDE="MISSING_stride_patch"
  sed -n '/class PinnedResidentLayerGroup/,/^class /p' "$DLO" 2>/dev/null \
    | grep -q as_strided && STRIDE="APPLIED_as_strided"
  {
    echo "### h3_switch_5090 launch $(date -Is)"
    echo "### model=$MROOT  label=$LABEL  resident=$RES"
    echo "### src=$SRC head=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo n/a)" \
         "branch=$(git -C "$SRC" branch --show-current 2>/dev/null || echo detached)" \
         "stride_patch=$STRIDE"
  } > "$LOG"
  cd "$BASE"
  setsid nohup env CUDA_VISIBLE_DEVICES=0,1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
    VLLM_DISABLED_KERNELS="${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}" \
    VLLM_TEST_FORCE_FP8_MARLIN="${VLLM_FORCE_MARLIN:-0}" \
    VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}" \
    "$VLM_BIN/vllm" serve "$MROOT" \
      --omni --host 127.0.0.1 --port 8091 --trust-remote-code \
      --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
      --usp 1 --ring 1 \
      --quantization fp8 \
      --enable-distributed-layerwise-offload \
      --dlo-no-use-allgather \
      --dlo-resident-layers "$RES" \
      --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
      --diffusion-attention-backend "${VLLM_ATTN:-CUDNN_ATTN}" \
      --enforce-eager \
      >> "$LOG" 2>&1 < /dev/null 9>&- &
  echo $! > "$RUN/vllm.pid"
}

start_vllm() { # $1 = model root (default: official base), $2 = variant label
  # PR#5910 + stride patch: global online FP8 (DiT+TE) + rank-local DLO.
  local MROOT="${1:-$MODEL_ROOT/FL2VA}"
  local LABEL="${2:-base}"
  local RES="${VLLM_DLO_RESIDENT:-50}"
  local have=""; [ -f "$RUN/vllm.variant" ] && have=$(cat "$RUN/vllm.variant")

  if healthy "http://127.0.0.1:8091/health"; then
    if [ "$have" = "$LABEL" ]; then
      echo "vllm already up ($have, model=$(cat "$RUN/vllm.model" 2>/dev/null || echo unknown))"
      return 0
    fi
    echo "vllm variant mismatch ($have -> $LABEL), restarting"
    stop_one vllm
  else
    stop_one vllm      # 端口未响应但进程可能半死: 先清干净,避免撞端口
  fi

  if port_busy 8091; then
    echo "REFUSED: :8091 仍被占用(未被 run/vllm.pid 跟踪的残留进程?)。" \
         "先查 'ss -ltnp | grep 8091' 再手动清理,以免起第二个服务撞端口。" >&2
    exit 1
  fi

  _launch_vllm "$MROOT" "$LABEL" "$RES"
  # setsid/nohup 即使 exec 失败也返回 0,所以要确认进程真的活着再落变体声明,
  # 否则 run/vllm.variant 会声称一个根本没在服务的 checkpoint。
  sleep 2
  if ! kill -0 "$(cat "$RUN/vllm.pid" 2>/dev/null)" 2>/dev/null; then
    echo "启动失败: 进程在 2s 内退出。日志末尾:" >&2
    tail -5 "$BASE/logs/vllm_server.log" >&2
    rm -f "$RUN/vllm.pid"
    exit 1
  fi
  printf '%s' "$LABEL" > "$RUN/vllm.variant"
  printf '%s' "$MROOT" > "$RUN/vllm.model"
  echo "vllm(pr5910 fp8+dlo, model=$LABEL, resident=$RES) launched pgid $(cat "$RUN/vllm.pid")"
  echo "  serving: $MROOT"
}

_launch_sglang() { # test 时被 stub
  local LOG="$BASE/logs/sglang_server.log"
  rotate_log "$LOG"
  cd "$BASE"
  local CUDAHOME="$HOME/miniconda3/envs/h3_sglang_NV_py312/lib/python3.12/site-packages/nvidia/cu13"
  setsid nohup env CUDA_VISIBLE_DEVICES=0,1 \
    CUDA_HOME="$CUDAHOME" PATH="$SGL_BIN:$CUDAHOME/bin:$PATH" \
    LIBRARY_PATH="$CUDAHOME/lib" LD_LIBRARY_PATH="$CUDAHOME/lib:${LD_LIBRARY_PATH:-}" \
    RUNAI_STREAMER_MEMORY_LIMIT=8589934592 \
    "$SGL_BIN/sglang" serve \
      --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
      --text-encoder-path "$HOME/data/dropbox/CV/h3/models_te_fp8/Qwen3-VL-32B-Instruct-FP8" \
      --num-gpus 2 --tp-size 2 --ulysses-degree 1 \
      --encoder-parallel fold \
      --quantization fp8 \
      --performance-mode speed \
      --layerwise-offload-components text_encoder,vae \
      --enable-torch-compile false \
      --host 127.0.0.1 --port 30010 \
      >> "$LOG" 2>&1 < /dev/null 9>&- &
  echo $! > "$RUN/sglang.pid"
}

start_sglang() {
  if healthy "http://127.0.0.1:30010/health"; then echo "sglang already up"; return 0; fi
  stop_one sglang        # 同上: 半死进程先清掉
  if port_busy 30010; then
    echo "REFUSED: :30010 仍被占用(残留进程?)。先查 'ss -ltnp | grep 30010'。" >&2
    exit 1
  fi
  _launch_sglang
  printf '%s' "fp8-te" > "$RUN/sglang.variant"
  echo "sglang launched pgid $(cat "$RUN/sglang.pid") — cold start takes minutes"
}

[ "$TARGET" = "__lib__" ] && return 0 2>/dev/null

# 两个 Claude session 共用这台机器,并发切换会互相踩(A 删了 pid 文件、B 的 stop_one
# 变空操作 -> 两个服务抢同一端口)。全局串行化,等待上限 10 分钟(冷启动量级)。
exec 9>"$RUN/.switch.lock"
flock -w 600 9 || { echo "另一个切换正在进行中(等待 600s 超时);稍后再试" >&2; exit 1; }

case "$TARGET" in
  comfy)
    stop_one sglang; stop_one vllm
    stop_comfy_workers "8190"          # tlora worker 与这两个抢 host RAM
    bash "$BASE/scripts/launch_comfy.sh" 1 8188 9>&- || true
    bash "$BASE/scripts/launch_comfy_5090_turbo.sh" 9>&- || true
    echo "comfy workers active (:8188 original, :8189 teacache)" ;;
  comfy-tlora)
    # profile comfy-int8-turbo-1c: 单个 GPU0:8190 worker 跑 Turbo LoRA。
    # host RAM 125G / 每 worker ~45G,三个装不下 -> 独占启动。
    stop_one sglang; stop_one vllm
    stop_comfy_workers "818(8|9)"
    bash "$BASE/scripts/launch_comfy_5090_tlora.sh" 9>&- ;;
  vllm)
    stop_comfy_workers; stop_one sglang; start_vllm "$MODEL_ROOT/FL2VA" base ;;
  vllm-turbo)
    # Turbo-LoRA merged checkpoint (07 topic), same TP2 FP8+DLO config.
    MERGED="$HOME/data/dropbox/CV/h3/models_merged/MiniMax-H3-Turbo-v4s600ema"
    [ -f "$MERGED/FL2VA/.complete" ] || [ -f "$MERGED/.complete" ] || { echo "merged checkpoint incomplete"; exit 1; }
    stop_comfy_workers; stop_one sglang; start_vllm "$MERGED/FL2VA" turbo-lora ;;
  sglang)
    stop_comfy_workers; stop_one vllm; start_sglang ;;
  *) echo "unknown target $TARGET"; exit 1 ;;
esac
