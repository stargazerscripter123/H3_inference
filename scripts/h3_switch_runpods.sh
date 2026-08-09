#!/bin/bash
# RunPods 4x5090 switcher for H3 serving (doc/TP4_5090.md route).
#
# Model / precision / topology are three ORTHOGONAL dimensions, because on this
# box they trade off independently:
#   - no GPU P2P at all (nvidia-smi topo -p2p r/w = CNS everywhere), so every
#     NCCL collective is host-staged and ring all-reduce traffic per GPU
#     (~2(N-1)/N x S) dominates -> TP4 is not automatically fastest.
#   - DiT weights per card, FULLY RESIDENT:
#                           BF16 TP4 16.5G | BF16 TP2 33G (exceeds 32G)
#                           FP8  TP4  8.3G | FP8  TP2 16.5G
#     With DLO only `resident` layers stay on card, so BF16 TP2xU2 @ r40 does
#     fit (measured peak 28.4G/card). All four combinations are usable.
#
#   h3_switch_runpods.sh <original|turbo> <bf16|fp8> <tp4|tp2u2> [resident]
#   h3_switch_runpods.sh stop
#
# 运维铁律(与 h3_switch_5090.sh / h3_switch.sh 保持一致,改任何一份都要保持):
#   变体追踪 / 复用同变体 / 端口占用拒启 / flock 串行化且守护进程 9>&- /
#   启动后验活 / 日志轮转不截断。
#   :8091 会被不同 checkpoint 复用,所以 run/vllm.variant 是防"静默服错模型"的关键。
set -u
WHICH="${1:?need original|turbo|stop}"
B=/workspace/h3
VENV="$B/env/bin"
RUN="$B/run"; mkdir -p "$RUN" "$B/logs"
PORT=8091
HEALTH="http://127.0.0.1:$PORT/health"

healthy() { curl -s -m 3 "$HEALTH" >/dev/null 2>&1; }

port_busy() { ss -ltn 2>/dev/null | grep -qE "[^0-9]${PORT}[[:space:]]"; }

rotate_log() { # 保留历史,不截断 —— 否则无法追溯上一次服务的是哪个 ckpt
  local lf="$1"
  [ -s "$lf" ] && mv "$lf" "${lf%.log}.$(date +%Y%m%d_%H%M%S).log"
  ls -1t "${lf%.log}."*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
}

stop_server() {
  local pf="$RUN/vllm.pid"
  if [ -f "$pf" ]; then
    local pg; pg=$(cat "$pf")
    kill -- -"$pg" 2>/dev/null || kill "$pg" 2>/dev/null || true
    sleep 4; kill -9 -- -"$pg" 2>/dev/null || true
    rm -f "$pf"
  fi
  pgrep -f '[v]llm serve' >/dev/null && { pkill -9 -f '[v]llm serve'; sleep 3; } || true
  rm -f "$RUN/vllm.variant" "$RUN/vllm.model"   # 别留下过期的变体声明
  echo 'server stopped'
}

# flock 串行化:两个并发 switch 会互相 kill 到只剩半个服务
exec 9>"$RUN/switch.lock"
flock -w 600 9 || { echo "另一个切换正在进行中(等待 600s 超时);稍后再试" >&2; exit 1; }

[ "$WHICH" = stop ] && { stop_server; exit 0; }

PREC="${2:?need bf16|fp8}"
TOPO="${3:?need tp4|tp2u2}"
RES_IN="${4:-}"

case "$WHICH" in
  original) MODEL="$B/base/MiniMax-H3/FL2VA" ;;
  turbo)    MODEL="$B/merged/MiniMax-H3-Turbo-v4s600ema/FL2VA"
            [ -f "$B/merged/MiniMax-H3-Turbo-v4s600ema/.complete" ] \
              || { echo 'merged checkpoint incomplete'; exit 1; } ;;
  *) echo "unknown model $WHICH"; exit 1 ;;
esac
case "$PREC" in
  bf16) QUANT=(); DEFRES=40 ;;
  fp8)  QUANT=(--quantization fp8); DEFRES=50 ;;
  *) echo "unknown precision $PREC"; exit 1 ;;
esac
case "$TOPO" in
  tp4)   TP=4; USP=1 ;;
  tp2u2) TP=2; USP=2 ;;   # TP pairs land on (0,1)(2,3) = one NUMA node each
  *) echo "unknown topology $TOPO"; exit 1 ;;
esac
RES="${RES_IN:-$DEFRES}"
LABEL="${WHICH}-${PREC}-${TOPO}-r${RES}"

# 复用:同变体且进程活着且健康 —— 冷启动约 210s,不该为一次重复调用白等
HAVE=""; [ -f "$RUN/vllm.variant" ] && HAVE=$(cat "$RUN/vllm.variant")
PID=""; [ -f "$RUN/vllm.pid" ] && PID=$(cat "$RUN/vllm.pid")
if [ "$HAVE" = "$LABEL" ] && [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && healthy; then
  echo "vllm($LABEL) already serving pgid $PID — reuse"
  exit 0
fi
[ -n "$HAVE" ] && [ "$HAVE" != "$LABEL" ] && echo "vllm variant mismatch ($HAVE -> $LABEL), restarting"

stop_server
if port_busy "$PORT"; then
  echo "端口 $PORT 仍被占用(不是本脚本启动的进程?),拒绝启动以免服错 checkpoint" >&2
  exit 1
fi

cd "$B"
LOG="$B/logs/vllm_server.log"
rotate_log "$LOG"
SRC=$(ls -d "$B"/src-*vllm-omni* "$B"/src/*vllm-omni* 2>/dev/null | head -1)
# 补丁状态按代码内容判定 —— 不能用 git diff:补丁一旦 commit,worktree 就是干净的,
# git diff 会反过来报"没打补丁"。不打这个补丁 FP8+DLO 会静默产出纯噪声。
DLO="$SRC/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py"
STRIDE="MISSING_stride_patch"
grep -q "as_strided" "$DLO" 2>/dev/null && STRIDE="APPLIED_as_strided"
{
  echo "### h3_switch_runpods launch $(date -Is)"
  echo "### model=$MODEL  label=$LABEL  resident=$RES  tp=$TP usp=$USP"
  echo "### src=$SRC head=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo n/a)" \
       "stride_patch=$STRIDE"
} >> "$LOG"

setsid nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  VLLM_DISABLED_KERNELS="${VLLM_KERNELS_DISABLE:-CutlassFP8ScaledMMLinearKernel}" \
  "$VENV/vllm" serve "$MODEL" \
    --omni --host 127.0.0.1 --port "$PORT" --trust-remote-code \
    --num-gpus 4 --tensor-parallel-size "$TP" --text-encoder-tp-size "${H3_TE_TP:-4}" \
    --usp "$USP" --ring 1 \
    "${QUANT[@]}" \
    --enable-distributed-layerwise-offload \
    --dlo-no-use-allgather \
    --dlo-resident-layers "$RES" \
    --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
    --diffusion-attention-backend "${VLLM_ATTN:-CUDNN_ATTN}" \
    --enforce-eager \
    >> "$LOG" 2>&1 < /dev/null 9>&- &
echo $! > "$RUN/vllm.pid"

# 启动即死要立刻报错并清掉 pid/variant,否则 run/vllm.variant 会声称一个
# 根本没在服务的 checkpoint,调用方还要白等满健康检查超时。
sleep 2
if ! kill -0 "$(cat "$RUN/vllm.pid")" 2>/dev/null; then
  rm -f "$RUN/vllm.pid"
  echo "vllm 启动后立即退出,见 $LOG" >&2
  tail -5 "$LOG" >&2
  exit 1
fi
printf '%s' "$LABEL" > "$RUN/vllm.variant"
printf '%s' "$MODEL" > "$RUN/vllm.model"
echo "vllm($LABEL) launched pgid $(cat "$RUN/vllm.pid") log=logs/vllm_server.log stride=$STRIDE"
