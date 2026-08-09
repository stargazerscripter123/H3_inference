#!/bin/bash
# =============================================================================
# h3.sh — 本部署的主入口。在这台机器上直接跑,不需要 ssh、不需要另一台工作机。
#
#   scripts/h3.sh --profile <profile> --first <图> --last <图>
#                 (--prompt-file <文件> | --prompt "文本")
#                 [--nfe N] [--seed N] [--seconds 5] [--name 标签] [--list]
#
# 三个 profile(864x480 / 124 帧 / 带原生音频 / warm 延迟):
#
#   comfy-int8-original-1c   30 步基线,零 tweak(默认 SDPA)      GPU1:8188  ~110s
#   comfy-int8-teacache-1c   12 步 + TeaCache(有损跳步缓存)      GPU0:8189  ~35s
#   comfy-int8-turbo-1c      4~6 步 + Turbo LoRA(少步蒸馏)       GPU0:8190  ~25s @NFE4
#
#   三者共用同一套权重(int8 DiT + nvfp4 TE + video/audio VAE),只有采样路径不同。
#   original 与 teacache 可以同时在跑(两张卡);turbo 独占 —— 见 comfyctl.sh 里
#   关于 RAM 的说明。
#
# 步数一律用 --nfe(实际 DiT forward 次数)
#   ComfyUI 侧 steps == forwards,所以 nfe 直接就是 steps。
#   **这个等式只在 ComfyUI 成立**:vLLM-Omni / SGLang 的 num_inference_steps=N
#   只跑 N-1 次 forward,那边要 +1。把这里的数字直接拿去和引擎侧比会差一步。
#
# 报时分两段
#   启动(切 worker + 等就绪)与推理分开打印。**新起的 worker,第一次推理是冷的**:
#   ComfyUI 的 /system_stats 在模型加载之前就响应(6s 就"就绪"),35G 权重是首次提交
#   时才懒加载的。实测 teacache 冷 40.0s / warm 35.0s。冷的那遍不能当性能数字,
#   本脚本会在输出末尾提示。要量速度就换 --seed 再跑一遍。
#
# 量速度必须换 seed
#   ComfyUI 会缓存整张图。参数完全相同的第二次提交直接返回缓存,你测到的是缓存
#   命中而不是推理。本脚本默认 seed=0;连续测同一 profile 时手工 --seed 递增。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../install/pins.env
source "$HERE/../install/pins.env"

PROFILES="comfy-int8-original-1c comfy-int8-teacache-1c comfy-int8-turbo-1c"

# profile -> "gpu|port|nfe默认|客户端额外参数"
# 画布三个 profile 统一 864x480。(上游主仓库里 original 因为漏写画布键落到了
# 832x480,导致三个 profile 的数字其实不可直接比;这里修掉。)
profile_def() {
  case "$1" in
    comfy-int8-original-1c) echo "1|8188|30|" ;;
    comfy-int8-teacache-1c) echo "0|8189|12|--teacache-thresh 0.10" ;;
    comfy-int8-turbo-1c)    echo "0|8190|4|--turbo-lora minimax_h3_turbo_v4_step600_ema.safetensors" ;;
    *) return 1 ;;
  esac
}

# H3 的 TE 用 nvfp4 那份(15G);int8 那份 26G 是 serving 线用的,这台机器上没必要。
TE="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
CANVAS_W=864
CANVAS_H=480

usage() { sed -n '2,40p' "$0"; }

PROFILE=""; FIRST=""; LAST=""; PROMPT=""; PROMPT_FILE=""
NFE=""; SEED=0; SECONDS_IN=5; NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift ;;
    --first) FIRST="${2:?}"; shift ;;
    --last) LAST="${2:?}"; shift ;;
    --prompt) PROMPT="${2:?}"; shift ;;
    --prompt-file) PROMPT_FILE="${2:?}"; shift ;;
    --nfe) NFE="${2:?}"; shift ;;
    --seed) SEED="${2:?}"; shift ;;
    --seconds) SECONDS_IN="${2:?}"; shift ;;
    --name) NAME="${2:?}"; shift ;;
    --list)
      printf '%-24s %-6s %-6s %-5s %s\n' PROFILE GPU PORT NFE 说明
      printf '%-24s %-6s %-6s %-5s %s\n' comfy-int8-original-1c 1 8188 30 "零 tweak 基线 ~110s"
      printf '%-24s %-6s %-6s %-5s %s\n' comfy-int8-teacache-1c 0 8189 12 "TeaCache 有损跳步 ~35s"
      printf '%-24s %-6s %-6s %-5s %s\n' comfy-int8-turbo-1c 0 8190 4 "Turbo LoRA 少步 ~25s"
      echo; echo "画布统一 864x480;帧数由 --seconds 推,必须满足 n%17==5"
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ -n "$PROFILE" ] || { echo "需要 --profile(可选: $PROFILES,或 --list)" >&2; exit 2; }
DEF=$(profile_def "$PROFILE") || { echo "未知 profile: $PROFILE" >&2; exit 2; }
IFS='|' read -r GPU PORT DEF_NFE EXTRA <<< "$DEF"
[ -n "$FIRST" ] && [ -n "$LAST" ] || { echo "需要 --first 与 --last" >&2; exit 2; }
[ -n "$PROMPT" ] || [ -n "$PROMPT_FILE" ] || { echo "需要 --prompt 或 --prompt-file" >&2; exit 2; }
[ -f "$FIRST" ] || { echo "找不到首帧: $FIRST" >&2; exit 2; }
[ -f "$LAST" ]  || { echo "找不到尾帧: $LAST"  >&2; exit 2; }
NFE="${NFE:-$DEF_NFE}"
NAME="${NAME:-h3_$(date +%m%d_%H%M%S)}"
[ -x "$H3_PYTHON" ] || { echo "找不到 $H3_PYTHON —— 先跑 install/bootstrap.sh" >&2; exit 1; }

# 帧数:24fps,且**必须满足 n % 17 == 5**(模型的 latent 时间打包决定的,不是随便凑的)。
# 训练范围约 124-362 帧;低于 124 会明显掉质量,高于 362 未验证。
FRAMES=$(python3 - "$SECONDS_IN" <<'PY'
import sys
n = round(float(sys.argv[1]) * 24)
n = n + ((5 - n % 17) % 17)
if n < 124:
    print(f"[warn] {sys.argv[1]}s 低于训练下限,提升到 124 帧 (~5.17s)", file=sys.stderr)
    n = 124
if n > 362:
    print(f"[warn] {n} 帧超出训练范围上限 362 (~15s),质量未验证", file=sys.stderr)
print(n)
PY
)

echo "任务 $NAME | profile $PROFILE | 画布 ${CANVAS_W}x${CANVAS_H} | ${FRAMES} 帧 | nfe $NFE | seed $SEED"

# --- 1. 起 worker(含互斥) ----------------------------------------------------
# 无条件跑 ensure:互斥不能有前提条件,否则 turbo 会和 8188/8189 抢 RAM 直接 OOM。
T0=$(date +%s)
ENS=$(bash "$HERE/comfyctl.sh" ensure "$PROFILE" 2>&1) || { echo "$ENS" >&2; exit 1; }
T1=$(date +%s)
WORKER_STATE=$(grep -o 'COMFYCTL=[a-z]*' <<< "$ENS" | tail -1 | cut -d= -f2)
grep -v '^COMFYCTL=' <<< "$ENS"   # COMFYCTL= 是给本脚本读的机器标记,不给人看

# --- 2. 画布统一 --------------------------------------------------------------
F1=$(bash "$HERE/prep_frames.sh" "$FIRST" "${NAME}_first.png" "$CANVAS_W" "$CANVAS_H")
F2=$(bash "$HERE/prep_frames.sh" "$LAST"  "${NAME}_last.png"  "$CANVAS_W" "$CANVAS_H")

PF_ARGS=()
if [ -n "$PROMPT_FILE" ]; then
  [ -f "$PROMPT_FILE" ] || { echo "找不到 prompt 文件: $PROMPT_FILE" >&2; exit 2; }
  PF_ARGS=(--prompt-file "$PROMPT_FILE")
else
  PF_ARGS=(--prompt "$PROMPT")
fi

# --- 3. 提交 ------------------------------------------------------------------
OUTDIR="$H3_BASE/outputs/$NAME"
mkdir -p "$OUTDIR"
# shellcheck disable=SC2086
OUT=$("$H3_PYTHON" "$HERE/run_fl2va.py" \
  --server "http://127.0.0.1:$PORT" \
  --te "$TE" \
  --first "$F1" --last "$F2" \
  "${PF_ARGS[@]}" \
  --width "$CANVAS_W" --height "$CANVAS_H" --length "$FRAMES" \
  --nfe "$NFE" --seed "$SEED" \
  --gpu-index "$GPU" \
  --prefix "$NAME/${NAME}_${PROFILE}" \
  --manifest-dir "$OUTDIR" \
  --profile "$PROFILE" \
  $EXTRA 2>&1 | tee /dev/stderr)
# 赋值语句的退出码 = 命令替换的退出码;脚本开头 set -o pipefail,所以 tee 成功
# 不会掩盖 run_fl2va.py 的失败。(别改成 ${PIPESTATUS[0]} —— 那取的是赋值本身。)
RC=$?
T2=$(date +%s)

[ "$RC" = 0 ] || { echo "生成失败(rc=$RC)" >&2; exit "$RC"; }

# run_fl2va.py 的 DONE 行是唯一的数据通道。
INFER=$(echo "$OUT" | grep -o 'wall=[0-9.]*' | tail -1 | cut -d= -f2)
echo
echo "===== 结果 ====="
printf '产物目录: %s\n' "$OUTDIR"
find "$OUTDIR" -name '*.mp4' -newermt "@$T0" 2>/dev/null | sed 's/^/  /'
printf '⏱  启动 %ss(切 worker + 等就绪) + 推理 %ss = 共 %ss\n' \
  "$((T1-T0))" "${INFER:-$((T2-T1))}" "$((T2-T0))"
# 冷启那一遍不能当性能数字用。判据是"worker 是不是这次新起的",不是"启动花了多久":
# /system_stats 在模型加载前就响应,新 worker 6s 就"就绪",但首次推理仍含 35G 权重的
# 加载(teacache 实测 40s vs warm 35s)。
[ "$WORKER_STATE" = launched ] && \
  echo "   ⚠ worker 是这次新起的,这遍推理含模型加载,不能当性能数字;换 --seed 再跑一遍取 warm"
