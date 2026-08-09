#!/bin/bash
# =============================================================================
# comfyctl.sh — ComfyUI worker 的启停与互斥。
#
#   scripts/comfyctl.sh ensure <profile>   起该 profile 的 worker(先停掉冲突的)
#   scripts/comfyctl.sh stop [profile]     停;不给 profile 则全停
#   scripts/comfyctl.sh status             看谁在跑
#
# 为什么需要"互斥"这件事
#   这台机器 RAM 125 GiB,**单个 ComfyUI worker 常驻 RSS 实测 46.4 GiB**。
#   三个 worker 共存必然被 OOM killer 收走。所以:
#     original(GPU1:8188) 与 teacache(GPU0:8189) 可以共存 —— 两张卡分开,RAM 约 93 GiB
#     turbo(GPU0:8190)    独占 —— 起它必须先杀掉 8188/8189
#   互斥不是优化,是这条产线能不能跑起来的前提。ensure 永远无条件执行,
#   不要给它加"如果已经在跑就跳过整段"之类的前提条件。
#
# 三个 profile 的 worker 参数(与延迟数字绑定,改了要重测):
#   comfy-int8-original-1c  GPU1 :8188  无额外 flag(默认 SDPA attention)
#   comfy-int8-teacache-1c  GPU0 :8189  --preview-method none --async-offload 2 --reserve-vram 1.5
#   comfy-int8-turbo-1c     GPU0 :8190  同上
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../install/pins.env
source "$HERE/../install/pins.env"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '   \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# profile -> "gpu port 额外flag"
profile_spec() {
  case "$1" in
    comfy-int8-original-1c) echo "1 8188 " ;;
    comfy-int8-teacache-1c) echo "0 8189 --preview-method none --async-offload 2 --reserve-vram 1.5" ;;
    comfy-int8-turbo-1c)    echo "0 8190 --preview-method none --async-offload 2 --reserve-vram 1.5" ;;
    *) return 1 ;;
  esac
}
ALL_PROFILES="comfy-int8-original-1c comfy-int8-teacache-1c comfy-int8-turbo-1c"

healthy() { curl -s -m 3 "http://127.0.0.1:$1/system_stats" >/dev/null 2>&1; }

# ★ 这个 pgrep 模式依赖 worker 命令行里 `main.py --listen 127.0.0.1 --port <P>`
#   这三个参数**紧挨着且按这个顺序**出现(见下面 launch_worker)。
#   改了 launch_worker 的 flag 顺序,这里会静默匹配不到 —— 于是杀不掉旧 worker,
#   于是三个 worker 共存,于是 OOM。改一处必须同步另一处。
# ★ 方括号 [m]ain.py 是为了让 pgrep 匹配不到 pgrep 自己那条命令行。
kill_ports() { # $1 = 端口的 ERE,如 "8190" 或 "818(8|9)"
  local pat="[m]ain.py --listen 127.0.0.1 --port $1" pids
  pids=$(pgrep -f "$pat" || true)
  [ -z "$pids" ] && return 0
  warn "停 worker(端口 $1): $(echo "$pids" | tr '\n' ' ')"
  for p in $pids; do kill "$p" 2>/dev/null; done
  sleep 3
  for p in $(pgrep -f "$pat" || true); do kill -9 "$p" 2>/dev/null; done
  sleep 1
}

launch_worker() { # $1=gpu $2=port $3...=额外 flag
  local gpu="$1" port="$2"; shift 2
  local log="$H3_BASE/logs/comfyui_$port.log"
  mkdir -p "$H3_BASE/logs"
  # 日志轮转而不是截断:出问题时要能回看上一次起的是什么。
  [ -s "$log" ] && mv "$log" "${log%.log}.$(date +%Y%m%d_%H%M%S).log"
  ls -1t "${log%.log}."*.log 2>/dev/null | tail -n +11 | xargs -r rm -f

  {
    echo "### comfyctl launch $(date -Is)"
    echo "### gpu=$gpu port=$port flags=$*"
    echo "### comfyui=$(git -C "$H3_BASE/ComfyUI" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  } >> "$log"

  cd "$H3_BASE/ComfyUI" || die "找不到 $H3_BASE/ComfyUI(先跑 install/bootstrap.sh)"
  # ★ CUDA_DEVICE_ORDER=PCI_BUS_ID 统一设:不设的话默认是 FASTEST_FIRST,
  #   "GPU0" 指的是哪张物理卡就不保证了。同型号双卡通常等价,但不保证。
  #   (上游主仓库的三个 launcher 在这一点上不一致,这里统一掉。)
  # ★ 9>&- 把 flock 的 fd 从子进程关掉。不加的话 setsid 出去的 worker 会继承 fd 9,
  #   把切换锁一直持有到 worker 生命周期结束,后面每次 ensure 都要等满 600s 超时。
  #   这个坑真踩过。
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
    setsid nohup "$H3_PYTHON" main.py \
      --listen 127.0.0.1 --port "$port" \
      "$@" \
      --output-directory "$H3_BASE/outputs" \
      --input-directory "$H3_BASE/inputs" \
      >> "$log" 2>&1 < /dev/null 9>&- &
  local pid=$!
  sleep 2
  kill -0 "$pid" 2>/dev/null || { echo; tail -30 "$log"; die "worker 起来就死了,日志见 $log"; }

  # 冷启动要把 20G DiT + 15G TE + VAE 读进来,首次约 45-60s。
  for i in $(seq 1 60); do
    healthy "$port" && { ok "worker :$port 就绪(GPU$gpu, $((i*3))s)"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo; tail -30 "$log"; die "worker 中途死了,日志见 $log"; }
    sleep 3
  done
  tail -30 "$log"; die "worker :$port 180s 没就绪"
}

cmd_ensure() {
  local prof="${1:?ensure 需要 profile 名}"
  local spec; spec=$(profile_spec "$prof") || die "未知 profile: $prof(可选: $ALL_PROFILES)"
  local gpu port flags
  gpu=$(echo "$spec" | cut -d' ' -f1)
  port=$(echo "$spec" | cut -d' ' -f2)
  flags=$(echo "$spec" | cut -d' ' -f3-)

  mkdir -p "$H3_BASE/run"
  # flock 串行化:两个并发 ensure 会互相 kill 到只剩半个。
  exec 9>"$H3_BASE/run/.comfyctl.lock"
  flock -w 600 9 || die "另一个 ensure 正在进行(等 600s 超时)"

  case "$prof" in
    comfy-int8-turbo-1c)
      # turbo 独占:RAM 装不下它和另外两个。
      kill_ports "818(8|9)" ;;
    *)
      # original 与 teacache 共存;但 turbo 必须让位。
      kill_ports "8190" ;;
  esac

  if healthy "$port"; then
    ok "worker :$port 已在跑,复用"
    # 最后一行是给 h3.sh 读的:worker 是复用的还是新起的。
    # 这件事不能靠"启动花了多久"来猜 —— ComfyUI 的 /system_stats 在模型加载**之前**
    # 就会响应(实测 6s 就绪),20G DiT + 15G TE 是首次提交时才懒加载的。
    # 所以新起的 worker 即使"就绪"很快,第一次推理仍然是冷的(实测 40s vs warm 35s)。
    echo "COMFYCTL=reused"
  else
    # shellcheck disable=SC2086
    launch_worker "$gpu" "$port" $flags
    echo "COMFYCTL=launched"
  fi
}

cmd_stop() {
  if [ $# -ge 1 ]; then
    local spec; spec=$(profile_spec "$1") || die "未知 profile: $1"
    kill_ports "$(echo "$spec" | cut -d' ' -f2)"
  else
    kill_ports "81(88|89|90)"
  fi
  ok "已停"
}

cmd_status() {
  local n=0
  for prof in $ALL_PROFILES; do
    local spec port; spec=$(profile_spec "$prof"); port=$(echo "$spec" | cut -d' ' -f2)
    if healthy "$port"; then
      printf '   \033[32m●\033[0m %-24s :%s  在跑\n' "$prof" "$port"; n=$((n+1))
    else
      printf '     %-24s :%s\n' "$prof" "$port"
    fi
  done
  [ "$n" -le 2 ] || warn "$n 个 worker 共存 —— RAM 125 GiB 装不下,随时可能 OOM"
}

case "${1:-}" in
  ensure) shift; cmd_ensure "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  status) cmd_status ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
