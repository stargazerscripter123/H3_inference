#!/bin/bash
# =============================================================================
# doctor.sh — 装完/出问题时跑这个。它给结论,不是只打印一堆信息。
#
#   bash install/doctor.sh [--skip-sha]
#
#   --skip-sha  跳过 40 GiB 的 sha256 复核(快很多,但只验字节数)
#
# 退出码 0 = 全绿;非 0 = 有项目没过,上面会写清是哪一项、该怎么修。
#
# 设计原则:每一项都要能证伪。"没报错"不等于"是对的" —— 这条产线上有好几处
# 出错时是静默的(LoRA 加载 0/518 不报错、TeaCache 缓存失效不报错、下错权重不报错),
# 所以这里检查的是**可观测的证据**,不是命令的退出码。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pins.env
source "$HERE/pins.env"

SKIP_SHA=0
[ "${1:-}" = "--skip-sha" ] && SKIP_SHA=1

pass=0; fail=0
ok()   { printf '\033[32m ✓\033[0m %-34s %s\n' "$1" "${2:-}"; pass=$((pass+1)); }
bad()  { printf '\033[31m ✗\033[0m %-34s %s\n' "$1" "${2:-}"; fail=$((fail+1)); }
info() { printf '\033[33m ·\033[0m %-34s %s\n' "$1" "${2:-}"; }

echo "H3 ComfyUI 部署自检 — $H3_BASE"
echo

# --- 硬件与系统 ---------------------------------------------------------------
if command -v nvidia-smi >/dev/null; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if [ "${DRV%%.*}" -ge "$H3_MIN_DRIVER" ]; then ok "驱动" "$DRV (需 ≥$H3_MIN_DRIVER)"
  else bad "驱动" "$DRV < $H3_MIN_DRIVER —— torch $H3_TORCH 跑不起来"; fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader \
    | while IFS=, read -r i n t u; do info "  GPU$i" "$n /$t 已用$u"; done
else bad "nvidia-smi" "没装"; fi

RAM=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
# 单 worker 实测 RSS 46.4 GiB。RAM 不够不会报错,会在起第二个 worker 时被 OOM killer 杀。
if [ "$RAM" -ge "$H3_MIN_RAM_GIB" ]; then ok "RAM" "${RAM} GiB (单 worker 要 ~47 GiB)"
else bad "RAM" "${RAM} GiB < ${H3_MIN_RAM_GIB}"; fi

command -v ffmpeg >/dev/null && ok "ffmpeg" "$(ffmpeg -version | head -1 | cut -d' ' -f3)" \
  || bad "ffmpeg" "没装 —— prep_frames.sh 要用"

# --- Python 环境 --------------------------------------------------------------
if [ -x "$H3_PYTHON" ]; then
  V=$("$H3_PYTHON" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null)
  ok "python" "$V  ($H3_PYTHON)"
  TI=$("$H3_PYTHON" - <<'PY' 2>/dev/null
import torch
print(f"{torch.__version__}|{torch.version.cuda}|{torch.cuda.device_count()}|{int(torch.cuda.is_available())}")
PY
)
  if [ -n "$TI" ]; then
    IFS='|' read -r tv tc td ta <<< "$TI"
    [ "$tv" = "$H3_TORCH" ] && ok "torch" "$tv / cuda $tc" \
      || bad "torch" "$tv != pin $H3_TORCH"
    [ "$ta" = 1 ] && ok "torch.cuda" "可用, ${td} 张卡" || bad "torch.cuda" "不可用"
  else bad "torch" "import 失败"; fi
else bad "conda 环境" "找不到 $H3_PYTHON"; fi

# --- ComfyUI 与节点 -----------------------------------------------------------
C="$H3_BASE/ComfyUI"
if [ -d "$C/.git" ]; then
  CUR=$(git -C "$C" rev-parse HEAD)
  [ "$CUR" = "$COMFYUI_COMMIT" ] && ok "ComfyUI" "@${CUR:0:7} (v$COMFYUI_VERSION)" \
    || bad "ComfyUI" "@${CUR:0:7} != pin ${COMFYUI_COMMIT:0:7} —— 换版本会改变采样路径"
else bad "ComfyUI" "$C 不是 git 仓库"; fi

check_node() { # $1=目录名 $2=期望 commit
  local d="$C/custom_nodes/$1"
  if [ -d "$d/.git" ]; then
    local cur; cur=$(git -C "$d" rev-parse HEAD)
    [ "$cur" = "$2" ] && ok "node $1" "@${cur:0:7}" || bad "node $1" "@${cur:0:7} != ${2:0:7}"
  else bad "node $1" "缺失"; fi
}
check_node ComfyUI-MiniMax-H3-Turbo   "$NODE_TURBO_COMMIT"
check_node ComfyUI-MiniMaxH3-TeaCache "$NODE_TEACACHE_COMMIT"
# MultiGPU 三个 profile 都不用它的节点,但它一加载就 patch model_management,
# 本部署的全部延迟数字都在它在场的前提下测得。缺了不会报错,只是数字不可比。
check_node ComfyUI-MultiGPU           "$NODE_MULTIGPU_COMMIT"

A="$C/custom_nodes/ComfyUI-MiniMax-H3-Turbo/$NODE_TURBO_ASSET"
[ -s "$A" ] && ok "Turbo 节点自带资产" "$(du -h "$A" | cut -f1)" \
  || bad "Turbo 节点自带资产" "缺 $NODE_TURBO_ASSET"

# --- 权重 --------------------------------------------------------------------
echo
if [ "$SKIP_SHA" = 1 ]; then
  info "权重" "只验字节数 (--skip-sha)"
  for spec in "${H3_WEIGHTS[@]}"; do
    IFS='|' read -r _ _ dst want _ <<< "$spec"
    f="$H3_BASE/$dst"; n=$(basename "$dst")
    if [ -f "$f" ]; then
      have=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
      [ "$have" = "$want" ] && ok "  $n" "bytes ok" || bad "  $n" "$have != $want"
    else bad "  $n" "缺失"; fi
  done
  info "" "注意:字节数分辨不了 fl2va 与 ref2va(两者大小相同),完整校验去掉 --skip-sha"
else
  if bash "$HERE/download_models.sh" --verify-only >/dev/null 2>&1; then
    ok "权重" "5 个文件 bytes + sha256 全部通过"
  else
    bad "权重" "有文件未通过 —— 跑 bash install/download_models.sh --verify-only 看详情"
  fi
fi

L="$H3_BASE/$H3_LORA_LINK"
if [ -L "$L" ] && [ -e "$L" ]; then ok "LoRA 符号链接" "-> $(readlink "$L")"
elif [ -f "$L" ]; then ok "LoRA(实体文件)" "非链接,也可用"
else bad "LoRA 链接" "$L 断了或不存在 —— 跑 bootstrap.sh --only link"; fi

# --- 节点能否真的被 ComfyUI 认出来 --------------------------------------------
# 只看目录在不在是不够的:节点 import 失败时 ComfyUI 会跳过它继续启动,
# 到提交图的时候才报 "node type not found"。
#
# 有 worker 在跑的话就问它的 /object_info —— 那是 ComfyUI 自己的答案,最权威。
# 没有 worker 才退回到脱离运行时的 import 探测,但那种探测对**会 patch ComfyUI
# 内部注册表的节点**(比如 MultiGPU 要改 TripleCLIPLoader)必然失败,那是探测方式
# 的局限,不是节点坏了 —— 所以只对我们真正要用的三个类下判断。
echo
NEED="MiniMaxH3TurboLoRA MiniMaxH3TurboSampler MiniMaxH3TeaCache"
LIVE=""
for p in 8188 8189 8190; do healthy_p=$(curl -s -m 3 "http://127.0.0.1:$p/object_info" 2>/dev/null) && \
  [ -n "$healthy_p" ] && { LIVE="$healthy_p"; LIVEPORT=$p; break; }; done

if [ -n "$LIVE" ]; then
  for want in $NEED; do
    if grep -q "\"$want\"" <<< "$LIVE"; then ok "节点类 $want" "ComfyUI :$LIVEPORT 已注册"
    else bad "节点类 $want" "ComfyUI :$LIVEPORT 不认识它 —— 提交图会 'node type not found'"; fi
  done
elif [ -x "$H3_PYTHON" ] && [ -d "$C" ]; then
  info "节点检查" "没有 worker 在跑,退回离线 import 探测(结论较弱)"
  OUT=$(cd "$C" && "$H3_PYTHON" - <<'PY' 2>&1
import importlib.util, pathlib, sys
for d in sorted(pathlib.Path("custom_nodes").iterdir()):
    init = d / "__init__.py"
    if not init.is_file():
        continue
    spec = importlib.util.spec_from_file_location(f"cn_{d.name}", init)
    try:
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        print(d.name, sorted(getattr(m, "NODE_CLASS_MAPPINGS", {})))
    except Exception as e:
        # 会 patch ComfyUI 注册表的节点在这里必然抛错,不代表它坏了。
        print(d.name, f"OFFLINE-IMPORT-FAIL: {type(e).__name__}: {e}")
PY
)
  for want in $NEED; do
    if grep -q "$want" <<< "$OUT"; then ok "节点类 $want" "离线 import 到"
    else bad "节点类 $want" "离线 import 不到"; fi
  done
  if grep -q "OFFLINE-IMPORT-FAIL" <<< "$OUT"; then
    info "  离线探测局限" "$(grep -o '^[^ ]* OFFLINE-IMPORT-FAIL.*' <<< "$OUT" | head -1 | cut -c1-90)"
    info "" "起一个 worker 后重跑本检查即可得到权威结论"
  fi
fi

# --- worker 现状 --------------------------------------------------------------
echo
for p in 8188 8189 8190; do
  if curl -s -m 3 "http://127.0.0.1:$p/system_stats" >/dev/null 2>&1; then
    info "worker :$p" "在跑"
  fi
done
NW=$(pgrep -cf '[m]ain.py --listen 127.0.0.1 --port 81(88|89|90)' 2>/dev/null || echo 0)
# 三个 worker 共存会把 125 GiB RAM 吃满(单个 ~47 GiB)。comfyctl.sh 保证互斥;
# 这里如果看到 >2,说明有人绕过 comfyctl 手工起了 worker。
if [ "${NW:-0}" -le 2 ]; then ok "worker 数量" "${NW:-0}(上限 2:GPU0 与 GPU1 各一个)"
else bad "worker 数量" "$NW 个共存 —— RAM 会被吃满,用 scripts/comfyctl.sh stop 收拾"; fi

echo
if [ "$fail" = 0 ]; then
  printf '\033[32m全部 %d 项通过。\033[0m 试一条:\n' "$pass"
  echo "  ./scripts/h3.sh --profile comfy-int8-turbo-1c --nfe 4 \\"
  echo "      --first a.png --last b.png --prompt-file prompts/smoke.txt"
else
  printf '\033[31m%d 项未通过\033[0m(%d 项通过)。修完再跑一次。\n' "$fail" "$pass"
fi
exit $(( fail > 0 ))
