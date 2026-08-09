#!/bin/bash
# =============================================================================
# bootstrap.sh — 从零把这台机器装成能跑 H3 ComfyUI 产线的状态。
#
#   bash install/bootstrap.sh [--dry-run] [--only <step>] [--skip-models]
#
#   --dry-run     只打印每一步要做什么、目标路径在哪,不动任何东西
#   --only <step> 只跑某一步: dirs|conda|comfyui|nodes|models|link|doctor
#   --skip-models 跳过权重下载(40.28 GiB),其余照装
#
# 幂等:每一步都先检查现状,已经对了就跳过。可以反复跑。
#
# 版本全部来自 install/pins.env —— 改版本去改那个文件,不要改这里。
#
# 装完跑 install/doctor.sh 自检,然后:
#   ./scripts/h3.sh --profile comfy-int8-turbo-1c --nfe 4 \
#       --first a.png --last b.png --prompt-file prompts/smoke.txt
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pins.env
source "$HERE/pins.env"

DRY=0; ONLY=""; SKIP_MODELS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="${2:?--only 需要步骤名}"; shift ;;
    --skip-models) SKIP_MODELS=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '   \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else eval "$@"; fi; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

echo "项目根: $H3_BASE"
[ "$DRY" = 1 ] && echo "(dry-run:不会修改任何东西)"

# --- 0. 前提自检 --------------------------------------------------------------
# 这一步永远跑:后面每一步都建立在这些前提上,先失败比装到一半失败好。
say "0/6 前提自检"

command -v git   >/dev/null || die "缺 git"
command -v curl  >/dev/null || die "缺 curl"
command -v ffmpeg>/dev/null || die "缺 ffmpeg (apt install ffmpeg;prep_frames.sh 要用)"
ok "git / curl / ffmpeg 就位"

if command -v nvidia-smi >/dev/null; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
  ok "驱动 $DRV / ${NGPU} 卡 / 最小单卡显存 ${VRAM} MiB"
  # 驱动版本决定能用哪个 CUDA wheel 变体。cu130 需要 580+;低了会在 import torch 时
  # 报 "CUDA driver version is insufficient",而不是在装的时候报。
  [ "${DRV%%.*}" -ge "$H3_MIN_DRIVER" ] || die "驱动 $DRV < $H3_MIN_DRIVER,跑不了 torch $H3_TORCH"
  [ "$VRAM" -ge "$H3_MIN_VRAM_MIB" ] || warn "单卡显存 ${VRAM} MiB < ${H3_MIN_VRAM_MIB},可能 OOM"
  [ "$NGPU" -ge 2 ] || warn "只有 ${NGPU} 张卡;original(GPU1) 与 teacache/turbo(GPU0) 将挤在同一张卡上"
else
  die "没有 nvidia-smi"
fi

RAM=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
# 单个 ComfyUI worker 实测常驻 RSS 46.4 GiB。RAM 是这台机器最硬的约束 ——
# 三个 worker 共存会 OOM,所以 comfyctl.sh 强制互斥。
[ "$RAM" -ge "$H3_MIN_RAM_GIB" ] && ok "RAM ${RAM} GiB" \
  || warn "RAM ${RAM} GiB < ${H3_MIN_RAM_GIB};单 worker 就要 ~47 GiB"

DISK=$(df -BG --output=avail "$H3_BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${DISK:-0}" -ge "$H3_MIN_DISK_GIB" ] && ok "磁盘可用 ${DISK} GiB" \
  || warn "磁盘可用 ${DISK:-?} GiB < ${H3_MIN_DISK_GIB}(权重就要 40.3 GiB)"

# --- 1. 目录骨架 --------------------------------------------------------------
if want dirs; then
  say "1/6 目录骨架"
  for d in inputs outputs logs run loras prompts; do
    if [ -d "$H3_BASE/$d" ]; then ok "$d/ 已在"
    else run "mkdir -p '$H3_BASE/$d'" && ok "建 $d/"; fi
  done
fi

# --- 2. conda 环境 ------------------------------------------------------------
if want conda; then
  say "2/6 conda 环境 $H3_CONDA_ENV"
  CONDA="$H3_CONDA_ROOT/bin/conda"
  [ -x "$CONDA" ] || die "找不到 $CONDA(先装 miniconda,或设 H3_CONDA_ROOT)"

  if [ -x "$H3_PYTHON" ]; then
    ok "环境已存在: $H3_PYTHON"
  else
    warn "环境不存在,将新建 python $H3_PY_VERSION"
    run "'$CONDA' create -y -n '$H3_CONDA_ENV' python=$H3_PY_VERSION"
  fi

  # torch 必须先于 requirements 装,且必须指定 cu130 index ——
  # 直接 pip install torch 会拿到默认变体(当前是 cu13x 之外的),import 时才炸。
  # 这个检查在 dry-run 下也跑:它只是 import torch,只读。不跑的话 dry-run 会
  # 对一台其实已经装好的机器报"将安装 torch",预演结果就不可信了。
  if [ -x "$H3_PYTHON" ] \
     && "$H3_PYTHON" -c "import torch,sys; sys.exit(0 if torch.__version__=='$H3_TORCH' else 1)" 2>/dev/null; then
    ok "torch $H3_TORCH 已就位"
  else
    warn "装 torch $H3_TORCH (index: $H3_TORCH_INDEX)"
    # 三件套都钉版本:不钉的话 index 里选到的 torchvision/torchaudio 可能与
    # requirements.lock.txt 里的不一致,后一步再"对齐"就会把它们重装一遍。
    run "'$H3_PYTHON' -m pip install --index-url '$H3_TORCH_INDEX' \
         'torch==$H3_TORCH' 'torchvision==$H3_TORCHVISION' 'torchaudio==$H3_TORCHAUDIO'"
  fi

  # requirements.lock.txt 是**权威包列表**,也是唯一被真正执行的那份 ——
  # conda 在这个环境里只管 python/pip/openssl,其余 107 个包全是 pip 装的。
  # 锁文件自带 --extra-index-url,单独 `pip install -r` 也能用。
  #
  # install/environment.yml 是同一环境的 conda 侧快照,**故意不拿它来建环境**:
  # 它的 conda 依赖带 defaults 频道的 build string(换机器/换日期未必解析得出),
  # pip: 段又和 requirements.lock.txt 重复。它的用途是出问题时拿来 diff
  # "现在的环境和采集那天差在哪",不是安装路径。
  warn "按 requirements.lock.txt 对齐其余 107 个包"
  run "'$H3_PYTHON' -m pip install -r '$HERE/requirements.lock.txt'"
fi

# --- 3. ComfyUI ---------------------------------------------------------------
if want comfyui; then
  say "3/6 ComfyUI @ ${COMFYUI_COMMIT:0:7} (v$COMFYUI_VERSION)"
  C="$H3_BASE/ComfyUI"
  if [ -d "$C/.git" ]; then
    CUR=$(git -C "$C" rev-parse HEAD 2>/dev/null)
    if [ "$CUR" = "$COMFYUI_COMMIT" ]; then ok "已在 pin 上"
    else
      warn "当前 $CUR,切到 pin"
      run "git -C '$C' fetch origin '$COMFYUI_COMMIT'"
      run "git -C '$C' checkout --detach '$COMFYUI_COMMIT'"
    fi
  else
    run "git clone '$COMFYUI_REPO' '$C'"
    run "git -C '$C' checkout --detach '$COMFYUI_COMMIT'"
  fi
  run "'$H3_PYTHON' -m pip install -r '$C/requirements.txt'"
fi

# --- 4. custom_nodes ----------------------------------------------------------
if want nodes; then
  say "4/6 custom_nodes(三个都零 pip 依赖)"
  CN="$H3_BASE/ComfyUI/custom_nodes"
  run "mkdir -p '$CN'"
  # 名字 repo commit 三元组;顺序无关。
  install_node() {
    local name="$1" repo="$2" commit="$3" d="$CN/$1"
    if [ -d "$d/.git" ]; then
      local cur; cur=$(git -C "$d" rev-parse HEAD 2>/dev/null)
      if [ "$cur" = "$commit" ]; then ok "$name 已在 pin ${commit:0:7}"; return; fi
      warn "$name 当前 $cur,切到 ${commit:0:7}"
      run "git -C '$d' fetch origin '$commit'"
      run "git -C '$d' checkout --detach '$commit'"
    else
      run "git clone '$repo' '$d'"
      run "git -C '$d' checkout --detach '$commit'"
      ok "$name -> ${commit:0:7}"
    fi
  }
  install_node ComfyUI-MiniMax-H3-Turbo    "$NODE_TURBO_REPO"    "$NODE_TURBO_COMMIT"
  install_node ComfyUI-MiniMaxH3-TeaCache  "$NODE_TEACACHE_REPO" "$NODE_TEACACHE_COMMIT"
  install_node ComfyUI-MultiGPU            "$NODE_MULTIGPU_REPO" "$NODE_MULTIGPU_COMMIT"

  # Turbo 节点仓库自带二进制资产,git clone 会带下来。如果有人改成打包/复制目录
  # 分发就可能漏掉,而漏了之后节点仍能 import、只在采样时才出问题 —— 所以在这里挡。
  A="$CN/ComfyUI-MiniMax-H3-Turbo/$NODE_TURBO_ASSET"
  if [ "$DRY" = 1 ]; then echo "   [dry-run] 校验 $A 存在"
  elif [ -s "$A" ]; then ok "Turbo 节点自带资产在 ($(du -h "$A" | cut -f1))"
  else die "Turbo 节点缺资产 $NODE_TURBO_ASSET —— 用 git clone,别用打包目录"; fi
fi

# --- 5. 权重 ------------------------------------------------------------------
if want models && [ "$SKIP_MODELS" = 0 ]; then
  say "5/6 权重(最小集 40.28 GiB)"
  if [ "$DRY" = 1 ]; then echo "   [dry-run] bash $HERE/download_models.sh"
  else bash "$HERE/download_models.sh" || die "权重获取失败"; fi
elif want models; then
  say "5/6 权重 —— 按 --skip-models 跳过"
fi

# --- 6. LoRA 符号链接 ---------------------------------------------------------
if want link; then
  say "6/6 LoRA 符号链接"
  # LoRA 实体扁平放在 $H3_BASE/loras/,ComfyUI 只认 models/loras/ 下的东西。
  # 用 symlink 而不是复制:744 MiB 不算大,但两份会让"改了哪份"变成新的错误来源。
  SRC="$H3_BASE/loras/minimax_h3_turbo_v4_step600_ema.safetensors"
  DST="$H3_BASE/$H3_LORA_LINK"
  if [ "$DRY" = 1 ]; then echo "   [dry-run] ln -sf $SRC $DST"
  elif [ -e "$SRC" ]; then
    run "mkdir -p '$(dirname "$DST")'"
    run "ln -sfn '$SRC' '$DST'"
    [ -e "$DST" ] && ok "链接就位 -> $(readlink "$DST")" || die "链接断了"
  else
    warn "LoRA 实体还没下载($SRC),跳过链接;下完权重后重跑 --only link"
  fi
fi

# --- 收尾 --------------------------------------------------------------------
if want doctor && [ "$DRY" = 0 ] && [ -z "$ONLY" ]; then
  say "自检"
  bash "$HERE/doctor.sh"
else
  say "完成"
  echo "   下一步: bash install/doctor.sh"
fi
