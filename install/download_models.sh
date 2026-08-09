#!/bin/bash
# =============================================================================
# download_models.sh — 取本部署需要的 5 个权重(合计 40.28 GiB)并逐个校验。
#
#   bash install/download_models.sh [--verify-only] [--force]
#
#   --verify-only  不下载,只对已有文件做 bytes + sha256 校验(装完体检用)
#   --force        即使校验通过也重下
#
# 清单、revision、期望 bytes 与 sha256 全部来自 install/pins.env。
#
# 为什么校验到 sha256 而不是只看字节数
#   Comfy-Org/MiniMax-H3 里 minimax_h3_ref2va_pruned_int8_convrot.safetensors 与
#   minimax_h3_fl2va_pruned_int8_convrot.safetensors **字节数完全相同**(20970379616)。
#   下错了 ComfyUI 照样加载、照样出片,只是结果莫名其妙。size-only 挡不住这个。
#
# 断点续传
#   走 curl -C -。40 GiB 断在中途很常见,重跑本脚本即可接上。
#   续传后仍然要过 sha256 —— 半个文件拼上另一半的字节数可能是对的。
#
# 认证
#   这两个 repo 目前都是公开的(gated=false),不需要 token。
#   如果哪天变 gated:把 token 放进 ~/.cache/huggingface/token 或导出 HF_TOKEN,
#   本脚本会自动带上 Authorization 头。**不要把 token 写进本仓库任何文件。**
#   另:匿名大流量下载会被 HF 限速(实测日累计几百 GB 后降到 ~3MB/s),认证后恢复。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pins.env
source "$HERE/pins.env"

VERIFY_ONLY=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done

ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '   \033[31m✗\033[0m %s\n' "$*" >&2; }

TOKEN="${HF_TOKEN:-}"
[ -z "$TOKEN" ] && [ -r "$HOME/.cache/huggingface/token" ] && TOKEN=$(tr -d '\n' < "$HOME/.cache/huggingface/token")
AUTH=()
[ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")

fail=0
for spec in "${H3_WEIGHTS[@]}"; do
  IFS='|' read -r key rel dst want_bytes want_sha <<< "$spec"
  case "$key" in
    comfy) repo="$HF_COMFY_REPO"; rev="$HF_COMFY_REV" ;;
    lora)  repo="$HF_LORA_REPO";  rev="$HF_LORA_REV"  ;;
    *) bad "pins.env 里未知的 repo key: $key"; fail=1; continue ;;
  esac
  out="$H3_BASE/$dst"
  url="https://huggingface.co/$repo/resolve/$rev/$rel"
  name=$(basename "$dst")

  printf '\n\033[1m%s\033[0m  (%s GiB)\n' "$name" "$(awk "BEGIN{printf \"%.2f\", $want_bytes/1073741824}")"

  verify() { # 返回 0 = 完全通过
    [ -f "$out" ] || { echo "missing"; return 1; }
    local have; have=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
    [ "$have" = "$want_bytes" ] || { echo "bytes $have != $want_bytes"; return 1; }
    local sha; sha=$(sha256sum "$out" | cut -d' ' -f1)
    [ "$sha" = "$want_sha" ] || { echo "sha256 $sha != $want_sha"; return 1; }
    return 0
  }

  if [ "$FORCE" = 0 ]; then
    if msg=$(verify); then ok "已在且校验通过"; continue
    elif [ "$VERIFY_ONLY" = 1 ]; then bad "校验失败: $msg"; fail=1; continue
    else warn "需要下载/续传: $msg"; fi
  elif [ "$VERIFY_ONLY" = 1 ]; then
    if msg=$(verify); then ok "校验通过"; else bad "校验失败: $msg"; fail=1; fi
    continue
  fi

  mkdir -p "$(dirname "$out")"
  echo "   ← $repo@${rev:0:7}/$rel"
  # -C - 续传;-f 让 HTTP 错误变成非零退出(否则会把错误页面写进文件);
  # --retry 应付 HF 偶发 5xx。
  if ! curl -fL -C - --retry 5 --retry-delay 5 "${AUTH[@]}" -o "$out" "$url"; then
    bad "下载失败"
    [ -z "$TOKEN" ] && warn "没有检测到 HF token;若该 repo 已改为 gated,需要认证"
    fail=1; continue
  fi
  if msg=$(verify); then ok "下载完成并校验通过"
  else bad "下载后校验仍失败: $msg"; fail=1; fi
done

echo
if [ "$fail" = 0 ]; then
  ok "全部 ${#H3_WEIGHTS[@]} 个权重 bytes + sha256 校验通过"
else
  bad "有权重未通过校验 —— 不要拿这套权重跑生成,结果不可信"
fi
exit "$fail"
