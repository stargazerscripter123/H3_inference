#!/bin/bash
# 公开仓库清洗门禁 —— 推送前跑,三项必须全 0。
#   bash scripts/check_repo_clean.sh
# 只扫**将要提交的内容**(git ls-files),不扫工作区:credentials/ 与 data/ 本来就
# 被 .gitignore 排除,扫工作区只会得到一堆无关命中然后被人习惯性忽略。
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

# 密钥:必须写成"前缀 + 至少 16 位随机体"。只匹配裸前缀(如 sk-or-v1-)会让
# 记录本正则的文档自己命中 —— 一个恒假警报的门禁比没有门禁更危险。
SECRET_RE='(hf|ghp|gho|ghs|github_pat)_[A-Za-z0-9]{16,}'
SECRET_RE+='|sk-or-v1-[a-f0-9]{16,}|sk-ant-[A-Za-z0-9_-]{16,}'
SECRET_RE+='|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY'

# 红线素材:项目 CLAUDE.md 立的规则 —— 未授权的真实可识别人物影像不得入库。
# 这里只按已知文件名前缀兜底;规则本身靠人守,门禁只挡复发。
REDLINE_RE='G21K'

fail=0
report() { # $1=名目 $2=命中内容
  local n; n=$(printf '%s' "$2" | grep -c . || true)
  if [ "$n" -eq 0 ]; then printf '  ✓ %-12s 0\n' "$1"
  else printf '  ✗ %-12s %d\n' "$1" "$n"; printf '%s\n' "$2" | sed 's/^/      /'; fail=1; fi
}

files=$(git ls-files)
report "密钥"      "$(printf '%s\n' "$files" | xargs grep -nIE "$SECRET_RE" 2>/dev/null)"
report "红线素材"  "$(printf '%s\n' "$files" | grep -iE "$REDLINE_RE")"
report "媒体/权重" "$(printf '%s\n' "$files" | grep -E '\.(mp4|mov|safetensors|pt|bin|ckpt)$')"

if [ "$fail" -eq 0 ]; then echo "门禁通过,可以推送"; else echo "门禁未过 —— 不要推送" >&2; fi
exit "$fail"
