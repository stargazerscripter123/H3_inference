#!/bin/bash
# =============================================================================
# prep_frames.sh — 把任意尺寸的首/尾帧统一到目标画布,放进 ComfyUI 的 input 目录。
#
#   scripts/prep_frames.sh <源图> <目标文件名> <宽> <高>
#
# 目标文件名是**不带路径的文件名**,会写到 $H3_BASE/inputs/ 下(ComfyUI 的
# --input-directory 指向那里,LoadImage 节点只认那个目录里的文件名)。
#
# 为什么必须先过这一道,不能把原图直接丢给 ComfyUI
#   H3 对首帧和尾帧的处理是**不对称**的:首帧会被 stretch(非等比拉伸)、尾帧会被
#   crop。同一对图直接进去,两端的几何就对不上,生成出来会有一个渐变的形变漂移。
#   在外面用同一个变换(scale-to-cover + center-crop)把两张都压到目标画布,
#   两端几何一致,这个问题就没了。
#
#   scale=w:h:force_original_aspect_ratio=increase → 等比放大到"至少覆盖"画布
#   crop=w:h(默认居中)                            → 裁掉溢出部分
#   全程等比,不产生形变。
# =============================================================================
set -euo pipefail

SRC="${1:?需要源图路径}"
DSTNAME="${2:?需要目标文件名(不带路径)}"
W="${3:?需要宽}"
H="${4:?需要高}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../install/pins.env
source "$HERE/../install/pins.env"

[ -f "$SRC" ] || { echo "找不到源图: $SRC" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "缺 ffmpeg" >&2; exit 1; }

mkdir -p "$H3_BASE/inputs"
DST="$H3_BASE/inputs/$DSTNAME"

ffmpeg -loglevel error -y -i "$SRC" -frames:v 1 \
  -vf "scale=${W}:${H}:force_original_aspect_ratio=increase,crop=${W}:${H}" \
  "$DST"

echo "$DSTNAME"
