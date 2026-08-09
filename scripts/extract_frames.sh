#!/bin/bash
# =============================================================================
# extract_frames.sh — 从一段源视频里抽一对"几何上连得起来"的首尾帧。
#
#   scripts/extract_frames.sh <src.mp4|mov> <起始秒> <输出目录> [帧数] [宽] [高]
#
# 两帧的间隔 = (帧数-1)/24 秒,默认 123/24 —— 正好对应 H3 的 124 帧跨度。
# 间隔对不上的话,模型要在给定的两端之间凑出一个不自然的运动速度。
#
# 画布默认 864x480(与 scripts/h3.sh 一致)。上游主仓库这里写死的是 832x480,
# 和实际产线画布不一致;这里改成参数并把默认对齐。
#
# 这只是素材准备工具,不在生成的必需链路上 —— h3.sh 会再用同一套
# scale-to-cover + center-crop 把输入压到目标画布(见 prep_frames.sh)。
# =============================================================================
set -euo pipefail
SRC="${1:?需要源视频}"
T0="${2:?需要起始秒}"
OUT="${3:?需要输出目录}"
FRAMES="${4:-124}"
W="${5:-864}"
H="${6:-480}"

mkdir -p "$OUT"
VF="scale=${W}:${H}:force_original_aspect_ratio=increase,crop=${W}:${H}"
T1=$(python3 -c "print($T0 + ($FRAMES - 1)/24.0)")

ffmpeg -loglevel error -y -ss "$T0" -i "$SRC" -frames:v 1 -vf "$VF" "$OUT/first_${W}x${H}.png"
ffmpeg -loglevel error -y -ss "$T1" -i "$SRC" -frames:v 1 -vf "$VF" "$OUT/last_${W}x${H}.png"
echo "pair: t0=$T0 t1=$T1 (${FRAMES} 帧跨度 @24fps, ${W}x${H})"
ls -l "$OUT/first_${W}x${H}.png" "$OUT/last_${W}x${H}.png"
