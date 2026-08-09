#!/bin/bash
# Extract a geometrically-connectable first/last frame pair from a source video.
# Frames are exactly 123 source-frames apart at 24fps (= H3 124-frame span) and
# formatted to the 832x480 canvas (scale-to-cover + center crop, no distortion).
# Usage: extract_frames.sh <src.mov> <start_seconds> <outdir>
set -euo pipefail
SRC="${1:?src}"
T0="${2:?start seconds}"
OUT="${3:?outdir}"
mkdir -p "$OUT"

VF="scale=832:480:force_original_aspect_ratio=increase,crop=832:480"
T1=$(python3 -c "print($T0 + 123/24.0)")

ffmpeg -loglevel error -y -ss "$T0" -i "$SRC" -frames:v 1 -vf "$VF" "$OUT/first_832x480.png"
ffmpeg -loglevel error -y -ss "$T1" -i "$SRC" -frames:v 1 -vf "$VF" "$OUT/last_832x480.png"
echo "pair: t0=$T0 t1=$T1"
ls -l "$OUT"/first_832x480.png "$OUT"/last_832x480.png
