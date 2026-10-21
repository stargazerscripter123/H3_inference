#!/usr/bin/env bash
set -euo pipefail
# Run manually on the target host; argument is the full film path or URL.
# Timeline is relative to the full film, not to an already-trimmed clip.
: "${1:?Usage: bash derive_refs.sh FULL_FILM_PATH_OR_URL}"
source_video="$1"
ffmpeg_bin="${OMNI_FFMPEG:-/home/isaac/workdir/h3/.tools/media/bin/ffmpeg}"
ffprobe_bin="${OMNI_FFPROBE:-/home/isaac/workdir/h3/.tools/media/bin/ffprobe}"
# -n prevents overwriting an existing derived asset.
mkdir -p /home/isaac/workdir/h3/inputs/tos/bench32-v1

# s1_w_image_1 at full-film t=240.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 240.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_w_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_w_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_w_image_1.png.sha256

# s1_a_image_1 at full-film t=55.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 55.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_a_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_a_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_a_image_1.png.sha256

# s1_b_image_1 at full-film t=90.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 90.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_b_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_b_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s1_b_image_1.png.sha256

# s2_w_image_1 at full-film t=241.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 241.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_1.png.sha256

# s2_w_image_2 at full-film t=241.500s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 241.5 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_2.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_2.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_2.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_2.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_w_image_2.png.sha256

# s2_a_image_1 at full-film t=56.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 56.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_1.png.sha256

# s2_a_image_2 at full-film t=56.500s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 56.5 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_2.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_2.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_2.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_2.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_a_image_2.png.sha256

# s2_b_image_1 at full-film t=91.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 91.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_1.png.sha256

# s2_b_image_2 at full-film t=91.500s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 91.5 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_2.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_2.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_2.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_2.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s2_b_image_2.png.sha256

# s3_w_image_1 at full-film t=242.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 242.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_image_1.png.sha256

# s3_w_audio_2 at full-film t=242.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 242.0 -i "$source_video" -map 0:a:0 -t 4 -vn -c:a pcm_s16le -ar 32000 -ac 2 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_audio_2.wav
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_audio_2.wav > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_audio_2.wav.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_audio_2.wav > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_w_audio_2.wav.sha256

# s3_a_image_1 at full-film t=57.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 57.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_image_1.png.sha256

# s3_a_audio_2 at full-film t=57.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 57.0 -i "$source_video" -map 0:a:0 -t 4 -vn -c:a pcm_s16le -ar 32000 -ac 2 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_audio_2.wav
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_audio_2.wav > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_audio_2.wav.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_audio_2.wav > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_a_audio_2.wav.sha256

# s3_b_image_1 at full-film t=92.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 92.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_image_1.png.sha256

# s3_b_audio_2 at full-film t=92.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 92.0 -i "$source_video" -map 0:a:0 -t 4 -vn -c:a pcm_s16le -ar 32000 -ac 2 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_audio_2.wav
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_audio_2.wav > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_audio_2.wav.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_audio_2.wav > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s3_b_audio_2.wav.sha256

# s4_w_image_1 at full-film t=243.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 243.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_image_1.png.sha256

# s4_w_video_2 at full-film t=243.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 243.0 -i "$source_video" -map 0:v:0 -t 4 -an -vf scale=-2:480:flags=lanczos,setsar=1,fps=24 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_video_2.mp4
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_video_2.mp4 > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_video_2.mp4.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_video_2.mp4 > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_w_video_2.mp4.sha256

# s4_a_image_1 at full-film t=58.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 58.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_image_1.png.sha256

# s4_a_video_2 at full-film t=58.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 58.0 -i "$source_video" -map 0:v:0 -t 4 -an -vf scale=-2:480:flags=lanczos,setsar=1,fps=24 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_video_2.mp4
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_video_2.mp4 > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_video_2.mp4.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_video_2.mp4 > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_a_video_2.mp4.sha256

# s4_b_image_1 at full-film t=93.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 93.0 -i "$source_video" -map 0:v:0 -frames:v 1 -an -vf scale=-2:480:flags=lanczos,setsar=1 /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_image_1.png
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_image_1.png.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_image_1.png > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_image_1.png.sha256

# s4_b_video_2 at full-film t=93.000s
"$ffmpeg_bin" -hide_banner -loglevel error -n -ss 93.0 -i "$source_video" -map 0:v:0 -t 4 -an -vf scale=-2:480:flags=lanczos,setsar=1,fps=24 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_video_2.mp4
"$ffprobe_bin" -v error -count_frames -show_streams -show_format -of json /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_video_2.mp4 > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_video_2.mp4.ffprobe.json
sha256sum /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_video_2.mp4 > /home/isaac/workdir/h3/inputs/tos/bench32-v1/s4_b_video_2.mp4.sha256
