#!/bin/bash
# One benchmark cell: launch config, health-wait, warmup(+quality frame), timed x3.
# Timing convention (07 topic): engine-side FIXED seed, warmup 1 + timed 3, median.
# usage: bench_matrix.sh <original|turbo> <bf16|fp8> <tp4|tp2u2> <resident> [nfe]
set -u
WHICH="$1"; PREC="$2"; TOPO="$3"; RES="$4"; NFE="${5:-6}"
B=/workspace/h3
TAG="${WHICH}_${PREC}_${TOPO}_r${RES}_nfe${NFE}"
OUTDIR="$B/outputs/bench"; mkdir -p "$OUTDIR"
RESULTS="$B/outputs/bench_results.tsv"
LOG="$B/logs/vllm_${WHICH}-${PREC}-${TOPO}-r${RES}.log"

bash "$B/scripts/h3_switch_runpods.sh" "$WHICH" "$PREC" "$TOPO" "$RES" >/dev/null 2>&1
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8091/health 2>/dev/null)
  [ "$code" = 200 ] && break
  if ! pgrep -f '[v]llm serve' >/dev/null; then
    REASON=$(grep -oiE 'OutOfMemoryError|CUDA error[^\"]*' "$LOG" 2>/dev/null | head -1)
    printf '%s\tSERVER_DIED\t-\t-\t%s\n' "$TAG" "${REASON:-see $LOG}" | tee -a "$RESULTS"
    exit 1
  fi
  sleep 10
done

run_one() {
  "$B/env/bin/python" "$B/scripts/run_fl2va_vllm.py" \
    --first "$B/inputs/first_864x480.png" --last "$B/inputs/last_864x480.png" \
    --prompt-file "$B/workflows/smoke_prompt.txt" \
    --seed 0 --nfe "$NFE" --width 864 --height 480 \
    --out "$OUTDIR/${TAG}_$1.mp4" 2>&1 | tail -1
}

W=$(run_one warmup)
echo "WARMUP $TAG :: $W"
case "$W" in
  *DONE*) ;;
  *) printf '%s\tREQUEST_FAILED\t-\t-\t%s\n' "$TAG" "$W" | tee -a "$RESULTS"; exit 1 ;;
esac
ffmpeg -y -loglevel error -i "$OUTDIR/${TAG}_warmup.mp4" -ss 2 -frames:v 1 "$OUTDIR/${TAG}_f2s.png"
PEAK=$(echo "$W" | grep -o 'peak_vram\[[^]]*\]')

T=()
for i in 1 2 3; do
  R=$(run_one "t$i")
  T+=("$(echo "$R" | grep -o 'wall=[0-9.]*' | cut -d= -f2)")
done
MED=$(printf '%s\n' "${T[@]}" | sort -n | sed -n 2p)
printf '%s\tOK\t%s\t%s/%s/%s\t%s\n' "$TAG" "$MED" "${T[0]}" "${T[1]}" "${T[2]}" "$PEAK" | tee -a "$RESULTS"
