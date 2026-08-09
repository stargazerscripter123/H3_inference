#!/bin/bash
set -u
B=/workspace/h3
run(){ echo "### $* ###"; bash "$B/scripts/bench_matrix.sh" "$@" 2>&1 | tail -3; }
run turbo bf16 tp2u2 40 6
run turbo fp8  tp2u2 50 4
run turbo fp8  tp4   50 4
echo "MATRIX2_COMPLETE"
