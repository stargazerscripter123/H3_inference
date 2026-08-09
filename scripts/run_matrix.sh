#!/bin/bash
# Sequential benchmark matrix. Each cell: launch, health, warmup+frame, timed x3.
set -u
B=/workspace/h3
run(){ echo "### $* ###"; bash "$B/scripts/bench_matrix.sh" "$@" 2>&1 | tail -3; }
run turbo bf16 tp4   50 6
run turbo fp8  tp4   50 6
run turbo fp8  tp2u2 50 6
run turbo bf16 tp2u2 40 6
echo "MATRIX_COMPLETE"
