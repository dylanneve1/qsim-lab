#!/bin/sh
# Timing matrix in priority order (n = 26/28 first). Each phase is one driver run that
# appends to baselines_raw.csv; holds wait for load1 <= 16 until $GATE (HH:MM), after
# which they start anyway and the rows are reported as loaded (load columns in the CSV).
# Plans: make_matrix.py (see README.md for the exact commands).
set -u
cd "$(dirname "$0")"
P=${PYTHON:-/dev/shm/qsim/venv/bin/python}
S=${PLANS:-/dev/shm/qsim/avx512/scratch}
OUT=${OUT:-baselines_raw.csv}
GATE=${GATE:-14:40}
STOP=${STOP:-15:45}
run() {
  echo "=== phase $1 (reps $2, rounds $3) $(date +%T)"
  $P bench_matrix.py "$S/$1" "$OUT" --reps "$2" --rounds "$3" --max-hold 420 \
     --stop-at "$STOP" --gate-until "$GATE"
}
run p1_26_c64_8.json 2 2
run p2_28_c64_8_fast.json 1 3
run p3_28_c64_8_slow.json 1 2
run p4_26_c128_8.json 2 2
run p5_28_c128_8.json 1 2
run p6_24_c64_8.json 2 2
run p7_24_c128_8.json 2 2
run p8_28_c64_16_fast.json 1 3
run p9_2426_c64_16.json 2 2
echo "=== done $(date +%T)"
