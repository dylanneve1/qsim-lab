#!/bin/sh
# AMX bf16x3 vs AVX-512 f32 dense k-qubit kernels (examples/amx_dense.rs).
# usage: run.sh [binary] [outdir]
# Build first (std only, no cargo needed):
#   rustc --edition 2021 -C opt-level=3 -C target-cpu=native examples/amx_dense.rs -o amx_dense
# Accuracy needs no lock. Timing: ONE method per process (AMX lowers the
# core clock for a while after it runs, which biases an interleaved AVX-512
# measurement), processes alternating A B C D A B C D, min of 5 in-process
# reps per process; everything under the shared bench lock with uptime /
# free recorded before and after (rules: /dev/shm/qsim/MACHINE.md).
# Summarise with: python3 summarize.py speed-<tag>.md
set -e
BIN=${1:-/dev/shm/qsim/avx512/bin/amx_dense}
OUT=${2:-$(dirname "$0")}
P8=0,2,4,6,8,10,12,14   # one vCPU per physical core (siblings are 2k, 2k+1)
TAG=$(date +%Y%m%dT%H%M)
"$BIN" acc > "$OUT/accuracy.md"
flock /dev/shm/qsim/bench.lock sh -c "
  echo '# before'; uptime; free -g | head -2
  echo '## micro (cpu 4)'; taskset -c 4 $BIN micro
  for cfg in '17 1 4' '22 1 4' '24 1 4' '20 8 $P8' '22 8 $P8' '24 8 $P8'; do
    set -- \$cfg
    for k in 5 6 7; do
      for rep in 1 2; do
        for m in avx512-f32 amx-bf16x3 amx-bf16x2 amx-bf16x1; do
          taskset -c \$3 $BIN speed \$1 \$2 5 \$k only=\$m | grep '^| [567] '
        done
      done
    done
  done
  echo '# after'; uptime; free -g | head -2
" > "$OUT/speed-$TAG.md" 2>&1
echo "wrote $OUT/accuracy.md $OUT/speed-$TAG.md"
