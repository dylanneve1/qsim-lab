#!/bin/bash
# Reproduces research/data/ooc/scaling.csv. Every timed run goes through the swarm lock.
# Usage: research/data/ooc/run_bench.sh [out.csv]   (env: OOC_SCRATCH, default /tmp/ooc-scratch)
# Keeps >= 6 GB free on / by construction: the largest state file is n=28 f32 = 2 GiB.
set -u
OUT=${1:-research/data/ooc/scaling.csv}
BENCH=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh
E=./target/release/examples/ooc_bench
export OOC_SCRATCH=${OOC_SCRATCH:-/tmp/ooc-scratch}
mkdir -p "$OOC_SCRATCH"
$E header > "$OUT"
run() { # args to ooc_bench
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt 8 ]; then echo "skip (only ${avail}G free): $*" >&2; return; fi
  echo "[run] $*" >&2
  $BENCH $E "$@" >> "$OUT"
  rm -f "$OOC_SCRATCH"/qsim_ooc_state_*.dat
}
for wl in qft brick; do
  for n in 22 24 26; do run ram $wl $n f32 3; done
  for n in 22 24 26 28; do
    v=0; [ $n -le 24 ] && v=1
    c=$((n>22?22:n-1))
    run ooc $wl $n f32 swap $c 0 0 $v
    run ooc $wl $n f32 window 20 4 1 $v
  done
  for n in 26 28; do
    run ooc $wl $n f32 swap 25 0 0 0
    run ooc $wl $n f32 window 20 4 0 0
    run ooc $wl $n f32 window 22 2 1 0
    run ooc $wl $n f32 window 18 6 1 0
    run ooc $wl $n f32 window 19 5 1 0
  done
done
echo done >&2
