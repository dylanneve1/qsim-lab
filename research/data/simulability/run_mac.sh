#!/bin/bash
# Runs the simulability grids on the Mac in locked chunks (<= ~3 min each):
# up to $WORKERS (default 2) single-threaded driver workers in parallel, one
# grid each, each stopping after --budget seconds; release the lock, let
# peers in, repeat. Workers run under nice: it is a personal laptop.
#   [WORKERS=2] [REPS=1] run_mac.sh OUTDIR GRID1 GRID2 ...
set -u
OUT=$1; shift
BIN=$HOME/qsim-sim-target/release/examples/simulability
DRV=$HOME/qsim-sim/research/data/simulability/driver.py
mkdir -p "$OUT"
remaining=("$@")
while [ ${#remaining[@]} -gt 0 ]; do
  until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
  echo "== lock acquired $(date +%T) load $(sysctl -n vm.loadavg)"
  pids=(); grids=()
  for g in "${remaining[@]:0:${WORKERS:-2}}"; do
    nice -n 10 python3 "$DRV" --reps "${REPS:-1}" --bin "$BIN" --grid "$g" --out "$OUT/$g.csv" --timeout 10 --budget 90 \
      --threads 1 >> "$OUT/$g.log" 2>&1 &
    pids+=($!); grids+=("$g")
  done
  next=("${remaining[@]:${WORKERS:-2}}")
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    [ $rc -ne 0 ] && next+=("${grids[$i]}")
  done
  rmdir /tmp/qsim-mac-bench.lock
  echo "== lock released $(date +%T); remaining: ${next[*]:-none}"
  remaining=("${next[@]+"${next[@]}"}")
  [ ${#remaining[@]} -gt 0 ] && sleep 20
done
echo "== all done $(date +%T)"
