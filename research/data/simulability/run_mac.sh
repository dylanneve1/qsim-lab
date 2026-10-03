#!/bin/bash
# Runs the simulability grids on the Mac in locked chunks (<= ~3 min each):
# up to $WORKERS (default 2) single-threaded driver workers in parallel, one
# grid each, each stopping after --budget seconds; release the lock, let
# peers in, repeat. Workers run under nice: it is a personal laptop.
#   [WORKERS=2] [REPS=1] [OBS=all|mid2|mid4] run_mac.sh OUTDIR GRID1 GRID2 ...
set -u
OUT=$1; shift
BIN=$HOME/qsim-sim-target/release/examples/simulability
DRV=$HOME/qsim-sim/research/data/simulability/driver.py
LOCK=${LOCK:-/tmp/qsim-mac-bench.lock}
W=${WORKERS:-2}
R=${REPS:-1}
OBS=${OBS:-all}
SUF=""; [ "$OBS" != all ] && SUF=".$OBS"
mkdir -p "$OUT"
remaining=("$@")
held=0
# never leave the shared lock behind (bash 3.2 on macOS: keep expansions simple)
trap '[ $held -eq 1 ] && rmdir "$LOCK"' EXIT
while [ ${#remaining[@]} -gt 0 ]; do
  until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
  held=1
  echo "== lock acquired $(date +%T) load $(sysctl -n vm.loadavg)"
  pids=(); grids=()
  for g in "${remaining[@]:0:$W}"; do
    nice -n 10 python3 "$DRV" --reps "$R" --bin "$BIN" --grid "$g" --obs "$OBS" --out "$OUT/$g$SUF.csv" --timeout 10 --budget 90 \
      --threads 1 >> "$OUT/$g$SUF.log" 2>&1 &
    pids+=($!); grids+=("$g")
  done
  next=()
  [ ${#remaining[@]} -gt $W ] && next=("${remaining[@]:$W}")
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    [ $rc -ne 0 ] && next+=("${grids[$i]}")
  done
  rmdir "$LOCK"
  held=0
  echo "== lock released $(date +%T); remaining: ${next[*]:-none}"
  remaining=("${next[@]+"${next[@]}"}")
  [ ${#remaining[@]} -gt 0 ] && sleep 20
done
echo "== all done $(date +%T)"
