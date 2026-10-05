#!/usr/bin/env bash
# Runs the quimb_paths.py cotengra searches one after another.
# Machine rules (coordinator, 12:35): at most 2 worker processes per search, every python process capped
# with prlimit --as=3000000000, start only when `free -m` shows >= 8000 MB available.
# Usage: bash run_paths.sh [queue]   queue = all (default) | s0 | long | s1 | s0re
# While the file $WORK/PAUSE exists, or another search is running, the queue waits between runs (used
# to interleave the timed runs). Order = priority.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-/dev/shm/qsim/ext/tn-py/work}
export PYTHONPATH=/dev/shm/qsim/ext/tn-py OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/dev/shm/qsim/venv/bin/python
AS=3000000000
mkdir -p "$WORK"
avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
run() { # <budget> <circuit stem> <configs...>
  while [ -e "$WORK/PAUSE" ] || pgrep -f "^$PY [^ ]*quimb_paths.py" > /dev/null || [ "$(avail_mb)" -lt 8000 ]; do
    sleep 15
  done
  local budget=$1 stem=$2; shift 2
  echo "$(date -Iseconds) start $budget $stem $* (avail $(avail_mb) MB)"
  nice -n 15 prlimit --as=$AS "$PY" "$HERE/quimb_paths.py" "$HERE/circuits/$stem.txt" --budget "$budget" \
    --configs "$@" --parallel 2 --out "$HERE/cotengra_paths.jsonl" --save-dir "$WORK"
  local rc=$?
  echo "$(date -Iseconds) done $budget $stem rc=$rc"
}
Q=${1:-all}
# History: 12:35-12:40 m16/m18 seed 0 'std' unsliced finished, their sliced_sl crashed on the 3 GB cap with
# 2 workers (BrokenProcessPool); 12:40-12:53 sliced_sl in-process (SliceFinder max_repeats=4) worked for
# m18 (13 trials, poor) but hit MemoryError at m20. So from 12:55 sliced_sl only for m <= 14; for m >= 16
# the sliced figures come from quimb_paths.py --slice-reconf on the best unsliced tree (single process).
if [ "$Q" = all ] || [ "$Q" = long ]; then
  run long syc53_m14_s0 unsliced sliced_sl
  run long syc53_m20_s0 unsliced
fi
if [ "$Q" = all ] || [ "$Q" = s1 ]; then
  for m in 10 12 14; do run std "syc53_m${m}_s1" unsliced sliced_sl; done
  for m in 16 18 20; do run std "syc53_m${m}_s1" unsliced; done
fi
if [ "$Q" = all ] || [ "$Q" = s0re ]; then
  for m in 10 12 14; do run std "syc53_m${m}_s0" unsliced sliced_sl; done
fi
echo "$(date -Iseconds) queue $Q finished"
