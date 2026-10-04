#!/bin/bash
# Runs a resumable collector in locked chunks on the Mac (research/simulability/planner-v2.md §1):
# take the shared bench lock, run CMD (which stops after its --budget and
# exits 3 while work remains), release the lock, let peers in, repeat.
#   run_chunks.sh LOG CMD...
set -u
LOG=$1; shift
LOCK=${LOCK:-/tmp/qsim-mac-bench.lock}
held=0
trap '[ $held -eq 1 ] && rmdir "$LOCK"' EXIT
while true; do
  until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
  held=1
  echo "== lock $(date +%T) load $(sysctl -n vm.loadavg)" >> "$LOG"
  "$@" >> "$LOG" 2>&1; rc=$?
  rmdir "$LOCK"; held=0
  echo "== released $(date +%T) rc=$rc" >> "$LOG"
  [ $rc -eq 0 ] && break
  [ $rc -ne 3 ] && { echo "== error rc=$rc" >> "$LOG"; break; }
  sleep 30
done
echo "== done $(date +%T)" >> "$LOG"
