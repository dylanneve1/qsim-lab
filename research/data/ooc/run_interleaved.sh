#!/bin/bash
# Interleaved old(swap) vs new(window) timings, 3 rounds, each run under bench.sh lock.
# Output: research/data/ooc/interleaved.csv ; load at each start in interleaved.load
cd "$(dirname "$0")/../../.."
export OOC_SCRATCH=/tmp/ooc-scratch; mkdir -p $OOC_SCRATCH
B=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh; E=./target/release/examples/ooc_bench
OUT=research/data/ooc/interleaved.csv; LOAD=research/data/ooc/interleaved.load
$E header > $OUT; : > $LOAD
for round in 1 2 3; do for wl in qft brick; do for n in 26 27 28; do
  c=22; [ $n -lt 28 ] && c=22
  echo "round $round $wl $n swap" >> $LOAD
  $B $E ooc $wl $n f32 swap $c 0 0 0 >> $OUT 2>> $LOAD; rm -f $OOC_SCRATCH/*.dat
  echo "round $round $wl $n window" >> $LOAD
  $B $E ooc $wl $n f32 window 20 4 1 0 >> $OUT 2>> $LOAD; rm -f $OOC_SCRATCH/*.dat
done; done; done
echo finished >> $LOAD
