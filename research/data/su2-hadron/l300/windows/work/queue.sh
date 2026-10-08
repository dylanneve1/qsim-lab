#!/bin/bash
cd /tmp/su2-254-win/work
export OMP_NUM_THREADS=1
R="nice -n 10 python3 run_win.py"
for L in 6 8 10; do for c in SCV meson; do $R $L $c 1.0; done; done
for c in SCV meson; do $R 10 $c 0 0.25 -0.25 0.5 -0.5 -1.0 0.75 -0.75 1.5 -1.5 2.0 -2.0; done
for c in SCV meson; do $R 12 $c 1.0; done
echo ALLDONE
