#!/bin/bash
cd /tmp/su2-254-win/work
export OMP_NUM_THREADS=1
while ! grep -q ALLDONE queue.log; do sleep 5; done
R="nice -n 10 python3 run_win.py"
for L in 6 8 12; do for c in SCV meson; do $R $L $c 0.0; done; done
echo ALLDONE2
