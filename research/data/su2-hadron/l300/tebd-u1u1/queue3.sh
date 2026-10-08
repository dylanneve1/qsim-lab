#!/bin/bash
cd /tmp/su2-254-opus
P=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while pgrep -f queue2.sh >/dev/null; do sleep 15; done
for job in "SCV 2048 0" "meson 2048 0" "SCV 2800 1" "meson 2800 1"; do
  set -- $job
  nice -n 10 $P -W ignore tebd_sym.py $1 $2 $3 0 60 20 1e-10 > log_$1_$2_lam$3.txt 2>&1
  echo "done $job $(date +%T)" >> queue.log
done
