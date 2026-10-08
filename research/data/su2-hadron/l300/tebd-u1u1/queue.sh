#!/bin/bash
cd /tmp/su2-254-opus
P=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while pgrep -f "tebd_sym.py SCV 256 1 0 60" >/dev/null; do sleep 10; done
for job in "SCV 256 0" "meson 256 1" "meson 256 0" "SCV 512 1" "meson 512 1" "SCV 512 0" "meson 512 0"; do
  set -- $job
  f=tebd_$1_chi$2_lam$3.0_0_60.json
  nice -n 10 $P -W ignore tebd_sym.py $1 $2 $3 0 60 20 1e-10 > log_$1_$2_lam$3.txt 2>&1
  echo "done $job $(date +%T)" >> queue.log
done
