#!/bin/bash
cd /tmp/su2-254-opus
P=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while pgrep -f queue.sh >/dev/null; do sleep 15; done
for job in "SCV 1024 1" "meson 1024 1" "SCV 1024 0" "meson 1024 0" "SCV 2048 1" "meson 2048 1"; do
  set -- $job
  avail=$(df --output=avail -k / | tail -1)
  nice -n 10 $P -W ignore tebd_sym.py $1 $2 $3 0 60 20 1e-10 > log_$1_$2_lam$3.txt 2>&1
  echo "done $job $(date +%T)" >> queue.log
done
