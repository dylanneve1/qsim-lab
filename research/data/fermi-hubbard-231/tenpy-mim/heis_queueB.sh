#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2
for job in "nu 5" "dd 5" "nu 10" "dd 10" "nu 15" "dd 15" "nu 20" "dd 20" "nu 25" "dd 25" "dd 30"; do
  set -- $job
  nice -n 10 $PY -W ignore -u heis.py 60 29 $1 $2 512 > heis/log_$1_k$2_chi512.txt 2>&1
done
