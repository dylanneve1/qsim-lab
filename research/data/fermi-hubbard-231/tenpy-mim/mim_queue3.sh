#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=3
run() { nice -n 10 $PY -W ignore -u mim.py "$@" >> mim/queue3.log 2>&1; echo "done $* exit $?" >> mim/queue3.status; }
for k in 25 30; do for obs in nu dd; do run $obs 10 $k 256 512; done; done
for k in 25 30; do for obs in nu dd; do run $obs 10 $k 256 768; done; done
