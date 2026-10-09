#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=3
while ps -eo args | grep -q "[m]im_queue3.sh"; do sleep 10; done
run() { nice -n 10 $PY -W ignore -u mim.py "$@" >> mim/queue4.log 2>&1; echo "done $* exit $?" >> mim/queue4.status; }
for k in 30 25; do for obs in nu dd; do run $obs 10 $k 256 1024; done; done
