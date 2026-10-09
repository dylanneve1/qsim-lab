#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2
while ps -eo args | grep -q "[m]im_queue1.sh"; do sleep 10; done
run() { nice -n 10 $PY -W ignore -u mim.py "$@" >> mim/queue2.log 2>&1; }
for obs in nu dd; do run $obs 10 15 512 256; done
for k in 25 30; do for obs in nu dd; do run $obs 12 $k 768 1024; done; done
for k in 25 30; do for obs in nu dd; do run $obs 12 $k 768 512; done; done
