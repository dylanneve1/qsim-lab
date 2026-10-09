#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=3
while kill -0 2337933 2>/dev/null; do sleep 10; done
echo "done nu 10 30 256 1024 (queue4 job)" >> mim/queue5.status
run() { nice -n 10 prlimit --as=3000000000 $PY -W ignore -u mim.py "$@" >> mim/queue5.log 2>&1; echo "done $* exit $?" >> mim/queue5.status; }
run dd 10 30 256 1024
run nu 10 25 256 1024
run dd 10 25 256 1024
