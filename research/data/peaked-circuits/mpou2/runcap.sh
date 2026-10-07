#!/bin/bash
# usage: runcap.sh NAME args...  (2.5 GB cap, nice, 2 BLAS threads) -> logs/NAME.log, pid in logs/NAME.pid
name=$1; shift
mkdir -p /tmp/mpou-v2/logs; cd /tmp/mpou-v2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore nohup prlimit --as=2700000000 nice /tmp/pk/research/data/peaked-circuits/.venv/bin/python -u "$@" > logs/$name.log 2>&1 &
echo $! > logs/$name.pid
