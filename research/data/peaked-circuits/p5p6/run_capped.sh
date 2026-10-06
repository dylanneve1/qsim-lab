#!/bin/bash
# usage: run_capped.sh NAME cmd...  -> 2.5 GB cap, nice, 2 BLAS threads, log to private/logs_NAME.log
name=$1; shift
cd /tmp/peaked-p5p6
log=private/logs_$name.log
PYTHONPATH=/tmp/peaked-generic OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore nohup prlimit --as=2700000000 nice /usr/bin/time -v "$@" > $log 2>&1 &
echo $! > .pid_$name
