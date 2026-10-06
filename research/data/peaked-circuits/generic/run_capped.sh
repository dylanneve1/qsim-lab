#!/bin/bash
# usage: run_capped.sh NAME cmd...  -> 2.5 GB address-space cap, niced, 2 BLAS threads, log to logs_NAME.log (private/ if PRIV=1)
name=$1; shift
cd /tmp/peaked-generic
log=logs_$name.log; [ "$PRIV" = "1" ] && log=private/logs_$name.log
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore nohup prlimit --as=2700000000 nice /usr/bin/time -v "$@" > $log 2>&1 &
echo $! > .pid_$name
