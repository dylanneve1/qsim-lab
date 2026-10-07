#!/bin/bash
# usage: run_capped.sh NAME cmd...  -> 2.5 GB cap, nice, 2 BLAS threads, log to logs/NAME.log
name=$1; shift
mkdir -p /tmp/p6-rings/logs
cd /tmp/p6-rings
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore nohup prlimit --as=2700000000 nice "$@" > logs/$name.log 2>&1 &
echo $!
