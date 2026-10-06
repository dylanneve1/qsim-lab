#!/bin/bash
# usage: run_bg.sh NAME cmd...   -> runs niced with 2 BLAS threads, logs to logs_NAME.log, pid in .pid_NAME
name=$1; shift
cd /tmp/peaked-generic
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore nohup nice /usr/bin/time -v "$@" > logs_$name.log 2>&1 &
echo $! > .pid_$name
