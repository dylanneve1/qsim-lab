#!/bin/bash
# usage: run_mpo.sh TAG QASM CUTOFF [extra args]  (VPS: 2.5 GB cap, nice, 3 threads)
tag=$1; q=$2; c=$3; shift 3
cd /tmp/peaked-p5p6/mposolver
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMBA_NUM_THREADS=3 RAYON_NUM_THREADS=3 nohup prlimit --as=2700000000 nice .venv/bin/p9solve --qasm $q --outdir ../private/mpo --tag $tag --samples 1000 --cutoff $c --no-parallel-rewire "$@" > ../private/mpo_$tag.out 2>&1 &
echo $! > ../.pid_mpo_$tag
