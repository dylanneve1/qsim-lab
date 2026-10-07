#!/bin/bash
# sequential TEBD queue (nice, 2 threads)
export SU2_CIRCUITS=/tmp/su2-scout/repo/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /tmp/su2-pr/research/data/su2-hadron
run(){ c=$1; chi=$2; lam=$3; ns=$4; f=tebd_runs/${c}_chi${chi}_lam${lam}.json; [ -f $f ] && return; nice -n 15 python3 tebd_lam.py --circ $c --chi $chi --lam $lam --nsteps $ns --out $f > tebd_runs/${c}_chi${chi}_lam${lam}.log 2>&1; }
for c in SCV meson; do for l in 0 1; do run $c 64 $l 12; done; done
for l in 0 1 -1 2 4 8; do run SCV 128 $l 12; done
for l in 0 1 2 4 8; do run meson 128 $l 12; done
for c in SCV meson; do for l in 0 1; do run $c 256 $l 12; done; done
echo QUEUE_DONE > tebd_runs/DONE
