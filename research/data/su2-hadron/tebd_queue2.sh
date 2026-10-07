#!/bin/bash
export SU2_CIRCUITS=/tmp/su2-scout/repo/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /tmp/su2-pr/research/data/su2-hadron
while [ ! -f tebd_runs/SCV_chi128_lam0.json ]; do sleep 10; done
run(){ c=$1; chi=$2; lam=$3; ns=$4; f=tebd_runs/${c}_chi${chi}_lam${lam}.json; [ -f $f ] && return; nice -n 15 python3 tebd_lam.py --circ $c --chi $chi --lam $lam --nsteps $ns --out $f > tebd_runs/${c}_chi${chi}_lam${lam}.log 2>&1; }
for l in -1 2 4 8; do run SCV 64 $l 10; done
for l in 2 4 8; do run meson 64 $l 10; done
run SCV 128 1 12
run meson 128 0 12; run meson 128 1 12
run SCV 256 0 10; run SCV 256 1 10
echo QUEUE_DONE > tebd_runs/DONE
