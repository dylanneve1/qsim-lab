#!/bin/bash
# one job at a time (RAM constraint)
export SU2_CIRCUITS=/tmp/su2-scout/repo/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /tmp/su2-pr/research/data/su2-hadron
while kill -0 397828 2>/dev/null; do sleep 10; done
run(){ c=$1; chi=$2; lam=$3; ns=$4; f=tebd_runs/${c}_chi${chi}_lam${lam}.json; [ -f $f ] && return; nice -n 15 python3 tebd_lam.py --circ $c --chi $chi --lam $lam --nsteps $ns --out $f > tebd_runs/${c}_chi${chi}_lam${lam}.log 2>&1; }
run meson 128 0 12; run meson 128 1 12
run SCV 256 0 9; run SCV 256 1 9
echo QUEUE_DONE > tebd_runs/DONE
