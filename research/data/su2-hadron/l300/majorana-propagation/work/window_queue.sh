#!/bin/bash
export SU2_CIRCUITS=/tmp/su2-scout/repo/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /tmp/su2-pr/research/data/su2-hadron; mkdir -p window_runs
for spec in "SCV 0 8" "SCV 26 34" "meson 26 34" "SCV 0 10" "SCV 25 35" "meson 25 35" "SCV 24 36" "meson 24 36"; do
  set -- $spec; nice -n 15 python3 window_lam.py $1 $2 $3 20 > window_runs/$1_$2_$3.log 2>&1; mv window_$1_$2_$3.json window_runs/ 2>/dev/null
done
echo DONE > window_runs/DONE
