#!/bin/bash
# sweep chiO convergence at fixed psi(k0=10, chi 512): nu and dd, k=20,25,30
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2
for chiO in 256 512; do
 for k in 20 25 30; do
  for obs in nu dd; do
    nice -n 10 $PY -W ignore -u mim.py $obs 10 $k 512 $chiO >> mim/queue1.log 2>&1
  done
 done
done
