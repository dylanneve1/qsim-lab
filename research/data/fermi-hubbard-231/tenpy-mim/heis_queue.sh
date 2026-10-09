#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2
for chi in 256 512 1024; do
  nice -n 10 $PY -W ignore -u heis.py 60 29 nu 30 $chi > heis/log_nu_k30_chi$chi.txt 2>&1
done
