#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for chi in 128 256 512 1024 2048; do
  nice -n 10 $PY -W ignore tebd_fh.py 60 $chi 30 1e-10 0.2 -2.0 > runs/log_chi$chi.txt 2>&1
  mv tebd_L60_chi$chi.json runs/ 2>/dev/null
done
