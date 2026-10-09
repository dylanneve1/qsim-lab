#!/bin/bash
cd /tmp/fh-231-t
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 $PY -W ignore -c "
import tebd_fh as tf
tf.run(60,3072,22,1e-10,0.2,-2.0,tag='',outdir='runs')
" > runs/log_chi3072.txt 2>&1
