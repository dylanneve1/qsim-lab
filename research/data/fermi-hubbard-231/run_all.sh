#!/bin/bash
export OMP_NUM_THREADS=2
for chi in 64 128 256 512; do
    echo "Running chi=$chi"
    nice -n 10 python3 /tmp/fh-231/production_run.py $chi > /tmp/fh-231/log_chi_${chi}.txt 2>&1
done
