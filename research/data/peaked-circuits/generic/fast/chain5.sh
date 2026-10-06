#!/bin/sh
PY=~/mlx-venv/bin/python
cd /tmp/peaked-mac/mlx
while [ ! -f logs/chain4.done ]; do sleep 10; done
$PY -u bench_tnos.py P9_hqap_1917.qasm 50 fast 600 3e5 1e-5 > logs/p9_tnos_fast_lc1e-5.log 2>&1
$PY -u bench_tnos.py P9_hqap_1917.qasm 50 fast64 600 3e5 1e-5 > logs/p9_tnos_fast64_lc1e-5.log 2>&1
echo done > logs/chain5.done
