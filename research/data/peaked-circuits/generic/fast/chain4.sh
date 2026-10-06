#!/bin/sh
PY=~/mlx-venv/bin/python
cd /tmp/peaked-mac/mlx
while [ ! -f logs/chain3.done ]; do sleep 10; done
$PY -u bench_tnos.py P9_hqap_1917.qasm 50 fast 600 3e6 > logs/p9_tnos_fast_long.log 2>&1
echo done > logs/chain4.done
