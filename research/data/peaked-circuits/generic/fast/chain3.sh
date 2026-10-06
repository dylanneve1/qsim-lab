#!/bin/sh
PY=~/mlx-venv/bin/python
cd /tmp/peaked-mac/mlx
while [ ! -f logs/chain2.done ]; do sleep 10; done
$PY -u bench_tnos.py P9_hqap_1917.qasm 50 fast1 900 3e5 > logs/p9_tnos_fast1.log 2>&1
echo done > logs/chain3.done
