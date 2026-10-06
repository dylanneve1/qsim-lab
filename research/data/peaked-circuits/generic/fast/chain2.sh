#!/bin/sh
PY=~/mlx-venv/bin/python
cd /tmp/peaked-mac/mlx
while [ ! -f logs/chain1.done ]; do sleep 10; done
$PY -u test_fast.py > logs/test_fast_mac.log 2>&1
for rep in 1 2; do for be in np128 np64 mlx; do
  $PY bench.py grow P9_hqap_1917.qasm 50 $be 1e6 900 > logs/p9_grow_${be}_r$rep.log 2>&1
done; done
for v in fast orig fast64 orig64; do
  $PY -u bench_tnos.py P9_hqap_1917.qasm 50 $v 900 3e5 > logs/p9_tnos_$v.log 2>&1
done
$PY -u kernels.py > logs/kernels.log 2>&1
echo done > logs/chain2.done
