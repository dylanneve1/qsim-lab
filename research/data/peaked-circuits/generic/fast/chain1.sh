#!/bin/sh
# sequential benchmark chain (one job at a time). logs in logs/
PY=~/mlx-venv/bin/python
cd /tmp/peaked-mac/mlx
$PY bench.py grow P9_hqap_1917.qasm 50 np128 1e6 900 prof > logs/p9_grow_np128_prof.log 2>&1
for be in mlx np64 mlxcpu; do
  $PY bench.py grow P9_hqap_1917.qasm 50 $be 1e6 900 > logs/p9_grow_$be.log 2>&1
done
for f in P10_heavy_hex_4020 P4_golden_mountain; do
  for be in np128 np64 mlx mlxcpu; do
    $PY bench.py solve $f.qasm $be > logs/solve_${f%%_*}_$be.log 2>&1
  done
done
echo done > logs/chain1.done
