#!/bin/sh
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd /tmp/mlx-port
while [ ! -f logs/chain_vps.done ]; do sleep 10; done
for o in "--lapack" "--c64 --batch --lapack"; do
  $PY bench_fast.py grow /tmp/peaked-gen/portal/P9_hqap_1917.qasm 50 fast 1e6 300 $o > logs/fast_grow_P9_fast_$(echo $o | tr -d ' -')_t1.log 2>&1
done
$PY bench_fast.py grow /tmp/peaked-gen/portal/P9_hqap_1917.qasm 50 quimb 1e6 300 > logs/fast_grow_P9_quimb_r2_t1.log 2>&1
for f in P10_heavy_hex_4020 P4_golden_mountain; do
  $PY bench_fast.py solve /tmp/peaked-gen/portal/$f.qasm fast --c64 --batch --lapack > logs/fsolve_${f%%_*}_fast_c64batchlapack.log 2>&1
done
echo done > logs/chain2_vps.done
