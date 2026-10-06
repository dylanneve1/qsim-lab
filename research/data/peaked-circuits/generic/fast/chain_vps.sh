#!/bin/sh
# VPS benchmark chain for tnoq_fast vs tnoq(quimb); single-threaded BLAS, CPU time recorded (VPS is shared)
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd /tmp/mlx-port
$PY bench_fast.py grow /tmp/peaked-gen/portal/P9_hqap_1917.qasm 50 quimb 1e6 300 > logs/fast_grow_P9_quimb_t1.log 2>&1
$PY bench_fast.py grow /tmp/peaked-gen/portal/P9_hqap_1917.qasm 50 fast 1e6 300 > logs/fast_grow_P9_fast_t1.log 2>&1
for o in "--c64" "--batch" "--c64 --batch"; do
  $PY bench_fast.py grow /tmp/peaked-gen/portal/P9_hqap_1917.qasm 50 fast 1e6 300 $o > logs/fast_grow_P9_fast_$(echo $o | tr -d ' -')_t1.log 2>&1
done
for f in P10_heavy_hex_4020 P4_golden_mountain; do
  $PY bench_fast.py solve /tmp/peaked-gen/portal/$f.qasm quimb > logs/fsolve_${f%%_*}_quimb.log 2>&1
  $PY bench_fast.py solve /tmp/peaked-gen/portal/$f.qasm fast > logs/fsolve_${f%%_*}_fast.log 2>&1
  $PY bench_fast.py solve /tmp/peaked-gen/portal/$f.qasm fast --c64 > logs/fsolve_${f%%_*}_fast_c64.log 2>&1
  $PY bench_fast.py solve /tmp/peaked-gen/portal/$f.qasm fast --batch > logs/fsolve_${f%%_*}_fast_batch.log 2>&1
done
echo done > logs/chain_vps.done
