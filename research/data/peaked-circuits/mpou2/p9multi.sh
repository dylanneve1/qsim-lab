#!/bin/bash
cd /tmp/peaked-mac/mpou2
for e in 1e-2 3e-3 1e-3 3e-4; do
  VECLIB_MAXIMUM_THREADS=4 nice ~/peaked-venv/bin/python -u run2.py P9_hqap_1917.qasm --m 49 --band 45 55 --tno_band --tno_cut 1e-3 --mode rel --eps $e --maxb 512 --log_every 200 --stop_elems 1e7 --tau 1e5 --match_hi 0 --max_time 1500 --sched pair --out private/p9_pair_e$e.json > logs/p9_pair_e$e.log 2>&1
  grep PEAK logs/p9_pair_e$e.log
done
