#!/bin/bash
cd /tmp/peaked-mac/mpou2
for m in "$@"; do
  VECLIB_MAXIMUM_THREADS=4 nice ~/peaked-venv/bin/python -u run2.py P6_titan_pinnacle.qasm --m $m --mode rel --eps 1e-2 --maxb 512 --log_every 1 --stop_elems 3e5 --tau 1e5 --match_hi 0 --max_time 40 --sched pair > logs/scan6_m$m.log 2>&1
  echo "m=$m $(grep '^\[' logs/scan6_m$m.log | tail -1 | cut -c1-200)"
done
