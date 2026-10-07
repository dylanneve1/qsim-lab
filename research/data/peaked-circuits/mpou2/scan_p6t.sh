#!/bin/bash
cd /tmp/peaked-mac/mpou2
while read m lo hi; do
  VECLIB_MAXIMUM_THREADS=4 nice ~/peaked-venv/bin/python -u run2.py P6_titan_pinnacle.qasm --m $m --band $lo $hi --tno_band --tno_cut 1e-3 --mode rel --eps 1e-2 --maxb 512 --log_every 1 --stop_elems 3e5 --tau 1e5 --match_hi 0 --max_time 60 --sched pair > logs/scan6t_m${m}_${lo}_${hi}.log 2>&1
  echo "m=$m band [$lo,$hi): $(grep 'TNO centre' logs/scan6t_m${m}_${lo}_${hi}.log | cut -c1-90) | $(grep '^\[' logs/scan6t_m${m}_${lo}_${hi}.log | tail -1 | cut -c1-170)"
done
