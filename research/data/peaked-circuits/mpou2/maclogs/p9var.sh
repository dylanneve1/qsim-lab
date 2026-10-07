#!/bin/bash
cd /tmp/peaked-mac/mpou2
while read -r v; do
  tag=$(echo $v | tr -d ' -')
  VECLIB_MAXIMUM_THREADS=4 nice ~/peaked-venv/bin/python -u run2.py P9_hqap_1917.qasm $v --tno_band --mode rel --eps 1e-3 --maxb 512 --log_every 500 --stop_elems 3e6 --tau 1e5 --match_hi 0 --max_time 200 --sched pair --out private/p9_var_$tag.json > logs/p9_var_$tag.log 2>&1
  echo "$v: $(grep -E 'PEAK|STOP' logs/p9_var_$tag.log | cut -c1-150)"
done
