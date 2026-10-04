#!/bin/bash
# minimal number of hook-free plaquettes for D=d (space-only screen, then 1-round spacetime), free orders
d=$1; mode=${2:-free}; lo=${3:-0}
for k in $(seq $lo 200); do
  r=$(nice -n 15 /tmp/cg-venv/bin/python cg_sat.py $d $d 1 $mode --space --hookfree $k | grep RESULT)
  echo "d=$d space K=$k $r"
  if echo "$r" | grep -q FOUND; then
    r2=$(nice -n 15 /tmp/cg-venv/bin/python cg_sat.py $d $d 1 $mode --warm --hookfree $k --log runs/hf_d${d}_${mode}_K$k.jsonl | grep -E "RESULT|FOUND")
    echo "d=$d R=1 K=$k $r2" | cut -c1-2000
    echo "$r2" | grep -q "RESULT FOUND" && break
  fi
done
