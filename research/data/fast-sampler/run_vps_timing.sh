#!/bin/bash
# x86 timing grid (VPS). usage: run_vps_timing.sh <stim_compare> <native stim> <out.jsonl>
B=$1; export STIM_CLI=$2; OUT=$3; export RAYON_NUM_THREADS=1
for p in 0.001 0.003; do
  for spec in "3 4000000" "7 1000000" "11 256000" "15 128000"; do
    set -- $spec
    /tmp/fw/bin/python "$(dirname "$0")/timing.py" "$B" /tmp/qsim-wt/fs-data/tim $1 $p $2 3 1 >> "$OUT" || echo FAIL
  done
done
