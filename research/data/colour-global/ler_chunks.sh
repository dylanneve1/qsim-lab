#!/bin/bash
# Interleaved A/B LER chunks (kf vs global D8 schedule), d=9, rounds=9, noisy-CNOT, BP+OSD order 100.
# Each chunk holds the Mac bench lock (< 3 min) and releases it before the next one.
cd ~/qsim-cg
OUT=~/qsim-wt-logs/cg_ler_d9.jsonl
L=./target/release/examples/color_ler
for spec in "0.003 300000 4" "0.002 500000 4" "0.0015 700000 4"; do
  set -- $spec; p=$1; shots=$2; n=$3
  for c in $(seq 1 $n); do
    for arm in kf d9_global_D8.sched; do
      seed=$(( 7000000 + c * 1000 + ${#arm} * 37 + ${p#0.} ))
      until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
      load=$(sysctl -n vm.loadavg | awk '{print $2}')
      r=$($L 9 9 cnot $p $arm $shots $seed 8 100)
      rmdir /tmp/qsim-mac-bench.lock
      echo "{\"chunk\":$c,\"seed\":$seed,\"load\":$load,\"r\":$r}" >> $OUT
      sleep 10
    done
  done
done
echo DONE >> $OUT
