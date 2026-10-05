#!/bin/bash
# Cross-machine correctness: every run gathers on the Mac and compares with the
# single-node blocked executor (--verify ref) or the analytic QFT (--verify qft).
W=~/qsim-dist-run/wan.sh
for prec in f64 f32; do
  for spec in "20 19 01" "20 17 00000001" "20 17 01101001" "21 18 10000000"; do
    set -- $spec; n=$1; L=$2; own=$3
    for wl in qft brick brickd; do
      for restore in 0 1; do
        $W run 2 300M 2 --workload $wl --depth 12 --n $n --prec $prec --local-bits $L --owner $own --basis 777 --restore $restore --verify ref 2>&1 | grep "node=0" | sed 's/ plan_ms.*norm/ ... norm/'
      done
    done
  done
done
$W run 2 300M 2 --workload qft --n 24 --prec f64 --local-bits 21 --owner 00000001 --basis 9876543 --verify qft 2>&1 | grep "node=0"
