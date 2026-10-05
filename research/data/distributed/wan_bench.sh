#!/bin/bash
# Mac (node 0, 7/8 of the state) <-> VPS (node 1, 1/8) over the SSH session.
# One run per locked chunk (each < 150 s), 65 s gap between chunks.
W=~/qsim-dist-run/wan.sh
LOG=~/qsim-wt-logs/dist-wan-bench.log
run() { # <vps-mem> <args...>
  local mem=$1; shift
  until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
  echo "# $(date +%T) mac_load=$(sysctl -n vm.loadavg) vps_load=$(ssh -o BatchMode=yes claudius cat /proc/loadavg | cut -d' ' -f1-3) args=$*" >> $LOG
  $W run 6 $mem 2 "$@" 2>&1 | grep "^dist" >> $LOG
  rmdir /tmp/qsim-mac-bench.lock
  sleep 65
}
for rep in 1 2; do
  for n in 24 26 28; do
    for wl in qft brick; do
      run 700M --workload $wl --n $n --prec f32 --local-bits $((n-3)) --owner 00000001
    done
  done
done
# Largest: n=29 f32 (4 GiB total, 512 MiB on the VPS), QFT checked analytically.
run 1000M --workload qft --n 29 --prec f32 --local-bits 26 --owner 00000001 --basis 123456789 --verify qft
run 1000M --workload brick --n 29 --prec f32 --local-bits 26 --owner 00000001
echo "# done $(date +%T)" >> $LOG
