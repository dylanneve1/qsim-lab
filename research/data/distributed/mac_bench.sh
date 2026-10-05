#!/bin/bash
# Mac-side baselines, interleaved, each (workload, n) group in one locked chunk:
#   ram  : single-node in-RAM blocked executor (dist_sv ram, 1 rep per round)
#   ooc  : out-of-core windowed scheduler, c=20 k=4, overlapped I/O, SSD scratch
#   loop : distributed, 2 processes over loopback TCP, G=3, node 1 owns 1/8
# Output CSV-ish lines in ~/qsim-wt-logs/dist-mac-bench.log
B=~/qsim-dist-target/release/examples/dist_sv
O=~/qsim-dist-target/release/examples/ooc_bench
export OOC_SCRATCH=~/qsim-dist-scratch
mkdir -p $OOC_SCRATCH
LOG=~/qsim-wt-logs/dist-mac-bench.log
for round in 1 2 3; do
  for spec in "qft 26" "brick 26" "qft 28" "brick 28"; do
    set -- $spec; wl=$1; n=$2
    until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
    echo "# round=$round $wl n=$n start=$(date +%T) load=$(sysctl -n vm.loadavg)" >> $LOG
    order="ram ooc loop"; [ $round = 2 ] && order="loop ram ooc"; [ $round = 3 ] && order="ooc loop ram"
    for m in $order; do
      case $m in
        ram)  RAYON_NUM_THREADS=8 $B ram --workload $wl --n $n --prec f32 --reps 1 >> $LOG 2>&1 ;;
        ooc)  echo -n "ooc " >> $LOG; RAYON_NUM_THREADS=8 $O ooc $wl $n f32 window 20 4 1 0 >> $LOG 2>&1 ;;
        loop) ~/qsim-dist-run/loop.sh 4 --workload $wl --n $n --prec f32 --local-bits $((n-3)) --owner 00000001 2>&1 | grep "node=0" >> $LOG ;;
      esac
    done
    rm -f $OOC_SCRATCH/*
    rmdir /tmp/qsim-mac-bench.lock
    sleep 65
  done
done
echo "# done $(date +%T)" >> $LOG
