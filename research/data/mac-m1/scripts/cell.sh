#!/bin/bash
# one locked A/B cell: cell.sh OUTTAG WL N PREC THREADS REPS DEPTH "mainCfgs" "l1Cfgs"
# interleaved: each rep runs both binaries (order alternates), one run per config.
TAG=$1 WL=$2 N=$3 P=$4 T=$5 R=$6 D=$7 MC=$8 LC=$9
B=~/qsim-l1r4-bin; OUT=~/qsim-l1r4-bin/data; mkdir -p $OUT
CSV=$OUT/${TAG}.csv
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/qsim-mac-bench.lock 2>/dev/null' EXIT
L=$(uptime | sed 's/.*averages: //'); TOPP=$(ps -Ao pcpu,comm -r | sed -n 2p | awk '{print $1, $NF}' | xargs basename 2>/dev/null)
echo "# $(date +%H:%M:%S) $WL n=$N $P t=$T load=[$L] top=[$(ps -Ao pcpu,comm -r | sed -n 2p | sed 's|/.*/||')]" >> $OUT/${TAG}.log
for ((r=0; r<R; r++)); do
  if (( r % 2 == 0 )); then ORD="main l1"; else ORD="l1 main"; fi
  for w in $ORD; do
    if [ $w = main ]; then [ -n "$MC" ] || continue; BIN=${BM:-$B/bench_main}; CF=$MC; else [ -n "$LC" ] || continue; BIN=${BL:-$B/bench_l1}; CF=$LC; fi
    pgrep -q "rustc|cargo" && echo "CONTAM rustc during $WL $N $P t=$T rep=$r" >> $OUT/${TAG}.log; RAYON_NUM_THREADS=$T $BIN $WL $N $P 1 $D $CF | grep '^| [bq]' | while read line; do echo "$T $w $line" >> $OUT/${TAG}.raw; done
  done
done
rmdir /tmp/qsim-mac-bench.lock; trap - EXIT
