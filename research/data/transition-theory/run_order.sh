#!/bin/bash
# Restartable runner: executes jobs.txt lines in the order given by order.txt
# (ids = line numbers = part file names), skipping finished parts; then batch 2.
D=~/qsim-tt-data
B=$D/tt; mkdir -p $D/parts
rm -f $D/done   # a stale flag from an earlier run would stop the SIGSTOP watcher (happened once, see caveats)
export RAYON_NUM_THREADS=1
( while true; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "qsim-tt-data/tt "; else pkill -CONT -f "qsim-tt-data/tt "; fi
    [ -f $D/done ] && { pkill -CONT -f "qsim-tt-data/tt "; exit 0; }
    sleep 3
  done ) &
while read -r i; do [ -s $D/parts/$i.csv ] || echo "$i $(sed -n "${i}p" $D/jobs.txt)"; done < $D/order.txt | \
  xargs -P 2 -L 1 bash -c 'id=$0; nice -n 5 '"$B"' "$@" > '"$D"'/parts/$id.tmp 2>/dev/null && mv '"$D"'/parts/$id.tmp '"$D"'/parts/$id.csv'
touch $D/done
