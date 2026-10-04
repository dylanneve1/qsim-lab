#!/bin/bash
# Mac campaign: 2 single-threaded workers (MAC CORE BUDGET), no bench lock,
# SIGSTOP while a peer holds /tmp/qsim-mac-bench.lock.
D=~/qsim-tt-data
B=$D/tt; mkdir -p $D/parts
export RAYON_NUM_THREADS=1
( while true; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "qsim-tt-data/tt "; else pkill -CONT -f "qsim-tt-data/tt "; fi
    [ -f $D/done ] && { pkill -CONT -f "qsim-tt-data/tt "; exit 0; }
    sleep 3
  done ) &
i=0
while read -r line; do i=$((i+1)); [ -s $D/parts/$i.csv ] || echo "$i $line"; done < $D/jobs.txt | \
  xargs -P 2 -L 1 bash -c 'id=$0; nice -n 5 '"$B"' "$@" > '"$D"'/parts/$id.tmp 2>/dev/null && mv '"$D"'/parts/$id.tmp '"$D"'/parts/$id.csv'
touch $D/done
