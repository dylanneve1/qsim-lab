#!/bin/bash
# Second batch (jobs2.txt -> parts2/), starts when batch 1 is done. Same rules as run.sh.
D=~/qsim-tt-data
until [ -f $D/done ]; do sleep 20; done
B=$D/tt; mkdir -p $D/parts2
export RAYON_NUM_THREADS=1
( while true; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "qsim-tt-data/tt "; else pkill -CONT -f "qsim-tt-data/tt "; fi
    [ -f $D/done2 ] && { pkill -CONT -f "qsim-tt-data/tt "; exit 0; }
    sleep 3
  done ) &
i=0
while read -r line; do i=$((i+1)); [ -s $D/parts2/$i.csv ] || echo "$i $line"; done < $D/jobs2.txt | \
  xargs -P 2 -L 1 bash -c 'id=$0; nice -n 5 '"$B"' "$@" > '"$D"'/parts2/$id.tmp 2>/dev/null && mv '"$D"'/parts2/$id.tmp '"$D"'/parts2/$id.csv'
touch $D/done2
