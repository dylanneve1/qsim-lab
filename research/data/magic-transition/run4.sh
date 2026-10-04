#!/bin/bash
D=~/qsim-mt-data
until [ -f ~/qsim-mt-data/done3 ]; do sleep 20; done
B=$D/bin2; mkdir -p $D/parts4
export RAYON_NUM_THREADS=1
( while true; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "qsim-mt-data/bin2"; else pkill -CONT -f "qsim-mt-data/bin2"; fi
    [ -f $D/done4 ] && { pkill -CONT -f "qsim-mt-data/bin2"; exit 0; }
    sleep 3
  done ) &
i=0
while read -r line; do i=$((i+1)); echo "$i $line"; done < $D/jobs4.txt | xargs -P 4 -L 1 bash -c 'id=$0; '"$B"' "$@" > '"$D"'/parts4/$id.csv 2>/dev/null'
touch $D/done4
