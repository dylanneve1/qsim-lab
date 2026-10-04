#!/bin/bash
# campaign runner: <=4 workers, SIGSTOP while the bench lock is held
B=~/qsim-mt-target/release/examples/magic_transition
D=~/qsim-mt-data; mkdir -p $D/parts
export RAYON_NUM_THREADS=1
( while true; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "examples/magic_transition scan"; else pkill -CONT -f "examples/magic_transition scan"; fi
    [ -f $D/done ] && { pkill -CONT -f "examples/magic_transition scan"; exit 0; }
    sleep 3
  done ) &
i=0
while read -r line; do
  i=$((i+1)); echo "$i $line"
done < $D/jobs.txt | xargs -P 4 -L 1 bash -c 'id=$0; shift; '"$B"' scan "$@" > '"$D"'/parts/$id.csv 2>/dev/null'
touch $D/done
