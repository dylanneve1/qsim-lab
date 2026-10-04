#!/bin/bash
# lr.sh TAG args... : one locked run of metal_bench, output appended to ~/qsim-metal-data/TAG.out
TAG=$1; shift
B=~/qsim-metal-target/release/examples/metal_bench
OUT=~/qsim-metal-data/$TAG.out
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/qsim-mac-bench.lock 2>/dev/null' EXIT
echo "# $(date +%H:%M:%S) args=[$*] thr=${RAYON_NUM_THREADS:-?} load=[$(uptime | sed 's/.*averages: //')] top=[$(ps -Ao pcpu,comm -r | sed -n 2,4p | sed 's|/.*/||' | tr '\n' ';')] contam=$(pgrep -q 'rustc|cargo' && echo YES || echo no) free=$(vm_stat | awk '/Pages free/{f=$3}/Pages inactive/{i=$3}END{printf "%.1fGB", (f+i)*16384/1e9}')" | tee -a $OUT
$B "$@" 2>&1 | tee -a $OUT
rmdir /tmp/qsim-mac-bench.lock; trap - EXIT
