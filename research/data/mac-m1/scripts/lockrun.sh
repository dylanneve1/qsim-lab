#!/bin/bash
# lockrun.sh TAG cmd... : run one command under the Mac bench lock, append output to data/TAG.out
TAG=$1; shift; OUT=~/qsim-l1r4-bin/data
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/qsim-mac-bench.lock 2>/dev/null' EXIT
echo "# $(date +%T) $* load=[$(uptime | sed 's/.*averages: //')] contam=$(pgrep -q 'rustc|cargo' && echo YES || echo no)" >> $OUT/$TAG.out
"$@" >> $OUT/$TAG.out 2>&1
