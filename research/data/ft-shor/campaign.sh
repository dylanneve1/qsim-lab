#!/bin/sh
# FT-Shor campaign (Mac, 2 workers). Usage: campaign.sh <jobfile> <outfile>
# Each job line: arguments to ft_shor. Workers pause (SIGSTOP) while the
# swarm bench lock is held.
B=${B:-$HOME/qsim-ft-target/release/examples/ft_shor}
JOBS=$1; OUT=$2
run() { $B $@ >> "$OUT"; }
export -f run 2>/dev/null
cat "$JOBS" | xargs -P 2 -I{} sh -c "$B {} >> $OUT" &
XP=$!
while kill -0 $XP 2>/dev/null; do
  if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "examples/ft_shor"; else pkill -CONT -f "examples/ft_shor"; fi
  sleep 5
done
pkill -CONT -f "examples/ft_shor"
echo DONE >> "$OUT.done"
