#!/bin/sh
# FT-Shor campaign (Mac, 2 workers). Usage: campaign.sh <jobfile> <outfile>
# Each job line: an example binary name + its arguments. Workers pause (SIGSTOP) while the
# swarm bench lock is held.
BIN=${BIN:-$HOME/qsim-ft-target/release/examples}
JOBS=$1; OUT=$2
cat "$JOBS" | xargs -P 2 -I{} sh -c "$BIN/{} >> $OUT" &
XP=$!
while kill -0 $XP 2>/dev/null; do
  if [ -d /tmp/qsim-mac-bench.lock ]; then pkill -STOP -f "examples/ft_"; else pkill -CONT -f "examples/ft_"; fi
  sleep 5
done
pkill -CONT -f "examples/ft_"
echo DONE >> "$OUT.done"
