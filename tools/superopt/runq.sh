#!/bin/bash
# Run a job list (one shell command per line) with <= $P workers; pause all
# workers (SIGSTOP) while anyone holds the Mac bench lock (MAC SHARING RULE).
# usage: runq.sh jobs.txt out.jsonl [P]
JOBS=$1; OUT=$2; P=${3:-3}
TAG="superopt-runq-$$"
( cat "$JOBS" | xargs -P "$P" -I{} bash -c "export SO_TAG=$TAG; {} >> $OUT 2>>${OUT%.jsonl}.err" ) &
XP=$!
paused=0
while kill -0 $XP 2>/dev/null; do
  pids=$(pgrep -f "blocks.py|peep.py" | tr '\n' ' ')
  if [ -d /tmp/qsim-mac-bench.lock ]; then
    [ $paused = 0 ] && [ -n "$pids" ] && kill -STOP $pids 2>/dev/null && paused=1
  else
    [ $paused = 1 ] && [ -n "$pids" ] && kill -CONT $pids 2>/dev/null; paused=0
  fi
  sleep 3
done
echo "runq done" >> ${OUT%.jsonl}.err
