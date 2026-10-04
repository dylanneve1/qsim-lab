#!/bin/bash
# Stim undetectable-logical search on full 9-round d=9 circuits, both bases, K-F and new.
# Memory watchdog: kill a search whose RSS exceeds 9 GB (16 GB laptop).
cd ~/qsim-cg
PY=~/qsim-bench-venv/bin/python
EV=${1:-4}; DEG=${2:-10}
for f in kf_d9_r9_z new_d9_r9_z kf_d9_r9_x new_d9_r9_x; do
  $PY stim_check.py $f.stim $EV $DEG >> stim_r9.txt 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do
    rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); rss=${rss:-0}
    [ "$rss" -gt "$peak" ] && peak=$rss
    if [ "$rss" -gt 9000000 ]; then kill $pid; echo "$f: KILLED at RSS ${rss} kB (ev<=$EV deg<=$DEG)" >> stim_r9.txt; fi
    sleep 5
  done
  echo "$f: peak RSS ${peak} kB" >> stim_r9.txt
done
echo DONE >> stim_r9.txt
