#!/bin/bash
# Audit timing on the Mac (M1 Pro), under the swarm bench lock, one lock hold per job (< 3 min),
# >= 30 s between holds. Waits up to 120 s for 1-min load < 4 inside the lock and records it.
cd ~/qsim-sa
OUT=~/qsim-wt-logs/sa-timing-mac.jsonl
BIN=~/qsim-sa-target/release/examples/stim_compare
PY=~/qsim-bench-venv/bin/python
STIM=~/qsim-bench-venv/bin/stim
export RAYON_NUM_THREADS=1
job() {
  until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
  for i in $(seq 1 24); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk "BEGIN{exit !($l < 4.0)}" && break
    sleep 5
  done
  echo "{\"note\":\"$* load1_at_start=$(sysctl -n vm.loadavg | awk '{print $2}')\"}" >> $OUT
  "$@" >> $OUT 2>&1
  rmdir /tmp/qsim-mac-bench.lock
  sleep 35
}
for spec in "7 1000000" "15 128000"; do
  set -- $spec
  job env STIM_CLI=$STIM $PY research/data/fast-sampler/timing.py $BIN ~/qsim-wt-logs/sa-stim $1 0.001 $2 3 0
done
job $PY research/data/fast-sampler-audit/e2e.py $BIN $STIM ~/qsim-wt-logs/sa-stim 7 0.001 3 10240,102400,1024000
job $PY research/data/fast-sampler-audit/e2e.py $BIN $STIM ~/qsim-wt-logs/sa-stim 15 0.001 3 10240,102400
job $PY research/data/fast-sampler-audit/e2e.py $BIN $STIM ~/qsim-wt-logs/sa-stim 15 0.001 2 1024000
echo '{"note":"DONE"}' >> $OUT
