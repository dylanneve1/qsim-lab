#!/bin/bash
# Audit timing on the VPS: each job starts only when the 1-min load is < 4 (polls every 20 s);
# the load at the end of each job is recorded in the JSON (load1).
cd /tmp/qsim-wt/sampler-audit
OUT=research/data/fast-sampler-audit/timing_vps.jsonl
BIN=/tmp/qsim-wt/sampler-audit-target/release/examples/stim_compare
STIM=/tmp/qsim-wt/sa-stim/build/out/stim
PY=/tmp/fw/bin/python
export RAYON_NUM_THREADS=1
waitload() { until awk '{exit !($1 < 4.0)}' /proc/loadavg; do sleep 20; done; }
job() { waitload; echo "{\"note\":\"$* load1_at_start=$(cut -d' ' -f1 /proc/loadavg)\"}" >> $OUT; "$@" >> $OUT 2>&1; }
for spec in "7 1000000" "15 128000"; do
  set -- $spec
  job env STIM_CLI=$STIM $PY research/data/fast-sampler/timing.py $BIN /tmp/qsim-wt/sa-work/tim $1 0.001 $2 3 0
done
job $PY research/data/fast-sampler-audit/e2e.py $BIN $STIM /tmp/qsim-wt/sa-work/tim 7 0.001 3 10240,102400,1024000
job $PY research/data/fast-sampler-audit/e2e.py $BIN $STIM /tmp/qsim-wt/sa-work/tim 15 0.001 3 10240,102400,1024000
echo '{"note":"DONE"}' >> $OUT
