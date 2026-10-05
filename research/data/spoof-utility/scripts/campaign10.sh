#!/bin/bash
cd ~/qsim-spoof
until grep -q DONE ~/qsim-wt-logs/spoof-campaign9.err 2>/dev/null; do sleep 20; done
export RAYON_NUM_THREADS=2
for k in 12 14 16 20; do
  nice -n 5 timeout 280 ./target/release/examples/spoof_patch $k 0.6 20 1e-3,1e-4,3e-5,1e-5 1 20 >> ~/qsim-wt-logs/spoof-patch.jsonl 2>/dev/null
done
echo DONE >> ~/qsim-wt-logs/spoof-campaign10.err
