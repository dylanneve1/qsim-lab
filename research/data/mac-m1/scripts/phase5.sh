#!/bin/bash
until grep -q PHASE4_DONE ~/qsim-l1r4-bin/data/sweepq.log 2>/dev/null; do sleep 10; done
R=~/qsim-l1r4-bin/lockrun.sh; PY=~/qsim-bench-venv/bin/python
cd ~/qsim-l1r4-bin; export T=8
$R sota2 $PY qsim_bench.py qft 22,24
$R sota2 $PY qsim_bench.py qft 26
echo PHASE5_DONE >> ~/qsim-l1r4-bin/data/sota2.out
