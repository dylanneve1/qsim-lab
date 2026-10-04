#!/bin/bash
# all equivalence cells of research/qec/fast-sampler.md (VPS, nice'd)
cd "$(dirname "$0")"
export QSIM_STIM_COMPARE=${QSIM_STIM_COMPARE:-/tmp/qsim-wt/fast-sampler-target/release/examples/stim_compare}
export WORK=${WORK:-/tmp/qsim-wt/fs-data/eq}
PY=${PY:-/tmp/fw/bin/python}
nice -n 15 $PY equivalence_fast.py 1000000 3,7,11,15 0.003 equivalence_fast_p0.003.jsonl AB
nice -n 15 $PY equivalence_fast.py 1000000 3,7,11,15 0.001 equivalence_fast_p0.001.jsonl AB
QSIM_FAST_RNG=xo nice -n 15 $PY equivalence_fast.py 1000000 3,7,11 0.003 equivalence_fast_xoshiro_p0.003.jsonl B
nice -n 15 $PY negative_control_fast.py > negative_control_fast.jsonl
