#!/bin/bash
# 10^6-shot equivalence cells of research/qec/sampler-x.md §4 (not timing runs; nice'd).
# usage: run_equivalence.sh <stim_compare> <work dir>
set -e
cd "$(dirname "$0")"
export QSIM_STIM_COMPARE=$1 WORK=$2
mkdir -p "$WORK"
PY=${PY:-python3}
# both directions, d = 3, 7, 11, 15, single thread, tables chosen automatically (= the CLI default)
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.001 1 auto equivalence_x.jsonl AB
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 1 auto equivalence_x.jsonl AB
# the other sampling paths: 8 threads with tables, 4 threads without, the AVX-512 kernel (Stim's circuit)
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 8 on equivalence_x.jsonl B
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 4 off equivalence_x.jsonl B
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.001 2 simd equivalence_x.jsonl B
# other Stim circuits (colour, repetition, unrotated, rotated X) incl. joint 4-detector histograms
nice -n 15 $PY equivalence_other_x.py "$1" "$2" 1000000 equivalence_other_x.jsonl 8 auto
# power: perturbed circuits must be rejected
nice -n 15 $PY negative_control_x.py 8 on > negative_control_x.jsonl
nice -n 15 $PY equivalence_other_x.py "$1" "$2" 1000000 equivalence_other_x_negative.jsonl 8 auto perturb
