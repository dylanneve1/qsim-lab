#!/bin/bash
# The 10^6-shot equivalence cells of research/qec/sampler-x.md §4 (not timing runs; nice'd).
# usage: run_equivalence.sh <stim_compare> <work dir> <circuit dir from make_circuits.py>
set -e
cd "$(dirname "$0")"
export QSIM_STIM_COMPARE=$1 WORK=$2
CD=$3
mkdir -p "$WORK"
PY=${PY:-python3}
# compiled path with hit tables, both directions, d = 3, 7, 11, 15, p = 0.3% and 0.1%
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 1 on equivalence_x.jsonl AB
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.001 1 on equivalence_x.jsonl AB
# other compiled paths: 8 threads, 4 threads without tables, the AVX-512 hit kernel
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 8 on equivalence_x.jsonl B
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 4 off equivalence_x.jsonl B
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.001 2 simd equivalence_x.jsonl B
# the frame sampler (scalar and AVX-512 builds)
nice -n 15 $PY equivalence_x.py 1e6 3,7,11,15 0.003 1 frames equivalence_x.jsonl AB
nice -n 15 $PY equivalence_x.py 1e6 3,7,11 0.001 1 frames-simd equivalence_x.jsonl B
# power: perturbed circuits must be rejected
nice -n 15 $PY negative_control_x.py 8 on > negative_control_x.jsonl
nice -n 15 $PY negative_control_x.py 1 frames > negative_control_x_frames.jsonl
# DEM support (T0) of the new compiler at every distance
nice -n 15 $PY support_x.py "$1" "$CD" 3,5,7,11,15,21,25 0.001,0.003 > support_x.jsonl
# other Stim circuits (colour, repetition, unrotated, rotated X), incl. joint 4-detector histograms
nice -n 15 $PY equivalence_other_x.py "$1" "$2" 1000000 equivalence_other_x.jsonl 8 auto
nice -n 15 $PY equivalence_other_x.py "$1" "$2" 1000000 equivalence_other_x.jsonl 1 frames
nice -n 15 $PY equivalence_other_x.py "$1" "$2" 1000000 equivalence_other_x_negative.jsonl 1 frames perturb
