#!/bin/zsh
# Interleaved A/B of the slice-evaluator tiers: A = QSIM_NO_AVX512=1 (AVX2 tier), B = default
# (AVX-512 tier on this CPU). usage (inside flock /dev/shm/qsim/bench.lock):
#   ab.sh <reps> <label> <cmd...>
reps=$1; label=$2; shift 2
here=${0:A:h}
echo "#### $label  reps=$reps  $(date '+%F %T %Z')"
for r in $(seq 1 $reps); do
  for arm in A B; do
    echo "-- $label arm $arm rep $r"
    if [[ $arm == A ]]; then
      QSIM_NO_AVX512=1 QSIM_SLICE_PROFILE=1 $here/timed.sh "$@" 2>&1 | grep -E "^(==|before|after|wall)|measured|factor|EH run|Shor|time |sliced profile|qubits="
    else
      QSIM_SLICE_PROFILE=1 $here/timed.sh "$@" 2>&1 | grep -E "^(==|before|after|wall)|measured|factor|EH run|Shor|time |sliced profile|qubits="
    fi
  done
done
