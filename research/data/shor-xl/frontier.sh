#!/bin/zsh
# Generic-N frontier (PREREG.md B): EH with the odd-order base of seed 1, config B (w_e=1, w_m=4),
# f32, up to 3 runs on the same base. usage (inside flock): frontier.sh <ge_shor binary> <bits...>
here=${0:A:h}; G=$1; shift
for b in "$@"; do
  N=$(awk -v b=$b '$1==b{print $2}' $here/generic_instances.txt)
  QSIM_GE_MAX_GB=9 $here/timed.sh $G run $N 1 1 4 lookups eh-odd f32 3 2>&1 | grep -v "ge window"
done
