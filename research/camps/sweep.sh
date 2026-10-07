#!/bin/bash
# usage: sweep.sh OUT "modes" "ds" "chis"
out=$1; modes=$2; ds=$3; chis=$4
for chi in $chis; do for d in $ds; do for m in $modes; do
  if grep -q "\"mode\": \"$m\", \"d\": $d, \"chi\": $chi," $out 2>/dev/null; then continue; fi
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 prlimit --as=2500000000 .venv/bin/python run70.py $m $d $chi >> $out 2>>err.log
done; done; done
