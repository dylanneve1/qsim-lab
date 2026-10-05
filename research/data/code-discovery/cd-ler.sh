#!/bin/bash
# LER campaign: k=12 family, Z basis, uniform circuit noise, BP+OSD-CS(10)
B=~/qsim-wt-logs/bb_search_bin2
T=1
run() { echo "# $*" >&2; nice -n 10 $B ler "$@" >> cd-ler.jsonl; }
for p in 0.003 0.002; do
  run 6 6 "x^3+y+y^2" "y^3+x+x^2" ibm 6 $p 20480 11 $T 10
  run 28 2 "1+x+x^3y" "1+x^2+x^20" -034521/452103- 8 $p 20480 12 $T 10
  run 28 2 "1+x+x^3y" "1+x^2+x^20" ibm 8 $p 20480 13 $T 10
done
run 12 6 "x^3+y+y^2" "y^3+x+x^2" ibm 12 0.003 16384 14 $T 10
