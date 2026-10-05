#!/bin/bash
B=~/qsim-wt-logs/bb_search_bin2
run() { echo "# $*" >&2; nice -n 10 $B ler "$@" >> cd-ler.jsonl; }
run 28 1 "1+x+x^3" "1+x^5+x^11" -034521/452013- 8 0.002 40960 41 1 40
run 28 2 "1+x+x^3y" "1+x^2+x^20" -034521/452103- 8 0.002 40960 42 1 40
