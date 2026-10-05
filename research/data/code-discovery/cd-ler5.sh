#!/bin/bash
B=~/qsim-wt-logs/bb_search_bin2
while pgrep -f cd-ler4.sh > /dev/null; do sleep 10; done
run() { echo "# $*" >&2; nice -n 10 $B ler "$@" >> cd-ler.jsonl; }
run 28 1 "1+x+x^3" "1+x^5+x^11" -034521/452013- 8 0.0015 81920 51 1 40
run 28 2 "1+x+x^3y" "1+x^2+x^20" -034521/452103- 8 0.0015 81920 52 1 40
run 28 1 "1+x+x^3" "1+x^5+x^11" -034521/452013- 8 0.002 40960 53 1 100
