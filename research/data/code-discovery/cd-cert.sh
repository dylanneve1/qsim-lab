#!/bin/bash
# exact distances of literature codes our search left unresolved / lit upper bounds
B=~/qsim-wt-logs/bb_search_bin2
L=30000000000
run() { echo "# $*" >&2; nice -n 10 $B params "$@" 64 $L >> cd-cert.jsonl; }
run 87 1 "1+x^2+x^13" "1+x^29+x^70"     # GT [[174,4,18]]
run 91 1 "1+x^9+x^13" "1+x+x^38"        # GB [[182,6,18]]
run 102 1 "1+x^16+x^35" "1+x+x^11"      # GB [[204,4,20]]
run 108 1 "1+x^14+x^22" "1+x+x^20"      # GB [[216,4,20]]
run 111 1 "1+x+x^68" "1+x^5+x^37"       # GT [[222,4,20]]
run 112 1 "1+x^3+x^22" "1+x+x^31"       # GB [[224,6,20]]
run 114 1 "1+x^22+x^47" "1+x^2+x^43"    # GT [[228,4,20]]
run 133 1 "1+x^5+x^109" "1+x^19+x^25"   # GT [[266,6,<=22]]
run 140 1 "1+x^16+x^31" "1+x^5+x^22"    # GT [[280,6,<=22]]
run 150 1 "1+x^49+x^93" "1+x^2+x^9"     # GT [[300,8,<=22]]
run 117 1 "1+x^13+x^29" "1+x+x^20"      # GB [[234,4,<=22]]
run 132 1 "1+x^13+x^20" "1+x+x^17"      # GB [[264,4,<=22]]
run 144 1 "1+x^20+x^25" "1+x+x^14"      # GB [[288,4,<=24]]
