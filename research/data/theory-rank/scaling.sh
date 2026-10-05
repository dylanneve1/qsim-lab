#!/bin/bash
# Scaling series for the low-rank families. Usage: scaling.sh BIN > scaling.jsonl
B=$1
SPECS=(
 "grover:n=8,it=2" "grover:n=16,it=2" "grover:n=24,it=2" "grover:n=48,it=1"
 "shorwin:nbits=4,w=2,in=half" "shorwin:nbits=5,w=2,in=half" "shorwin:nbits=6,w=2,in=half" "shorwin:nbits=7,w=2,in=half"
 "shorwin:nbits=8,w=4,in=one" "shorwin:nbits=16,w=4,in=one"
 "draper:bits=3,in=basis" "draper:bits=4,in=basis" "draper:bits=6,in=basis" "draper:bits=7,in=basis" "draper:bits=8,in=basis"
 "draper:bits=8,in=plusab" "draper:bits=32,in=plusab"
 "walk:m=3,steps=2" "walk:m=4,steps=2" "walk:m=6,steps=2" "walk:m=8,steps=2" "walk:m=8,steps=4" "walk:m=16,steps=2"
 "gidney:bits=8,in=basis" "gidney:bits=32,in=basis"
 "cuccaro:bits=3,in=plusa" "cuccaro:bits=4,in=plusa" "cuccaro:bits=6,in=plusa" "cuccaro:bits=7,in=plusa" "cuccaro:bits=8,in=plusa"
 "qpe:t=2,s=6,kind=stab" "qpe:t=3,s=6,kind=stab" "qpe:t=4,s=6,kind=stab" "qpe:t=5,s=6,kind=stab" "qpe:t=6,s=8,kind=stab"
 "rct:n=32,L=8,t=4" "rct:n=32,L=8,t=8" "rct:n=32,L=8,t=12" "rct:n=32,L=8,t=16"
 "qft:n=6,in=basis" "qft:n=8,in=basis" "qft:n=10,in=basis"
)
for s in "${SPECS[@]}"; do
  timeout 150 nice -n 15 $B profile "$s" 1 1024 || echo "{\"spec\":\"$s\",\"error\":\"timeout/fail\"}"
done
