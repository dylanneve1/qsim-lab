#!/bin/bash
# Per-family branching-rank table. Usage: families.sh BIN CAP > families.jsonl
B=$1; CAP=${2:-4096}
SPECS=(
 "qft:n=12,in=basis" "qft:n=12,in=plus" "qft:n=12,in=graph" "qft:n=24,in=basis"
 "cuccaro:bits=5,in=basis" "cuccaro:bits=5,in=plusa" "cuccaro:bits=5,in=plusab" "cuccaro:bits=64,in=basis" "cuccaro:bits=16,in=plusa"
 "gidney:bits=3,in=basis" "gidney:bits=3,in=plusa" "gidney:bits=64,in=basis" "gidney:bits=16,in=plusa"
 "draper:bits=5,in=basis" "draper:bits=5,in=plusa" "draper:bits=5,in=plusab" "draper:bits=16,in=basis"
 "shorwin:nbits=4,w=2,in=one" "shorwin:nbits=4,w=2,in=half" "shorwin:nbits=32,w=4,in=one" "shorwin:nbits=62,w=4,in=one" "shorwin:nbits=8,w=2,in=half"
 "shor:nbits=3,w=2" "shor:nbits=4,w=2"
 "grover:n=6,it=2" "grover:n=6,it=6" "grover:n=32,it=2" "grover:n=64,it=1"
 "ising:n=12,steps=1" "ising:n=12,steps=2" "ising:n=12,steps=4" "ising:n=24,steps=1"
 "ising:n=12,steps=4,dt=0.7853981633974483,J=1,h=1"
 "heis:n=12,steps=2" "qaoa:n=12,p=1" "qaoa:n=12,p=2" "hea:n=12,layers=1"
 "qpe:t=6,s=6,kind=stab" "qpe:t=10,s=16,kind=stab" "qpe:t=3,s=6,kind=trotter"
 "walk:m=4,steps=2" "hhl:t=4,m=2"
 "rct:n=12,L=8,t=4" "rct:n=12,L=8,t=8" "rct:n=12,L=8,t=16" "rct:n=64,L=8,t=12"
)
for s in "${SPECS[@]}"; do
  timeout 150 nice -n 15 $B profile "$s" 1 $CAP || echo "{\"spec\":\"$s\",\"error\":\"timeout/fail\"}"
done
