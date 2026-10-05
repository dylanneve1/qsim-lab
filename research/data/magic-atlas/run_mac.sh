#!/bin/bash
# Mac timing chunks for research/simulability/magic-atlas.md. Usage: run_mac.sh CHUNK OUT.jsonl
# Takes the swarm bench lock for the chunk (keep each chunk < 3 min).
B=${B:-$HOME/qsim-magic-atlas-target/release/examples/magic_atlas}
CHUNK=$1; OUT=$2
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/qsim-mac-bench.lock' EXIT
echo "{\"chunk\":\"$CHUNK\",\"load\":\"$(sysctl -n vm.loadavg)\",\"date\":\"$(date -u +%FT%TZ)\"}" >> "$OUT"
t1() { RAYON_NUM_THREADS=1 perl -e "alarm shift; exec @ARGV" 60 "$B" "$@" >> "$OUT" 2>/dev/null || echo "{\"fail\":\"$*\"}" >> "$OUT"; }
case $CHUNK in
law1)
  for rep in 1 2 3; do
    for t in 12 14 16 18 20; do t1 time cstate "qpe:t=$t,s=64,kind=stab" 1 30; done
  done ;;
law2)
  for rep in 1 2 3; do
    for t in 16 20 22 24; do t1 time cstate "rct:n=64,L=32,t=$t" 1 30; t1 time factored "rct:n=64,L=32,t=$t" 1 30; done
  done ;;
eng*)
  case $CHUNK in
    eng1) SPECS="qft:n=24,in=basis qft:n=24,in=graph cuccaro:bits=10,in=basis cuccaro:bits=10,in=plusa gidney:bits=8,in=basis";;
    eng2) SPECS="draper:bits=12,in=basis shorwin:nbits=4,w=2,in=one grover:n=12,it=4 ising:n=24,steps=4,dt=0.1 qaoa:n=24,p=2,graph=reg3";;
    eng3) SPECS="hea:n=24,layers=2 qpe:t=10,s=14,kind=stab walk:m=8,steps=4 hhl:t=8,m=7 rct:n=24,L=12,t=24";;
  esac
  for rep in 1 2 3; do
    for s in $SPECS; do
      for e in sv cstate factored recycled; do t1 time $e "$s" 1 26; done
    done
  done ;;
demo1)
  for rep in 1 2 3; do
    perl -e "alarm shift; exec @ARGV" 120 "$B" demo-shor 62 4 >> "$OUT"
    perl -e "alarm shift; exec @ARGV" 120 "$B" demo-qft 1024 >> "$OUT"
  done ;;
demo2)
  for rep in 1 2; do
    perl -e "alarm shift; exec @ARGV" 120 "$B" demo-qpe 23 256 >> "$OUT"
  done ;;
esac
echo "{\"chunk_end\":\"$CHUNK\",\"load\":\"$(sysctl -n vm.loadavg)\",\"date\":\"$(date -u +%FT%TZ)\"}" >> "$OUT"
