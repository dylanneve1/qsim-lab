#!/bin/bash
# Audit §16 (r4-audit2): locked Mac timing chunks, each < ~3 min.
# usage: audit_bench.sh CHUNK   (CHUNK = atlas | sim | stim | noise)
cd ~/qsim-r4a2
B=~/qsim-r4a2/target/release/examples
OUT=~/qsim-wt-logs/r4a2-bench.log
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 2; done
for i in $(seq 1 30); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  awk "BEGIN{exit !($l < 4.0)}" && break
  sleep 3
done
echo "=== chunk $1 start $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)" >> $OUT
case $1 in
atlas)
  for r in 1 2 3; do /usr/bin/time -p $B/magic_atlas demo-shor 62 4 >> $OUT 2>&1; done
  for r in 1 2 3; do RAYON_NUM_THREADS=1 $B/magic_atlas time recycled 'shorwin:nbits=16,w=4,in=one' 1 8 >> $OUT 2>&1; done
  ;;
sim)
  for r in 1 2 3; do
    for e in sv cstate planx; do echo "qaoa24p2 $e" >> $OUT; RAYON_NUM_THREADS=1 $B/simulability run $e 'qaoa:n=24,p=2,deg=3,nn=0' 1 >> $OUT 2>&1; done
    for e in sv hsf planx; do echo "brick24D3 $e" >> $OUT; RAYON_NUM_THREADS=1 $B/simulability run $e 'brick:n=24,D=3,nn=0' 1 >> $OUT 2>&1; done
  done
  ;;
stim)
  mkdir -p ~/qsim-wt-logs/r4a2-stim
  RAYON_NUM_THREADS=1 ~/qsim-bench-venv/bin/python research/data/qec-r4/stim_timing.py $B/stim_compare ~/qsim-wt-logs/r4a2-stim 7 500000 3 >> $OUT 2>&1
  ;;
noise)
  QSIM_NOISE_KMIN=1 RAYON_NUM_THREADS=4 $B/shor_noise strat 899 689 4 phaseflip 1 4000 99 8192 > ~/qsim-wt-logs/r4a2-noise.csv 2>> $OUT
  ;;
esac
echo "=== chunk $1 end $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)" >> $OUT
rmdir /tmp/qsim-mac-bench.lock
