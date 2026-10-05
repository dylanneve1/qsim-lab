#!/bin/bash
# Third SPD campaign (queued after campaign2): noise-aware runs and the larger
# heavy-hex lattices. Same pause / memory rules.
cd ~/qsim-spoof
until grep -q DONE ~/qsim-wt-logs/spoof-campaign2.err 2>/dev/null; do sleep 30; done
OUT=${OUT:-~/qsim-wt-logs/spoof-campaign3.jsonl}
LOG=~/qsim-wt-logs/spoof-campaign3.err
BIN=./target/release/examples/spoof_utility
export RAYON_NUM_THREADS=2
memok() { vm_stat | awk '/Pages free|Pages inactive/{s+=$NF} END{exit !(s*16384/1e9 >= 4)}'; }
run() {
  until memok; do sleep 20; done
  local t0=$(date +%s)
  nice -n 5 $BIN "$@" --mem-gb 2.2 >> $OUT 2>>$LOG &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then
      kill -STOP $pid; while [ -d /tmp/qsim-mac-bench.lock ]; do sleep 5; done; kill -CONT $pid
    fi
    sleep 3
  done
  echo "$(date +%H:%M:%S) [$(( $(date +%s) - t0 ))s] $*" >> $LOG
}
TH="0.0,0.1,0.2,0.3,0.5,0.7,0.8,1.0,1.2,1.3,1.4,1.5,1.5707"
TH4="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.5707"
# D. noise-aware: p fitted once at theta=0 from the unmitigated data
#    (5 steps: M_z(0)=0.8345 -> p=0.0266; 20 steps: Z62(0)=0.5689 -> p=0.0209),
#    noise amplification G = 1, 1.2, 1.6 as in the experiment's ZNE.
for G in 1 1.2 1.6; do
  p5=$(python3 -c "print(0.0266*$G)"); p20=$(python3 -c "print(0.0209*$G)")
  run 3a --thetas $TH --delta 1e-7 --depol $p5
  run 3b --thetas $TH --delta 1e-6 --depol $p5
  run 3c --thetas 0.0,0.25,0.5,0.75,1.0,1.1,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.5707 --delta 1e-5 --depol $p5 --stream 2
  run 4b --thetas $TH4 --delta 1e-5 --depol $p20
done
# E. larger lattices: M_z at 5 steps, bulk Z at 5 and 20 steps
for L in 433 1121; do
  run mz --lattice $L --thetas $TH --delta 1e-5,1e-7
done
run z215 --lattice 433 --steps 5 --thetas $TH --delta 1e-6
run z559 --lattice 1121 --steps 5 --thetas $TH --delta 1e-6
run z62 --lattice 127 --steps 5 --thetas $TH --delta 1e-6
run z215 --lattice 433 --thetas $TH4 --delta 1e-4,3e-5
run z559 --lattice 1121 --thetas $TH4 --delta 1e-4,3e-5
echo DONE >> $LOG
