#!/bin/zsh
# load-gated cells: wait (<= 15 min) for 1-min load < 6, then a locked run
export RAYON_NUM_THREADS=8
L=~/qsim-metal-data/lr.sh
gate() { for i in {1..180}; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); (( l < 6 )) && return; sleep 5; done; echo "# gate timeout load=$l" >> ~/qsim-metal-data/clean.out; }
gate; $L clean qft 24,26 5 0 cpu gpu; sleep 30
gate; $L clean qft 28 3 0 cpu gpu; sleep 30
gate; $L clean brick 24 3 20 cpu gpu; sleep 30
gate; $L clean brick 26 3 20 cpu gpu; sleep 30
gate; $L clean bw 28 6; sleep 30
echo DONE >> ~/qsim-metal-data/clean.out
