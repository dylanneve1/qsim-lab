#!/bin/zsh
export RAYON_NUM_THREADS=8
L=~/qsim-metal-data/lr.sh
$L clean qft 28 3 0 cpu gpu; sleep 30
$L clean brick 24 3 20 cpu gpu; sleep 30
$L clean brick 26 3 20 cpu gpu; sleep 30
$L clean qft 20,22 5 0 cpu gpu; sleep 30
$L clean brick 20,22 5 20 cpu gpu; sleep 30
QSIM_BENCH_BASIS=5a5a5a5 $L clean qft 28 1 0 cpu gpu; sleep 30
$L clean bw 28 6
echo DONE >> ~/qsim-metal-data/clean.out
