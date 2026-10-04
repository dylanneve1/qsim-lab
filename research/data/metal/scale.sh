#!/bin/zsh
export RAYON_NUM_THREADS=8
L=~/qsim-metal-data/lr.sh
SM="gpu:regs=0,tg=12,slots=6,thr=512,batch=3"
$L scale bw 26 10; sleep 30
$L scale qft 20,22,24,26 3 0 cpu gpu $SM naive; sleep 30
$L scale brick 20,22,24 3 20 cpu gpu $SM naive; sleep 30
$L scale qft 28 3 0 cpu gpu $SM; sleep 30
$L scale brick 26 3 20 cpu gpu $SM; sleep 30
$L scale brick 28 3 20 cpu gpu; sleep 30
echo "FREE before 29: $(vm_stat | awk '/Pages free/{f=$3}/Pages inactive/{i=$3}END{printf "%.1fGB", (f+i)*16384/1e9}')" >> ~/qsim-metal-data/scale.out
$L scale bw 29 4; sleep 30
$L scale qft 29 3 0 cpu gpu; sleep 30
echo DONE >> ~/qsim-metal-data/scale.out
