#!/bin/bash
# kahypar unsliced search (budget $2 s) -> pickled tree -> lean slicing down to width $3
. /tmp/doped-zx/venv/bin/activate; cd /tmp/doped-zx
D=$1; S=$2; W=$3
OMP_NUM_THREADS=1 prlimit --as=1100000000 nice timeout $((S+1500)) python search.py $D raw none $S kahypar,greedy > pipe_s$D.log 2>&1
cp tree_D${D}_raw_none.pkl tree_D${D}_long.pkl
OMP_NUM_THREADS=1 prlimit --as=1100000000 nice timeout 2400 python lean.py $D $W tree_D${D}_long.pkl > pipe_l$D.log 2>&1
