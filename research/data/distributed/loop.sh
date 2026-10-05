#!/bin/bash
# Two-process loopback run on the Mac: loop.sh <threads-per-node> <dist_sv run args...>
# Node 1 listens, node 0 connects; prints both result lines.
B=~/qsim-dist-target/release/examples/dist_sv
T=$1; shift
PORT=$((47200 + RANDOM % 500))
RAYON_NUM_THREADS=$T $B run --node 1 --listen 127.0.0.1:$PORT "$@" > /tmp/qsim-dist-n1.$PORT.log 2>/tmp/qsim-dist-n1.$PORT.err &
P=$!
sleep 0.3
RAYON_NUM_THREADS=$T $B run --node 0 --connect 127.0.0.1:$PORT "$@"
wait $P
cat /tmp/qsim-dist-n1.$PORT.log
grep -v "^listening" /tmp/qsim-dist-n1.$PORT.err
rm -f /tmp/qsim-dist-n1.$PORT.log /tmp/qsim-dist-n1.$PORT.err
