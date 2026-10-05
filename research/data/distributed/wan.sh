#!/bin/bash
# Mac (node 0) <-> VPS (node 1) over an SSH session (stdin/stdout of the remote
# process; the VPS sshd forbids port forwarding).
# wan.sh <mode> <mac-threads> <vps-mem-limit> <vps-threads> <args...>
B=~/qsim-dist-target/release/examples/dist_sv
MODE=$1; T=$2; MEM=$3; VT=$4; shift 4
ARGS="$*"
REMOTE="RAYON_NUM_THREADS=$VT systemd-run --user --scope -q -p MemoryMax=$MEM -p MemorySwapMax=0 -- nice -n 5 /tmp/qsim-wt/dist-bin/dist_sv $MODE --stdio 1 --node 1 $ARGS"
RAYON_NUM_THREADS=$T $B $MODE --node 0 --spawn "ssh -T -o BatchMode=yes -o ServerAliveInterval=30 claudius '$REMOTE'" $ARGS
