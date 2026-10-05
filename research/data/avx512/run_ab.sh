#!/bin/bash
# Kernel-tier and tuning A/B runs for research/performance/avx512.md (qsim-lab only).
# usage: run_ab.sh <phase> ; binaries in $BIN (default /dev/shm/qsim/avx512/bin):
#   sv_file_bench-main  built from main 6b21728
#   sv_file_bench-new   built from this branch
# Each cell is one hold of the bench lock (ab.py takes it per cell). Output CSVs land next to
# this script.
set -e
D=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-/dev/shm/qsim/avx512/bin}
M=$BIN/sv_file_bench-main
N=$BIN/sv_file_bench-new
AB="python3 $D/ab.py"
WL=qft,brick_cz,brick_su4,qv,qaoa
case "$1" in
  tiers)   # main vs AVX2 tier vs AVX-512 tier (same branch binary), defaults + dense k=2
    for prec in f32 f64; do
      $AB --out $D/ab_tiers.csv --threads 8 --prec $prec --n ${NS:-24,26} --wl $WL --rounds 3 \
        --cfg main=$M --cfg main_d2=$M:dense=2 \
        --cfg avx2_d2=$N@QSIM_NO_AVX512=1:dense=2 --cfg avx512_d2=$N:dense=2
    done ;;
  block)   # block size / fusion width / slots / L1 tile on the AVX-512 tier
    for prec in ${PRECS:-f32 f64}; do
      $AB --out $D/ab_tune.csv --threads ${T:-8} --prec $prec --n ${NS:-26} --wl $WL --rounds 3 \
        --cfg b256_d2=$N:dense=2 --cfg b512_d2=$N:dense=2,block_kib=512 \
        --cfg b1024_d2=$N:dense=2,block_kib=1024 --cfg b1024_d0=$N:dense=0,block_kib=1024 \
        --cfg b1024_d3=$N:dense=3,block_kib=1024 --cfg b1024_s8=$N:dense=2,block_kib=1024,slots=8 \
        --cfg b1024_t32=$N:dense=2,block_kib=1024,tile_kib=32
    done ;;
  *) echo "phase: tiers | block"; exit 2 ;;
esac
