#!/bin/bash
C=~/qsim-l1r4-bin/cell.sh
L1="fma t32:tile_kib=32 t64:tile_kib=64"
# n=28 f32 (bench-only builds with the state cap raised)
export BM=~/qsim-l1r4-bin/bench28_main BL=~/qsim-l1r4-bin/bench28_l1
for T in 8 6 4; do
  $C ab28 qft 28 f32 $T 3 0 main "$L1"
  $C ab28 brick 28 f32 $T 3 20 main "$L1"
done
unset BM BL
# block-size sweep (branch binary only; b256 = default block, repeated in each cell as the anchor)
U="b256:block_kib=256 b512:block_kib=512 b1024:block_kib=1024 b2048:block_kib=2048 b4096:block_kib=4096"
TT="b256:block_kib=256 b256_t64:block_kib=256,tile_kib=64 b512_t64:block_kib=512,tile_kib=64 b1024_t64:block_kib=1024,tile_kib=64 b2048_t64:block_kib=2048,tile_kib=64 b4096_t64:block_kib=4096,tile_kib=64"
TK="b256:block_kib=256 b1024_t16:block_kib=1024,tile_kib=16 b1024_t32:block_kib=1024,tile_kib=32 b1024_t128:block_kib=1024,tile_kib=128 b2048_t32:block_kib=2048,tile_kib=32 b2048_t128:block_kib=2048,tile_kib=128"
SL="b256:block_kib=256 b1024_s4:block_kib=1024,slots=4 b1024_s8:block_kib=1024,slots=8 b1024_t64_s4:block_kib=1024,tile_kib=64,slots=4 b1024_t64_s8:block_kib=1024,tile_kib=64,slots=8"
for P in f32 f64; do
  for set in "$U" "$TT" "$TK" "$SL"; do
    $C sweep brick 24 $P 6 3 20 "" "$set"
    $C sweep brick 26 $P 8 3 20 "" "$set"
  done
  $C sweep qft 26 $P 8 3 0 "" "$U"
  $C sweep qft 26 $P 8 3 0 "" "$TT"
done
echo PHASE2_DONE >> ~/qsim-l1r4-bin/data/sweep.log
