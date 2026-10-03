#!/bin/bash
until grep -q PHASE3_DONE ~/qsim-l1r4-bin/data/sweepr.log 2>/dev/null; do sleep 10; done
C=~/qsim-l1r4-bin/cell.sh
U="b256:block_kib=256 b512:block_kib=512 b1024:block_kib=1024 b2048:block_kib=2048 b4096:block_kib=4096"
TT="b256:block_kib=256 b512_t64:block_kib=512,tile_kib=64 b1024_t64:block_kib=1024,tile_kib=64"
for P in f64 f32; do
  $C sweepq qft 26 $P 8 5 0 "" "$U"
  $C sweepq qft 26 $P 8 5 0 "" "$TT"
  $C sweepq qft 24 $P 6 5 0 "" "$U"
done
export BM=~/qsim-l1r4-bin/bench28_main BL=~/qsim-l1r4-bin/bench28_l1
$C ab28r qft 28 f32 8 3 0 main "fma t32:tile_kib=32 t64:tile_kib=64 b1024:block_kib=1024 b1024_t64:block_kib=1024,tile_kib=64"
$C ab28r brick 28 f32 8 3 20 "" "fma b1024:block_kib=1024 b1024_t64:block_kib=1024,tile_kib=64"
echo PHASE4_DONE >> ~/qsim-l1r4-bin/data/sweepq.log
