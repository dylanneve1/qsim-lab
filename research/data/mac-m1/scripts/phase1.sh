#!/bin/bash
# A/B grid: main vs l1-tiling branch (flag off = NEON FMA only; t32/t64 = tiling on)
C=~/qsim-l1r4-bin/cell.sh
L1="fma t32:tile_kib=32 t64:tile_kib=64"
for T in 8 6 4; do
  for P in f32 f64; do
    $C ab qft 22,24 $P $T 5 0 main "$L1"
    $C ab qft 26 $P $T 3 0 main "$L1"
    [ $P = f32 ] && [ $T != 4 ] && $C ab qft 28 $P $T 3 0 main "$L1"
    $C ab brick 22 $P $T 5 20 main "$L1"
    $C ab brick 24 $P $T 3 20 main "$L1"
    $C ab brick 26 $P $T 3 20 main "$L1"
    [ $P = f32 ] && [ $T != 4 ] && $C ab brick 28 $P $T 3 20 main "$L1"
  done
done
# sanity: branch with simd=0 must equal main (same portable kernels)
$C sanity brick 24 f32 8 3 20 main "off:simd=0 fma"
echo PHASE1_DONE >> ~/qsim-l1r4-bin/data/ab.log
