#!/bin/bash
until grep -q PHASE2_DONE ~/qsim-l1r4-bin/data/sweep.log 2>/dev/null; do sleep 10; done
C=~/qsim-l1r4-bin/cell.sh; R=~/qsim-l1r4-bin/lockrun.sh; PY=~/qsim-bench-venv/bin/python
cd ~/qsim-l1r4-bin
# head-to-head vs qsim / Aer (RESULTS.md circuits), f32, 8 threads, all on the Mac
export T=8
$R sota $PY qsim_bench.py qft 22,24
$R sota $PY qsim_bench.py qft 26
$R sota $PY qsim_bench.py brick 22,24
$R sota $PY aer_bench.py qft 22,24,26
$R sota $PY aer_bench.py brick 22,24
$C sotalab qft 22,24,26 f32 8 3 0 main "fma t64:tile_kib=64 b1024:block_kib=1024 b1024_t64:block_kib=1024,tile_kib=64"
$C sotalab brick 22,24 f32 8 3 20 main "fma t64:tile_kib=64 b1024:block_kib=1024 b1024_t64:block_kib=1024,tile_kib=64"
$C sotalab qft 22,24,26 f32 6 3 0 "" "fma b1024:block_kib=1024 b1024_t64:block_kib=1024,tile_kib=64"
$C sotalab brick 22,24 f32 6 3 20 "" "fma b1024:block_kib=1024 b1024_t64:block_kib=1024,tile_kib=64"
# re-runs of cells that overlapped another agent's cargo build
export BM=~/qsim-l1r4-bin/bench28_main BL=~/qsim-l1r4-bin/bench28_l1
L1="fma t32:tile_kib=32 t64:tile_kib=64"
$C ab28r qft 28 f32 4 3 0 main "$L1"
$C ab28r brick 28 f32 4 3 20 main "$L1"
unset BM BL
U="b256:block_kib=256 b512:block_kib=512 b1024:block_kib=1024 b2048:block_kib=2048 b4096:block_kib=4096"
TT="b256:block_kib=256 b256_t64:block_kib=256,tile_kib=64 b512_t64:block_kib=512,tile_kib=64 b1024_t64:block_kib=1024,tile_kib=64 b2048_t64:block_kib=2048,tile_kib=64 b4096_t64:block_kib=4096,tile_kib=64"
TK="b256:block_kib=256 b1024_t16:block_kib=1024,tile_kib=16 b1024_t32:block_kib=1024,tile_kib=32 b1024_t128:block_kib=1024,tile_kib=128 b2048_t32:block_kib=2048,tile_kib=32 b2048_t128:block_kib=2048,tile_kib=128"
SL="b256:block_kib=256 b1024_s4:block_kib=1024,slots=4 b1024_s8:block_kib=1024,slots=8 b1024_t64_s4:block_kib=1024,tile_kib=64,slots=4 b1024_t64_s8:block_kib=1024,tile_kib=64,slots=8"
for set in "$U" "$TT" "$TK" "$SL"; do
  $C sweepr brick 24 f32 6 3 20 "" "$set"
  $C sweepr brick 26 f32 8 3 20 "" "$set"
done
echo PHASE3_DONE >> ~/qsim-l1r4-bin/data/sweepr.log
