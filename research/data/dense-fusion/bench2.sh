#!/bin/bash
# dense-fusion A/B on the M1 Pro (cost rule): one locked chunk per line, >=65 s gaps
B=$HOME/qsim-integ-target/release/examples/l1_bench
OUT=$HOME/qsim-integ-data/fusion_ab2.md
export CSV=$HOME/qsim-integ-data/fusion_ab2.csv GHZ=3.228
run() {
  local th=$1; shift
  until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
  echo "## $(date -u +%FT%TZ) threads=$th load=$(sysctl -n vm.loadavg) args: $*" >> $OUT
  RAYON_NUM_THREADS=$th timeout 150 $B "$@" >> $OUT 2>&1
  rmdir /tmp/qsim-mac-bench.lock
  echo "## end $(date -u +%FT%TZ) load=$(sysctl -n vm.loadavg)" >> $OUT
  sleep 65
}
C="off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3"
run 6 brick 24 f32 3 20 $C
run 6 su4 24 f32 3 20 $C
run 6 qft 24 f32 3 1 $C
run 6 brick 24 f64 3 20 $C
run 6 su4 24 f64 3 20 $C
run 8 brick 26 f32 3 20 $C
run 8 su4 26 f32 3 20 $C
run 6 brick 20 f32 5 20 $C
run 6 su4 20 f32 5 20 $C
run 6 ghz 24 f32 3 1 $C
echo DONE >> $OUT
