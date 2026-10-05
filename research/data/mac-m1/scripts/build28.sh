#!/bin/bash
# bench-only builds with the 1 GiB state cap raised to 4 GiB (NOT committed) for n=28 f32
set -e
C=/opt/homebrew/bin/cargo
for d in main:~/qsim-l1r4-main:~/qsim-l1r4-target-main l1:~/qsim-l1r4:~/qsim-l1r4-target; do
  IFS=: read tag wt tgt <<< "$d"; wt=$(eval echo $wt); tgt=$(eval echo $tgt)
  cd $wt; sed -i '' 's/^pub const MAX_STATE_BYTES: u128 = 1 << 30;/pub const MAX_STATE_BYTES: u128 = 1 << 32;/' src/engines/statevector.rs
  grep -q "1 << 32" src/engines/statevector.rs
  CARGO_TARGET_DIR=$tgt nice -n 10 $C build --release --example l1_bench
  cp $tgt/release/examples/l1_bench ~/qsim-l1r4-bin/bench28_$tag
  git checkout src/engines/statevector.rs
done
echo BUILD28_OK
