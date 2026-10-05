#!/usr/bin/env bash
# Reproduces the data of research/shor/approx-modexp.md.
#
#   export CARGO_TARGET_DIR=...            (optional)
#   cargo build --release --example approx_modexp
#   export GIDNEY_SRC=/path/to/gidney25/src   (the Zenodo release's src/, for the Python checks)
#   bash research/data/approx-modexp/run_experiments.sh [part]
#
# parts: xcheck verify dist sweep gate model moon (default: all but moon)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bin="${CARGO_TARGET_DIR:-$here/../../../target}/release/examples/approx_modexp"
out="$here/out"
mkdir -p "$out"
part="${1:-all}"

# instances (balanced semiprimes): n = 8, 10, 12, 14, 20, 24 bits
N8="N=143 g=2"            # 11*13
N10="N=899 g=2"           # 29*31
N12="N=3127 g=3122"       # 53*59, the paper's Figure 4 instance (main0 --n 3127 --g 3122)
N14="N=11663 g=2"         # 107*109
N20="N=1022117 g=2"       # 1009*1013
N24="N=16016003 g=2"      # 4001*4003

run() { echo "+ $*" >&2; "$@"; }

if [[ $part == all || $part == xcheck ]]; then
  # tables vs the paper's own precomputation; full-superposition run of the
  # paper's own approx_modexp on a genuinely quantum backend vs this simulator
  (cd "$here" && run python3 xcheck_tables.py) | tee "$out/xcheck_tables.txt"
  (cd "$here" && run python3 qbackend.py $N8 m=10 f=6 mask=2 tag=n8) | tee "$out/xcheck_qbackend_n8.txt"
  (cd "$here" && run python3 qbackend.py N=323 g=3 mode=eh f=7 mask=3 w1=2 w3a=2 w3b=3 w4=3 tag=n9eh) | tee "$out/xcheck_qbackend_n9eh.txt"
fi

if [[ $part == all || $part == verify ]]; then
  # every branch, outcome streams: all-0, all-1, 8 random seeds
  {
    run "$bin" verify $N8 m=10 f=6 mask=2 seeds=8
    run "$bin" verify $N10 m=20 f=8 mask=paper seeds=8
    run "$bin" verify $N12 m=22 f=10 mask=paper w1=4 w3a=2 w3b=3 w4=4 seeds=8
    run "$bin" verify $N12 mode=eh f=10 mask=paper w1=3 w3a=2 w3b=3 w4=4 seeds=8
    run "$bin" verify $N14 mode=eh f=12 mask=paper w1=5 w3a=1 w3b=2 w4=4 seeds=8
  } | tee "$out/verify.txt"
fi

if [[ $part == all || $part == dist ]]; then
  {
    run "$bin" dist $N10 m=20 f=8 mask=paper peaks_out="$out/peaks_n10.csv"
    run "$bin" dist $N12 m=22 f=10 mask=paper w1=4 w3a=2 w3b=3 w4=4 peaks_out="$out/peaks_n12.csv"
    run "$bin" dist $N12 mode=eh f=10 mask=paper w1=3 w3a=2 w3b=3 w4=4
    run "$bin" dist $N14 mode=eh f=12 mask=paper w1=5 w3a=1 w3b=2 w4=4
  } | tee "$out/dist.txt"
fi

if [[ $part == all || $part == sweep ]]; then
  base="$N12 mode=eh w1=3 w3a=2 w3b=3 w4=4"
  run "$bin" sweep $base f=10 key=mask vals=0,1,2,3,4,5,6,7,8,9 out="$out/sweep_mask_f10.csv"
  run "$bin" sweep $N12 m=22 w1=4 w3a=2 w3b=3 w4=4 f=10 key=mask vals=0,2,4,5,6,7,8,9 out="$out/sweep_mask_f10_shor.csv"
  run "$bin" sweep $base mask=4 key=f vals=6,7,8,9,10,11 out="$out/sweep_f_mask4.csv"
  run "$bin" sweep $base f=10 mask=paper key=w1 vals=1,2,3,4,5,6 out="$out/sweep_w1.csv"
  run "$bin" sweep $base f=10 mask=paper key=w3 vals=1,2,3,4 out="$out/sweep_w3.csv"
  run "$bin" sweep $base f=10 mask=paper key=w4 vals=1,2,3,4,5,6 out="$out/sweep_w4.csv"
fi

if [[ $part == all || $part == gate ]]; then
  {
    run "$bin" gate w4=2 f=6 trials=64
    run "$bin" gate w4=3 f=8 T=211 trials=32
  } | tee "$out/gate.txt"
fi

if [[ $part == all || $part == model ]]; then
  # the paper's own success-rate model (main1), evaluated exactly, at the
  # circuit's mask widths (W·2^t in units of N)
  (cd "$here" && run python3 paper_model.py 3127 3122 1 32 64 128 256 313) | tee "$out/model_n12.txt"
fi

if [[ $part == moon ]]; then
  {
    run "$bin" dist $N20 mode=eh s=2 f=14 mask=paper w1=5 w3a=2 w3b=3 w4=4 unmasked=0
    run "$bin" dist $N24 mode=eh s=3 f=16 mask=paper w1=5 w3a=2 w3b=3 w4=4 unmasked=0
  } | tee "$out/moon.txt"
fi
