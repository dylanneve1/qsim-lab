#!/usr/bin/env bash
# Regenerates research/data/shor/ripple_benchmarks.txt (ripple-carry oracle,
# research/shor/shor.md). Run from the repo root after `cargo build --release`.
set -e
Q=./target/release/qsim
OUT=research/data/shor/ripple_benchmarks.txt
rm -f "$OUT"

run() {
    echo "== $*" | tee -a "$OUT"
    $Q run shor "$@" | tee -a "$OUT"
}

echo "Starting Ripple benchmarks at $(date)" | tee "$OUT"

# Small and medium N: block vs gate-by-gate
run --modulus 15 --semiclassical --sparse --oracle ripple --seed 1
run --modulus 15 --semiclassical --sparse --oracle ripple --gate-by-gate --seed 1
run --modulus 21 --semiclassical --sparse --oracle ripple --seed 1
run --modulus 21 --semiclassical --sparse --oracle ripple --gate-by-gate --seed 1
run --modulus 143 --semiclassical --sparse --oracle ripple --seed 1
run --modulus 143 --semiclassical --sparse --oracle ripple --gate-by-gate --seed 1
run --modulus 1003 --semiclassical --sparse --oracle ripple --seed 1
run --modulus 1003 --semiclassical --sparse --oracle ripple --gate-by-gate --seed 1

# Target N ~ 1e6 semiprimes
run --modulus 1003883 --semiclassical --sparse --oracle ripple --seed 1
run --modulus 1005973 --semiclassical --sparse --oracle ripple --seed 1

echo "Finished Ripple benchmarks at $(date)" | tee -a "$OUT"
