# ZX phase-region pre-pass

## Hypothesis

The Pauli-path backend is exponential in the number of non-Clifford Z phases.
An exact, extraction-free phase-teleportation subset can therefore be valuable:
move a one-legged Z spider through CZ, through a CNOT control, and across a
SWAP (with a wire relabel), then fuse it with another Z spider.  This is a
strictly local graph-like ZX rewrite and does not use approximation or a
general circuit extractor.

## Change

`src/zx.rs` adds `zx_simplify` / `ZxSimplify::zx_simplify`. Measurements are
hard barriers. It canonicalises Z/S/T/Rz/Phase to a minimal Clifford+T phase
spelling, commutes only direct identities, and fuses angles modulo `2π`.
The pass also removes adjacent inverse pairs. It intentionally does **not**
claim full PyZX `full_reduce`: local complementation, pivoting and arbitrary
phase-polynomial parity-gadget reduction require a complete graph-like
diagram representation and extractor, which this small circuit IR does not
yet have.

## Exactness

`zx::tests::random_circuits_are_exact_up_to_global_phase` compares original
and reduced state vectors for seeded random Clifford+T circuits for every
width 1..=12; fidelity must exceed `1 - 1e-12`. The dedicated CNOT-control
and SWAP test starts with four T/T† gates and reduces to zero.

## Measurements

Benchmarks are pending the shared build queue at the time of this note. Any
timing added here must be run through `../bench.sh` on the shared 4-vCPU VM;
no unverified speedup is claimed. The asymptotic result is exact: a removed
T removes one possible Pauli-path branch, an up-to-2x factor in its worst
case, while state-vector work drops by one diagonal gate.

## Negative result / scope boundary

Random Clifford+T layers usually place T gates behind H gates or CNOT target
legs, which are intentional barriers to this conservative pass. Thus this
pre-pass is expected to help structured phase-heavy circuits, not uniformly
random circuits. Full ZX local-complementation/pivoting was rejected for this
milestone rather than risk an unverified extractor.
