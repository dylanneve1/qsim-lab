# Exact resynthesis probes (P6 investigation)

These are exact simplification passes for {single-qubit, CZ} circuits. They exist to expose the structure that
masking obfuscation hides. Every pass is exact up to the rounding of the QASM angles (~1e-7).

- `makhlin.py FILE.qasm`: groups consecutive same-pair CZs (with the 1q gates between them) into 2-qubit blocks and
  computes each block's Makhlin invariants. It then counts blocks whose invariants match the *inverse* of another
  block's, a mirror-pair signature that is independent of single-qubit sweeps.
- `resyn.py FILE.qasm`: merges runs on the same pair and splits operator-Schmidt rank-1 blocks back into 1q gates,
  repeating until nothing changes.
- `commute.py FILE.qasm`: commutation-aware merging. A block may move back past blocks it commutes with (exact
  matrix test) to meet a block on the same pair. `commute.py test` checks exactness on 20 random circuits.

Findings, portal P6 (62q, 3494 CZ):
- P6 is not P5-like. P5 has 902 two-CZ blocks in 43 perfect-matching layers. P6 has 2593 blocks with CZ-run sizes
  {1: 1865, 2: 555, 3: 173} over 182 sparse layers.
- 186 of its multi-CZ blocks are exactly local (Makhlin identity class), i.e. masking gadgets. Removing them takes
  3494 CZ to 2342 two-qubit units in 3 passes. Commutation-aware merging removes only 6 more.
- What remains: operator-Schmidt rank 2 (CZ class) 1862, rank 3 21, rank 4 451. Inverse-invariant pairs among the
  rank-4 units are short-range and nested on relabelled wires (e.g. units 320↔329 and 321↔328). That indicates
  many local identity patches with SWAP relabelling between them, not one global mirror.
