# Phase folding: graph-based T-count reduction beyond local peephole

Owner request: "turn the circuits into graphs and optimize." The circuit DAG
(`src/dag.rs`, `research/compiler/dag.md`) cuts about half the T gates of random
Clifford+T circuits with commutation-aware peephole. This note adds the next
step: **phase folding** over the parity of each wire (Amy, Maslov & Mosca
2014), and compares against PyZX (the ZX-calculus reference).

## Prior art
- M. Amy, D. Maslov, M. Mosca, *Polynomial-time T-depth optimization of
  Clifford+T circuits via matroid partitioning*, IEEE TCAD 2014
  (arXiv:1303.2042): phase polynomials, parity tracking, merge rotations on
  the same parity. This is what `compile::phase_fold` implements.
- L. Heyfron, E. Campbell, *An efficient quantum compiler that reduces T
  count* (TODD), 2018 (arXiv:1712.01557): re-synthesise phase polynomials by
  Reed-Muller decoding. **Not implemented here.**
- A. Kissinger, J. van de Wetering, *Reducing T-count with the ZX-calculus*,
  PRA 2020 (arXiv:1903.10477), and PyZX: graph-like diagrams, spider
  fusion, local complementation, pivoting, gadget fusion, phase
  teleportation. Used as the **reference** (`pyzx 0.10.7`). **Not
  reimplemented** (see "Unfinished").
- M. Amy, *Towards large-scale functional verification of universal quantum
  circuits* (2018): path sums; their `H`-elimination rules explain what the
  ZX tools find beyond phase folding.

## What was built
`src/compile/phasefold.rs`, `compile::phase_fold(&Circuit) -> Optimized`
(`U_original = e^{i global_phase} U_circuit`, same contract as
`compile::optimize`).

1. One pass over the ops keeps, for every wire, an *affine parity* of
   variables `x` (bitset) plus a constant bit. `CNOT` xors parities,
   `X` flips the constant, `SWAP` swaps the wires' entries.
2. `T, T†, S, S†, Z, Rz(θ), Phase(θ)` are Z-rotations. Each adds its angle to
   the *term* for that parity (a hash map keyed by the parity bitset). A wire
   with constant 1 negates the angle and moves `e^{iθ}` to the global phase;
   `Y = i·X·Z`.
3. `H, Rx, Ry, U, SX, iSWAP`, the target of `Toffoli`, and measurement /
   reset / noise / classically-controlled ops give the affected wire a
   **fresh variable** (so no later rotation can merge with an earlier one
   through the qubit that changed basis). Diagonal gates (`CZ`, `CPhase`) and
   Toffoli controls leave parities untouched.
4. Emission: every non-rotation op stays in place; each term is emitted once
   at the position of its *first* occurrence. Angles are exact: an integer
   count of π/4 (mod 8) plus a float residual for generic `Rz` angles. Odd
   multiples of π/4 become one `T`/`T†` (plus `S`/`Z`), even multiples
   Cliffords, anything else one `Phase(θ)`.

Why the merged rotation can go to the first occurrence: conjugating the later
Z-string back through the gates between the two occurrences, it has no
support on a wire that got a fresh variable exactly when its parity does not
contain that variable, so it commutes with the gate that introduced it.
`CNOT/X/SWAP` only permute the computational basis.

Guarantees (tested, see below): exact unitary up to the tracked global phase;
the **non-Clifford gate count never increases** (each emitted term has at
least as many odd-π/4 inputs as outputs); idempotent in the non-Clifford
count.

### Wiring
`PlanOptions::phase_fold: bool`, **default false** (also in
`PlanOptions::none()`). When true, `front_end` runs `peephole -> phase_fold
-> peephole` before SWAP elimination and state propagation. `simulate()`
(`pipeline::plan_options()`) does not set it, so its behaviour is unchanged
(`tests/compiler/phasefold.rs::plan_option_keeps_simulate_identical` also compares
amplitudes with the option on).

## Exactness evidence (`tests/compiler/phasefold.rs`, proptest, 300 cases each)
- Full unitary, all `2^n` basis columns, n ≤ 4, random Clifford+T
  (`H,S,S†,T,T†,X,Y,Z,CNOT,CZ,SWAP`) and Clifford+Rz+extras (`Rz` with
  random angles, `CPhase`, `Ry`, `SX`, `Toffoli`).
- State comparison on a random universal state, n up to 10, length up to 200,
  both families, tolerance 1e-10 **including the tracked global phase**.
- Non-Clifford count never grows (all families); idempotence; composition
  `peephole -> fold -> peephole` is exact and never increases the count.
- Hand cases (T·T† across CNOT pairs, merge across an H on another wire, no
  merge across an H on the same wire, no merge across a measurement).
- `examples/phasepoly_bench.rs` additionally state-checks every row of the
  table below (≤ 22 qubits) against the original circuit.
- Whole suite: `cargo test`, `cargo clippy --all-targets -- -D warnings`,
  `cargo fmt --check` pass.

## Results 1: non-Clifford count (raw: `research/data/phasepoly/counts.csv`, `pyzx_counts.csv`)
"orig" is the non-Clifford gate count of the circuit. "peep" is the DAG
commutation-aware peephole (`compile::optimize`). "fold" is phase folding
alone. "pfp" is peephole -> fold -> peephole (what the plan option runs).
PyZX 0.10.7: "basic" = `basic_optimization`; "teleport" = `teleport_reduce`
(full_reduce + phase teleportation, circuit structure kept); "full" =
`full_reduce` + extraction + `basic_optimization`. ERR = my 90 s per-method
time limit.

| circuit | orig | peep | fold | **pfp** | PyZX basic | PyZX teleport | PyZX full |
|---|---|---|---|---|---|---|---|
| random C+T n=8 d=50 | 98 | 60 | 32 | **32** | 32 | 32 | 32 |
| random C+T n=12 d=80 | 249 | 163 | 95 | **95** | 93 | 93 | 93 |
| random C+T n=16 d=100 | 437 | 261 | 155 | **155** | 155 | 155 | 155 |
| random C+T n=32 d=200 | 1594 | 1062 | 610 | **590** | 586 | ERR | ERR |
| Cuccaro adder n=8 | 112 | 96 | 64 | **64** | 96 | 64 | 64 |
| Cuccaro adder n=16 | 224 | 192 | 128 | **128** | 192 | 128 | 128 |
| Cuccaro adder n=32 | 448 | 384 | 256 | **256** | 384 | 256 | 256 |
| Toffoli ladder n=6 r=1 | 56 | 50 | 36 | **36** | 50 | 36 | 36 |
| Toffoli ladder n=10 r=2 (is the identity, see below) | 224 | 194 | 132 | **96** | 194 | 0 | 0 |
| Toffoli ladder n=16 r=2 (identity) | 392 | 338 | 228 | **168** | 338 | 0 | 0 |
| ripple add_mod n=3 (`shor_ripple`) | 210 | 180 | 120 | **112** | 180 | 70 | 70 |
| ripple add_mod n=4 | 280 | 240 | 160 | **152** | 240 | 94 | 94 |
| ripple add_mod n=5 | 350 | 300 | 200 | **192** | 300 | 114 | 114 |
| ripple add_mod n=6 | 420 | 360 | 240 | **232** | 360 | 140 | 140 |
| ripple controlled-U_a n=3 | 1785 | 1531 | 1027 | **895** | 1531 | 571 | 571 |
| ripple controlled-U_a n=4 | 3276 | 2808 | 1880 | **1616** | 2808 | ERR | ERR |
| ripple controlled-U_a n=5 | 4123 | 3535 | 2367 | **2191** | 3535 | ERR | ERR |
| ripple controlled-U_a n=6 | 6636 | 5688 | 3804 | **3408** | 5688 | ERR | ERR |
| AQFT n=8 m=3 (π/2^m cut-off) | 54 | 43 | 32 | **32** | 43 | 32 | 32 |
| AQFT n=16 m=3 | 126 | 99 | 72 | **72** | 99 | 72 | 72 |
| AQFT n=16 m=5 | 195 | 145 | 95 | **95** | 145 | 95 | 95 |
| AQFT n=32 m=4 | 354 | 267 | 180 | **180** | 267 | 180 | 180 |
| full QFT n=12 | 198 | 143 | 88 | **88** | 143 | 88 | 88 |

Fold wall time: 0.02 ms (small) to 3 ms (16k-gate controlled-U_a, n=6).
PyZX teleport: 0.04 s to 50 s for the same circuits (Python).

Reading it:
- Phase folding beats the DAG peephole everywhere: about half the remaining
  non-Clifford gates on random Clifford+T (peephole 60 → fold 32 on n=8), and
  **exactly the PyZX teleport / full_reduce count on the QFT/AQFT and
  Cuccaro circuits**. On the small random circuits it is within 1–2% of PyZX
  (95 vs 93, 590 vs 586); PyZX's basic_optimization has a similar fold
  inside and gets 93/586.
- **ZX wins where Hadamard structure is deep**: the ripple modular adder,
  where PyZX reaches 70 vs our 112 (n=3), i.e. ZX is ~1.6x better there
  (1.57-1.68x across n=3..6 and the controlled-U_a rows), and the doubled Toffoli ladder, which is
  the identity and which the ZX tools find, while folding only halves it.
  Folding stops at every `H` on the *same* wire. ZX pivoting removes
  the Hadamard pair of a Toffoli (path-sum `HH`/`Elim` rules) and exposes
  more equal parities. Reproducing that is the unfinished stretch.
- The ladder with `r=2` is an involution, so the ZX answer 0 is correct and
  was not planned by me; I kept the row because it shows the gap honestly.

## Results 2: end-to-end `simulate()` wall time
`examples/phasepoly_e2e.rs`: `compile_sampling` with the pipeline's
`plan_options()` (adaptive engine on) with `phase_fold` off vs on, then 64
shots, **compile + fold included**, interleaved off/on 3 times each, min of
3. Run through `bench.sh` on the shared 4-vCPU VM (load at start 3.3–3.9, so
treat differences below ~5% as noise). Raw: `research/data/phasepoly/e2e_run1.csv`,
`e2e_adders.csv`.

| circuit (Clifford+T, measured) | n | engine | gates off→on | non-Clifford off→on | off s | on s | speed-up |
|---|---|---|---|---|---|---|---|
| Cuccaro adder n=12, 4 inputs superposed | 26 | adaptive | 291→247 | 144→96 | 0.0355 | 0.0322 | 1.10x |
| same, 6 superposed | 26 | adaptive | 319→275 | 144→96 | 0.1242 | 0.1000 | 1.24x |
| same, 8 superposed | 26 | adaptive | 347→303 | 144→96 | 0.6256 | 0.5444 | 1.15x |
| Cuccaro n=14, 5 superposed | 30 | adaptive | 345→293 | 168→112 | 0.2630 | 0.2145 | 1.23x |
| Cuccaro n=16, 5 superposed | 34 | adaptive | 385→325 | 192→128 | 1.1398 | 0.9742 | 1.17x |
| random Clifford+T n=20/24/16 (few T) | 16–24 | adaptive / tableau | ≈ same | ≈ same | 0.0004–0.0005 | same | 0.86–1.04x (noise) |
| glued Clifford+T n=20, 26 (200–312 T) | 20, 26 | state vector | ≈ same | — | 0.039 / 3.85 | same | 0.95–1.00x |

- **Verdict**: a real but modest win (10–24%) on arithmetic circuits that
  land on the adaptive engine; the cost there is dominated by the active
  register `d` (unchanged, 26 and 34), not by the number of rotations, so
  cutting 33% of the T gates only buys ~15–20%. Zero help on random
  Clifford+T (T gates there sit between Hadamard-heavy Clifford layers, so
  few parities repeat) and none on state-vector-bound circuits, which is why
  the option is **off by default**.
- The fold itself costs 0.03–3 ms and is below noise.
- Cases where the dispatcher already picks the tableau or Pauli paths are
  tiny and unaffected.

## Unfinished / negative
- **ZX stretch not implemented in Rust.** PyZX numbers above are the
  reference. The gap (ripple add_mod 112 vs 70) needs full graph-like
  simplification with phase teleportation; the delicate part is the exact
  global-phase bookkeeping of pivots/local complementation (testable against
  the state vector, but a few hundred lines). Folding recovers all the gain
  on QFT/AQFT/Cuccaro without it.
- TODD / matroid-partition re-synthesis of each phase-polynomial block is not
  done; folding places each merged rotation at its first occurrence.
- Folding does not reduce `active_dimension` of the adaptive engine on the
  adder rows (d is bound by the superposed inputs), so the end-to-end gain is
  bounded by gate count.
- Possible follow-up: apply the fold before Toffoli lowering is not
  meaningful (Toffolis are barriers); apply it to lowered circuits.

## Commands
```
CARGO_BUILD_JOBS=2 cargo test --test phasefold
cargo run --release --example phasepoly_bench          # counts + state checks + circuits/*.qasm
/tmp/fw/bin/python research/data/phasepoly/pyzx_compare.py   # PyZX columns
bench.sh ./target/release/examples/phasepoly_e2e [filter]    # end-to-end A/B
```
Machine: AMD EPYC-Rome, 4 vCPU shared with other agents (load 3–4 during
timed runs), Rust stable, release profile.

## Audit, round 4 (3 Oct 2026, exp/phasepoly-r4)
Rebased onto main fb30f56 cleanly. Independent differential fuzz
`tests/audit/audit_phasefold.rs` against the naive reference SV, comparing the whole
instrument (every measurement / reset / flip branch, unnormalised, global
phase included): all gate kinds incl. iSWAP/SX/U/Toffoli/CPhase, Clifford+T
up to 12 qubits, mid-circuit measurement, reset, classically controlled
gates (incl. rotations), Pauli flips, repeated measuring rounds, Cuccaro
adders, Toffoli ladders, Shor ripple controlled-U_a (10 qubits), and the
`PlanOptions::phase_fold` pipeline. No discrepancy in ~10k cases
(`QSIM_FUZZ_ITERS=40`); six deliberately broken variants of the pass are
each caught. Counts above reproduced exactly with `phasepoly_bench`.

Reading the CSV: `t_*` columns count only `T`/`T†` gates. After
peephole -> fold -> peephole the random n=32 circuit has 182 T + 158 T† **and
250 `Phase(odd·π/4)`** gates, so its T-count is **590** (the `nc` column),
not the 340 of `t_pfp`. Use `nc_*` for T-count claims.

Observation (not done): `Measure` refreshes the wire's variable, which is
sound but conservative: a Z-basis measurement commutes with diagonal
rotations and leaves the wire's value unchanged (the audit's mutation that
removes it stays exact), so rotations could also merge across measurements.
