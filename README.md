# qsim-lab

> **Results (1 Oct 2026):** head-to-heads against Google qsim, Qiskit Aer and Stim, engine speedups, Shor at N ≈ 10⁶ and below-threshold surface codes are in [RESULTS.md](RESULTS.md). The architecture for the next phase is in [research/ARCHITECTURE.md](research/ARCHITECTURE.md).

A quantum circuit simulator written from scratch in Rust, for learning how
different simulation methods scale. It has four backends that share one
circuit representation:

| backend | module | state stored as | memory | handles |
|---|---|---|---|---|
| state vector | `statevector` | `2^n` complex amplitudes | `2^n × 8 B` (f32) or `× 16 B` (f64) | any gate |
| stabilizer tableau (CHP) | `stabilizer` | `2n` Pauli generators, bit packed | `4n^2` bits | Clifford gates only |
| Pauli paths (Clifford+T) | `pauli_path` | a sum of Pauli strings (Heisenberg picture) | up to `2^t` terms for `t` T gates | any gate, exact expectation values |
| matrix product state | `mps` | `n` tensors of size `χ × 2 × χ` | `≈ 32 n χ^2` bytes | any gate; cost set by entanglement |

The benchmarks below show the trade-offs: the state vector is exact and
general but doubles in size with every qubit; the tableau handles tens of
thousands of qubits but only for Clifford circuits; adding T gates to a
Clifford circuit costs time exponential in the number of T gates (not in the
number of qubits); and an MPS is cheap exactly when the state has little
entanglement.

The code depends on `num-complex`, `num-traits`, `rayon`, `rand`, `faer`
(SVD and QR for the MPS) and `clap` (CLI), plus `proptest` for tests. There
is no `unsafe` code.

## Building and running

```sh
cargo build --release
cargo test

# examples
cargo run --release --example bell
cargo run --release --example ghz
cargo run --release --example bernstein_vazirani
cargo run --release --example grover
cargo run --release --example qft
cargo run --release --example shor        # factors 15 and 21
cargo run --release --example clifford_t  # the tableau refuses a T gate

# CLI
cargo run --release -- run ghz --qubits 6 --backend stab
cargo run --release -- run shor --modulus 21
cargo run --release -- bench sv           # also: qft, stab, clifford-t, mps
```

A minimal use of the library:

```rust
use qsim_lab::{Circuit, StateVectorF32, Tableau};
use rand::SeedableRng;

let mut c = Circuit::new(3);
c.h(0).cnot(0, 1).cnot(1, 2).measure_all();
let mut rng = rand::rngs::StdRng::seed_from_u64(1);

let mut sv = StateVectorF32::new(3);
let bits = c.run(&mut sv, &mut rng).unwrap();   // dense simulation

let mut tab = Tableau::new(3);
let bits2 = c.run(&mut tab, &mut rng).unwrap(); // stabilizer simulation
```

Conventions: qubit `q` is bit `q` of a basis-state index (qubit 0 is the
least significant bit). Two-qubit gate matrices are indexed by
`2·bit(first argument) + bit(second argument)`, so `Cnot(c, t)` has the
textbook matrix.

## The backends

### State vector

An `n`-qubit state is a vector of `2^n` amplitudes, generic over `f32` or
`f64` (`StateVector<T>`). A single-qubit gate on qubit `q` is applied in
place: the amplitudes whose indices differ only in bit `q` form `2^(n-1)`
pairs, and the 2×2 matrix is applied to each pair. The vector is walked as
runs of `2^q` "bit clear" amplitudes followed by `2^q` "bit set" amplitudes,
so the inner loop works on two equally long contiguous slices, which LLVM
vectorises. Two-qubit gates use the same idea with four slices. The outer
loops are split across threads with rayon in blocks of 16K amplitudes;
when the target qubit is high and there are only a few long runs, the runs
themselves are split.

Gates are special-cased where it saves memory traffic: diagonal gates (Z, S,
T, RZ, phase) only scale the amplitudes whose bit is 1, X/CNOT/SWAP are pure
swaps, and CZ/controlled-phase only touch a quarter of the vector. A gate is
limited by memory bandwidth, not arithmetic.

Measurement computes the probability of the qubit being 1 with a parallel
reduction, then collapses and renormalises. `sample(shots)` draws many
full bit strings without collapsing, in one pass over the vector plus a sort
of the random numbers.

**Scaling.** Memory is `2^n` amplitudes, and every gate touches a constant
fraction of them, so both memory and time per gate double per qubit. 30
qubits in f64 is 16 GiB; 50 qubits would be 16 PiB.

**Memory cap.** This was developed on a shared machine, so `StateVector`
refuses to allocate more than 1 GiB (`MAX_STATE_BYTES`). In practice the
benchmarks stop at **26 qubits in f32 (512 MiB)** and **25 qubits in f64
(512 MiB)**. Raise the constant if you have the memory.

### Stabilizer tableau

By the Gottesman–Knill theorem, a state built from `|0…0>` with Clifford
gates (H, S, CNOT and the Paulis) is described by `n` commuting Pauli
operators that stabilise it. The Aaronson–Gottesman (CHP) tableau stores
these `n` stabilizers plus `n` destabilizers as `2n` rows of `2n` bits (the
`x` and `z` parts of each Pauli string) with a sign bit. A gate updates two
columns of every row; a measurement multiplies rows together.

Memory is `4n^2` bits: 10,000 qubits take 48 MiB, 46,336 qubits take 1 GiB
(this crate's cap, `MAX_TABLEAU_BYTES`), and 1,000,000 qubits would take
about 125 GB. That quadratic term is why a stabilizer simulator that is fast
at 20,000 qubits fails at a million.

Implementation details:

* The four `n×n` blocks are bit packed in `u64` words, with `n` rounded up to
  a multiple of 64.
* The blocks are stored qubit-major (a line per qubit), so an H or CNOT is a
  few word operations over `2n/64` words. Read the other way round, the same
  lines are the rows of the inverse tableau `C†`. As in Stim, the tableau also
  keeps the inverse rows' signs. A deterministic Z measurement is then a sign
  lookup (`O(n/64)` instead of `O(n^2/64)`), and a random one is done by
  applying gates to generator indices in one pass over the lines. Gates and
  measurements never transpose the tableau: syndrome-extraction rounds run
  about 100x faster than with the textbook layout switching (see
  `research/stab.md`). `set_sign_tracking(false)` skips the per-gate sign work
  when no single-qubit measurements follow.
* The sign of a product of two Pauli rows is computed 64 qubits at a time
  with masks and `popcount`.
* Measuring every qubit with the textbook CHP procedure costs `O(n^2)` per
  deterministic outcome, `O(n^3)` in total (2.4 s for 5,000 qubits, see
  below). `measure_all` instead puts the stabilizer rows in row-echelon form
  once, which shows the state as a uniform superposition over an affine set
  of bit strings, and samples from that set. The elimination keeps the
  destabilizers consistent, so `sample(shots)` does not disturb the state.

Gates outside the Clifford group are rejected with `SimError::Unsupported`
(phase rotations by multiples of π/2 are accepted as powers of S).

### Clifford+T by Pauli-path summation

T is the cheapest gate that takes you out of the Clifford group, and
Clifford+T is universal. To compute an expectation value
`<0| U† O U |0>` for a Pauli observable `O`, the `pauli_path` module
pushes `O` backwards through the circuit, `O → G† O G`, last gate first.
Clifford gates map a Pauli string to a single Pauli string with a sign (the
same rules as the tableau). A T gate, or any Z rotation by θ, maps
`X → cos θ X + sin θ Y` and `Y → cos θ Y − sin θ X`, so every term with an
X or Y on that qubit splits in two. With `t` T gates there are up to `2^t`
terms; identical strings are merged after each split. At the end,
`<0…0|P|0…0>` is ±1 if `P` contains only I and Z, and 0 otherwise.

The cost is polynomial in the number of qubits and exponential in the number
of non-Clifford gates. This is the Heisenberg-picture counterpart of a
"sum over stabilizer states" simulator, which splits the state instead of the
observable; both are exact and both are `O(2^t poly(n))`. It gives exact
expectation values and marginal distributions of a few qubits; it does not
produce full samples.

### Matrix product state

The state is written as a chain of tensors, one per qubit, with shape
`(χ_left, 2, χ_right)`. An amplitude is a product of matrices, one slice per
qubit. The bond dimension `χ` across a cut equals the Schmidt rank of the
state across that cut, so a product state has `χ = 1`, a GHZ state has
`χ = 2` at any size, and a generic state has `χ = 2^(n/2)` in the middle of
the chain.

A gate on neighbouring qubits contracts their two tensors, applies the 4×4
matrix and splits the result with an SVD (faer). Singular values beyond the
bond cap, or below a relative cutoff of `1e-14`, are dropped. The chain is
kept in mixed-canonical form (the orthogonality centre is moved with QR
decompositions), so each truncation is locally optimal and the discarded
weight is the actual error. The product of `(1 − discarded weight)` over all
truncations is reported as a fidelity estimate. Gates on distant qubits are
routed with SWAPs. Sampling moves the centre to the left end, after which the
conditional probabilities can be read off left to right in `O(n χ^2)` per
shot.

(The first version used nalgebra's SVD, which returned wrong singular values
for some complex 2×2 matrices; the cross-check tests caught it, and the MPS
now uses faer.)

## Benchmarks

All numbers were measured on the development machine: a VM with **4 vCPUs
(AMD EPYC-Rome, one thread per core, 16 MiB L3)** and 7.7 GB of RAM shared
with other services, built with `cargo build --release` for the default
x86-64 target (no `-C target-cpu=native`), Rust 1.93. Times are wall-clock
seconds from a single run with `qsim bench …`. Every benchmark run stayed
under 1.1 GB peak RSS.

### State vector: GHZ (H + n−1 CNOTs), then 1000 samples

Baseline: a plain numpy script simulating a 26-qubit GHZ state vector
(complex64, 0.5 GiB) took **11.2 s**. That number comes from an earlier run of
a separate numpy script and was not re-measured here, so treat the ratio as
approximate.

| n | precision | memory | alloc (s) | gates (s) | 1000 shots (s) | total (s) |
|---|---|---|---|---|---|---|
| 20 | f32 | 8.0 MiB | 0.001 | 0.005 | 0.001 | 0.006 |
| 22 | f32 | 32.0 MiB | 0.004 | 0.016 | 0.001 | 0.021 |
| 24 | f32 | 128.0 MiB | 0.014 | 0.074 | 0.005 | 0.092 |
| 25 | f32 | 256.0 MiB | 0.029 | 0.156 | 0.010 | 0.195 |
| **26** | **f32** | **512.0 MiB** | 0.054 | 0.299 | 0.017 | **0.370** |
| 20 | f64 | 16.0 MiB | 0.002 | 0.007 | 0.001 | 0.009 |
| 22 | f64 | 64.0 MiB | 0.008 | 0.031 | 0.002 | 0.041 |
| 24 | f64 | 256.0 MiB | 0.030 | 0.141 | 0.007 | 0.179 |
| **25** | **f64** | **512.0 MiB** | 0.049 | 0.267 | 0.012 | **0.329** |

The 26-qubit f32 GHZ state takes 0.37 s against numpy's 11.2 s, about 30×
faster. Time doubles per qubit, as expected.

QFT (n H gates, n(n−1)/2 controlled phases, n/2 SWAPs), f32:

| n | gates | time (s) | time per gate (ms) |
|---|---|---|---|
| 16 | 144 | 0.007 | 0.05 |
| 20 | 220 | 0.039 | 0.18 |
| 22 | 264 | 0.167 | 0.63 |
| 24 | 312 | 0.808 | 2.59 |
| 26 | 364 | 3.517 | 9.66 |

### Stabilizer tableau: GHZ, then measure every qubit

| n | tableau memory | prepare (s) | measure all (s) | total (s) | CHP qubit-by-qubit measure (s) |
|---|---|---|---|---|---|
| 1,000 | 512 KiB | 0.000 | 0.001 | 0.002 | 0.027 |
| 2,000 | 2.0 MiB | 0.001 | 0.002 | 0.003 | 0.190 |
| 5,000 | 12.2 MiB | 0.006 | 0.016 | 0.028 | 2.391 |
| 10,000 | 48.1 MiB | 0.025 | 0.047 | 0.072 | - |
| 20,000 | 191.4 MiB | 0.097 | 0.185 | 0.282 | - |
| 30,000 | 429.6 MiB | 0.211 | 0.405 | 0.616 | - |
| 40,000 | 762.9 MiB | 0.435 | 0.741 | 1.176 | - |
| **46,336** | **1023.8 MiB** | 0.533 | 0.931 | **1.464** | - |

46,336 qubits is the largest size under the 1 GiB tableau cap. Memory grows
as `n^2`; time grows a bit faster than `n^2` once the tableau no longer fits
in cache. The last column is the plain CHP procedure, measuring one qubit at a
time, which grows as `n^3` and was only run up to 5,000 qubits. For reference,
Stim's GHZ demo took 2.3 s at 20,000 qubits and crashed at 1,000,000 qubits
(a 125 GB tableau); that figure comes from a different harness, so it is not
a like-for-like comparison. Here, `Tableau::try_new(1_000_000)` returns an
error instead of allocating.

### Clifford+T: cost against the number of T gates

64 qubits, which a state vector cannot hold (it would need 128 EiB). Each
round is a random Clifford block of three layers (each layer: a random
single-qubit Clifford on every qubit, then 32 CNOT/CZ/SWAP gates on random
pairs) followed by one T gate on a random qubit. The circuit with `t + 4` T gates is the one with `t` T gates
plus four rounds at the start, and `<Z_0>` is computed exactly.

| T gates | gates total | Pauli terms (peak) | time (s) |
|---|---|---|---|
| 0 | 288 | 1 | 0.0004 |
| 4 | 1444 | 1 | 0.0003 |
| 8 | 2600 | 1 | 0.0003 |
| 12 | 3756 | 8 | 0.028 |
| 16 | 4912 | 48 | 0.067 |
| 20 | 6068 | 432 | 0.089 |
| 24 | 7224 | 2160 | 0.144 |
| 28 | 8380 | 10818 | 0.225 |
| 32 | 9536 | 54783 | 0.423 |
| 36 | 10692 | 277552 | 1.127 |
| 40 | 11848 | 1405743 | 4.120 |
| 44 | 13004 | over the 4,194,304-term limit | (aborted after 8.0 s) |

With no T gates the observable stays one Pauli string, as it would on the
tableau. The first few T gates do nothing because they sit near the end of the
circuit, outside the backward light cone of `Z_0`. Once the propagated
observable covers most of the qubits, each T gate multiplies the number of
terms by about 1.5 (an X or Y lands on the T qubit about half the time), and
the run time follows. The run aborts at 44 T gates when the term budget
(`DEFAULT_MAX_TERMS`, about 260 MB here) is exceeded.

The table above was measured with the original engine, which is now
`pauli_path::expectation_legacy`. `<Z_0>` is exactly 0 in every row, so it is a
cost benchmark only. `pauli_path::expectation` now uses the rotation-frame
engine (`pauli_frame`). That engine compiles the Cliffords away with a tableau,
leaving only `t` Pauli rotations. It then drops terms that provably cannot
contribute: their X part lies outside the span of the remaining rotation axes.
On this family, with fewer T gates than qubits, that removes `Z_0` before any
propagation. For a non-degenerate version (a stabilizer of the circuit's
Clifford skeleton, whose value is non-zero) the frontier at the same term budget
moves from 40 T gates (legacy) to 100. See `research/pauli.md` for the exactness
argument, the tests, and interleaved A/B timings
(`qsim bench clifford-t --observable stab --engine legacy|frame`).

### MPS: GHZ and random circuits

GHZ (bond cap 64, never reached):

| n | max bond | memory | time (s) |
|---|---|---|---|
| 100 | 2 | 12.4 KiB | 0.003 |
| 1,000 | 2 | 124.9 KiB | 0.023 |
| 10,000 | 2 | 1.2 MiB | 0.223 |

Random brickwork circuits: each layer applies random RY and RZ rotations to
every qubit, then CNOTs on alternating neighbour pairs. 24 qubits, bond cap
256:

| depth | max bond | fidelity estimate | memory | time (s) |
|---|---|---|---|---|
| 2 | 2 | 1.000000 | 2.9 KiB | 0.001 |
| 4 | 4 | 1.000000 | 10.6 KiB | 0.001 |
| 8 | 16 | 1.000000 | 138.6 KiB | 0.009 |
| 12 | 64 | 1.000000 | 1.6 MiB | 0.143 |
| 16 | 241 | 1.000000 | 12.2 MiB | 0.956 |
| 20 | 256 | 0.999994 | 18.7 MiB | 4.328 |
| 24 | 256 | 0.997527 | 18.7 MiB | 9.797 |
| 32 | 256 | 0.909861 | 18.7 MiB | 15.161 |

60 qubits, bond cap 32:

| depth | max bond | fidelity estimate | memory | time (s) |
|---|---|---|---|---|
| 4 | 4 | 1.000000 | 28.6 KiB | 0.003 |
| 8 | 16 | 1.000000 | 426.6 KiB | 0.019 |
| 16 | 32 | 0.734902 | 1.6 MiB | 0.382 |
| 32 | 32 | 0.000088 | 1.6 MiB | 1.269 |

The bond dimension doubles every two layers until it reaches the cap. After
that, memory stops growing but every gate throws away weight, and the
fidelity estimate decays. At 60 qubits and depth 32 the result has almost
nothing to do with the true state. A GHZ state on 10,000 qubits costs 1.2 MiB
because it never needs more than `χ = 2`.

## Tests

`cargo test` runs 61 tests, including one doctest; eight of them are proptest
properties with 64 generated cases each:

* **Cross-checks between backends**: random Clifford circuits give the same
  exact outcome distribution on the tableau and the state vector; every
  tableau stabilizer has expectation +1 on the state vector (which checks the
  sign bits); tableau and MPS sampling match state-vector probabilities; MPS
  amplitudes on random circuits with every gate type match the state vector
  (overlap 1 up to global phase); MPS measurement collapses the same way as
  the state vector; Pauli-path expectation values of random Pauli observables
  and 3-qubit marginals match the state vector; f32 and f64 agree.
* **Gate identities** on random 3-qubit states: HH = I, S² = Z, T² = S,
  HZH = X, the CNOT conjugation rules (X⊗I → X⊗X, I⊗Z → Z⊗Z, …), CZ from
  CNOT, SWAP from three CNOTs, and exactness of the Toffoli/RX/RY/CPhase
  decompositions; the same identities on the tableau.
* **Known states and algorithms**: Bell and GHZ states, Bernstein–Vazirani on
  all three sampling backends (30 bits on the tableau and MPS), Grover's
  success probability, QFT of basis states against the closed form, and
  Shor's algorithm factoring 15 and 21.
* **Properties (proptest)**: random circuits preserve the norm; a circuit
  followed by its inverse is the identity; stabilizer outcome probabilities
  are 0 or `2^-k`; the MPS without truncation is exact; repeated
  measurements agree; the Toffoli truth table.

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings` and `cargo test`.

## Limitations

* The state vector does no gate fusion or cache blocking: each gate is a
  separate pass over memory, so deep circuits on 24+ qubits are bandwidth
  bound (about 10 ms per controlled-phase at 26 qubits).
* The Pauli-path simulator gives expectation values and small marginals, not
  samples of all qubits, and its term budget is a hard limit.
* The tableau's Gaussian elimination is sequential, and a random measurement
  of one qubit costs `O(n^2 / 64)` word operations.
* The MPS routes long-range gates with SWAPs and uses a fixed bond cap.
  There is no TEBD-style gate grouping or adaptive truncation.
* Shor's algorithm uses permutation oracles for the controlled modular
  multiplications instead of building them from gates.
* Memory caps are compile-time constants chosen for a shared machine.

## Licence

MIT, see [LICENSE](LICENSE).
