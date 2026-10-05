# Pauli paths (Clifford+T): lab notebook

Branch `exp/pauli`. Topic: exact Pauli-path propagation (Heisenberg picture)
for `<0|U† O U|0>` with Clifford+T (and general Z-rotation) circuits. No
truncation: every result here is exact up to floating-point rounding (see the
caveat on `drop_below` below).

**Machine.** Shared 4-vCPU VM (AMD EPYC-Rome, AVX2, 7.7 GB). Other agents were
compiling and benchmarking at the same time. The 1-minute load average at the
start of the timed runs was **11–19**. Every timed run went through
`qsim-swarm/bench.sh` (flock), but the lock does not stop other agents'
compiles. Absolute times are therefore inflated and noisy. The speed claims
below are **ratios from interleaved A/B runs** (A, B, C, A, B, C, …, each run
in its own lock), reported as min-of-5 together with the per-pair spread.
Release build, portable (no `target-cpu=native`).

## 1. What `pauli_frame.rs` is

It is **not** Stim-style Pauli-frame sampling, so it stays in this topic and
was not handed to the stabilizer agent. It is a *rotation-frame* compiler for
Pauli propagation:

1. **Clifford absorption.** A Heisenberg tableau (the images of the `2n`
   generators under `P → C_k† P C_k`) pushes every Clifford gate to the end:
   `U = C · R_m ⋯ R_1` with `R_j = exp(-i θ_j Q_j / 2)`. Here `Q_j` is the T
   gate's `Z_a` conjugated by the Cliffords before it. The cost is
   `O(gates · n)` word operations, independent of the number of terms. The
   observable becomes `O' = C† O C`. Only `m` = (number of T/Rz gates)
   per-term passes remain. The legacy engine made one pass over all terms
   *per gate*, Cliffords included.
2. **Term store.** Fixed-width keys `[u64; W]` (W = 1, 2, 4, 8, so up to 512
   qubits) in 64 hash shards. A rotation is two embarrassingly parallel
   phases: emit `P·Q` into per-destination buffers, then merge. The legacy
   engine merges by sorting after every split.
3. Two **pruning rules** and one **rotation merge**, all exact (§2).

This is the Pauli-rotation / phase-polynomial view of Clifford+T, as used in
T-count optimisers and in PauliPropagation.jl. The pruning rules are a
Heisenberg-picture form of the known fact that a Clifford+T circuit with `t`
T gates "compresses" to `t` active qubits (Jozsa–Van den Nest;
Yoganathan–Jozsa–Strelchuk): `d_k` below is exactly the size of that active
register.

## 2. Exactness: why each rule cannot drop a contributing term

Notation: a Hermitian Pauli string is `P = i^{x·z} X^x Z^z` with
`x, z ∈ GF(2)^n`. `<0|P|0> = 1` if `x = 0` (for any `z`), and `0` otherwise.
After rotations `R_m … R_{k+1}` have been processed, the remaining value is
the linear functional `f_k(A) = <0| R_1† ⋯ R_k† A R_k ⋯ R_1 |0>`, applied to
the current operator `A = Σ c_P P`.

**Lemma (x-span pruning).** Let `W_k = span{x(Q_1), …, x(Q_k)}`. If
`x(P) ∉ W_k`, then `f_k(P) = 0`.

*Proof.* `R† P R` is either `P` (if `[P,Q] = 0`) or `cos θ P − i sin θ P Q`.
By induction, `R_1† ⋯ R_k† P R_k ⋯ R_1` is a linear combination of strings
`± P Q_{i_1} ⋯ Q_{i_r}`, with phases, for subsets `{i_1, …, i_r} ⊆ {1..k}`.
The x part of such a product is `x(P) + Σ x(Q_{i_s})`, which lies in the coset
`x(P) + W_k`. That coset does not contain 0 when `x(P) ∉ W_k`, so every string
has `x ≠ 0` and `<0|·|0> = 0`. ∎

`f_k` is linear, so removing `c_P P` from `A` leaves `f_k(A)` unchanged. This
holds *exactly* (not "up to small terms"), whatever the other terms are and
whatever later merges happen.

**Lemma (CNOT frame + z projection).** Let `V` be a CNOT network with linear
map `L` on `GF(2)^n`. Then `V|0> = |0>`, and `V X^x Z^z V† = X^{Lx} Z^{L^{-T} z}`
(no phase; the code tracks the ±1 that comes from re-normalising
`i^{x·z}`). Conjugating every axis and `O'` by `V` therefore leaves every
`f_k` value unchanged. The symplectic product is preserved,
`(Lx)·(L^{-T}z) = x·z`, so commutation relations and product phases are
preserved too. The `W_k` are nested, so a single `L` can map `W_k` onto
`span{e_1, …, e_{d_k}}` for every `k` at once (with `d_k = dim W_k`): take a
basis of `W_m` built in axis order, complete it, and invert. In that frame:

* the pruning test becomes a mask test (`x` has no bit at position `≥ d_k`);
* qubits `≥ d_k` are inert from stage `k` on. Every surviving term and every
  remaining axis has `x = 0` there. So the z bits there never enter a
  commutation test (`x(P)·z(Q) + z(P)·x(Q)`), never enter a product phase
  (each phase term needs an x bit at the same position), and never enter the
  final value (`<0|Z^z|0> = 1` for every `z`). Hence `f_k(P) = f_k(P')`,
  where `P'` is `P` with those z bits cleared. Replacing `P` by `P'` and
  merging the resulting duplicates is exact by linearity. ∎

**Rotation merging.** `exp(-iθ₂Q/2)` can be moved next to an earlier
`exp(-iθ₁Q/2)` when every rotation in between commutes with `Q`. The two
then combine into `exp(-i(θ₁+θ₂)Q/2)`. The scan stops at the first
anticommuting rotation. A total angle that is ≡ 0 mod 2π is a global phase
and is removed. Angles that are multiples of π/2 get exact `cos/sin`.

**Caveat: `drop_below = 1e-14` (pre-existing, also in legacy).** Terms with
`|c| ≤ 1e-14` are deleted after a merge. This threshold is absolute, and the
coefficients of T-circuit paths scale like `2^{-b/2}` after `b` branchings, so
in principle it becomes a truncation once `b ≳ 90`. I checked the benchmark
range with an exact variant (`--engine frame-nodrop`, `drop_below = 0`): at
`t = 84…102` the values and peak term counts are identical
(`research/data/pauli/stab_drop_vs_nodrop.md`), because pruning keeps the
number of branchings per path small. *Recommendation:* make the threshold
relative, or 0, before anyone runs `t ≫ 100`. Speed was not distinguishable at
this load.

### Tests (all in `tests/engines/pauli_frame.rs`; plus the existing suites)

Every test runs all 16 combinations of `prune × merge × parallel × fuse` and
compares them against the state vector (tolerance 1e-9) and against the legacy
engine (1e-12).

* `frame_matches_statevector_on_skeleton_stabilizers` (new). `n = 4…14`,
  `t ∈ {n/2, n, 2n, 3n}` (capped at 30). The observables are `K Z_S K†`, where
  `K` is the circuit's Clifford skeleton (T gates removed) and `S` is a random
  set. In the rotation frame these become `Z_S`, so their values are
  generically non-zero: **89/108 (82 %) non-zero**. Pruning really fires:
  39,924 terms are pruned across the run, and the peak drops below legacy's in
  some cases.
* `benchmark_family_matches_statevector` (new). The *exact* circuit family of
  `qsim bench clifford-t` (`bench::clifford_t_family`, same seed), at
  `n = 8…14` and `t ∈ {n/2, n, 2n, 3n}`, for both benchmark observables.
  Non-zero values: `<Z_0>` 5/16, stab 15/16.
* `frame_matches_legacy_at_64_qubits_with_nonzero_values` (new). `n = 64`,
  `t ∈ {8, 16, 20, 24}`, skeleton stabilizers with random `S`, all 16 option
  combinations against legacy (1e-12). At least 6/8 values are non-zero.
* `pruned_observable_has_exactly_zero_value` (new). Whenever the observable is
  pruned before any propagation (peak 0), the state vector gives exactly 0.
* Pre-existing, from the previous agent: universal random circuits, Clifford+T
  rounds, Toffoli networks, multi-term observables, wide registers
  (70/130/200 qubits, W = 2/4), and the adder.

**Mutation testing (do the tests catch a wrong rule?).** I introduced three
deliberate bugs, one at a time, ran `cargo test --test pauli_frame`, and then
reverted:

| mutation | tests failing (of 11) |
|---|---|
| x-span off by one (`d_k` computed *without* axis `k`, i.e. pruning against `W_{k-1}`) | 9 |
| z projection clears one bit too many (`d_k − 1`) | 9 |
| rotation merge ignores anticommuting rotations in between | 4 |

The new skeleton-stabilizer test catches the first two. It does not catch the
third: random circuits never repeat an axis. The Toffoli and universal-circuit
tests do catch it. Note that the pre-existing wide-register test
(`frame_matches_legacy_on_wide_registers`) passes under **all three**
mutations: its random low-weight observables have value 0, so it could not see
a broken pruning rule. That is the degeneracy the brief warned about.

## 3. The README benchmark is degenerate

`qsim bench clifford-t` (64 qubits, rounds of a depth-3 random Clifford block
followed by T on a random qubit, nested in `t`) reports `<Z_0>`, which is
exactly 0 in every row. The lemma says why. `O' = C†Z_0C` is essentially a
random Pauli, and `x(O') ∈ W_t` (with `dim W_t ≤ t`) has probability about
`2^{t−64}`. For `t < 64` the frame engine therefore drops the observable at
stage 0 (peak terms 0, a few ms; `research/data/pauli/z0_frame_frontier.md`).
That result is correct (§2) but says nothing about speed.

**Non-degenerate version: `--observable stab`.** The observable is
`K Z_0 K†`, where `K` is the Clifford skeleton of the circuit (a stabilizer of
`K|0>`). Its frame image is `Z_0`, which is never pruned at stage 0. Its value
is non-zero on every row of the table below (it decays roughly like
`2^{−(#anticommuting)/2}`). All three engines print identical values wherever
more than one of them ran.

## 4. Numbers

### 4a. Representation/merging win, without pruning (verifies claim (a))

Interleaved A/B, 5 reps, min-of-5. Per-pair ratios are in brackets.

| benchmark | legacy | frame-noprune | ratio | frame (pruned) | load |
|---|---|---|---|---|---|
| z0, t = 36 (277,552 peak terms) | 4.08 s | 0.0625 s | **65×** [47–117] | 0.010 s (peak 0, degenerate) | 16–19 |
| stab, t = 36 (1,012,191 peak terms) | 9.56 s | 0.153 s | **63×** [61–118] | 0.0095 s (peak 1) | 12–15 |
| adder 16 bits, Z_cout (327,678 peak) | 0.256 s | 0.094 s | **2.7×** | 0.0148 s (peak 4), **17×** | 11–12 |

Raw output: `research/data/pauli/ab_z0_t36.txt`, `ab_stab_t36.txt`,
`ab_adder16.txt`.

The "16.7 s → 0.16–0.21 s" from the previous agent's log was a loaded legacy
run. The README's unloaded legacy time at t = 36 is 1.13 s. Interleaved, the
**ratio is about 60×** on the random family. Peak term counts are identical,
so this is entirely representation: Cliffords cost `O(n)` per gate in the
tableau instead of a pass over ~10^5–10^6 terms, and hash-shard merging
replaces sort-merging. On the adder the ratio is only 2.7×, because the
circuit has fewer Clifford gates per T gate.

### 4b. T-count frontier (stab, 64 qubits, 4,194,304-term budget)

Full tables: `stab_legacy_noprune_frontier.md` (single runs, load 15–18) and
`stab_frame_frontier.md` (min of 3, load 14.5).

| T gates | legacy peak / time | frame-noprune peak / time | frame peak / time | value |
|---|---|---|---|---|
| 24 | 2,252 / 0.136 s | 2,252 / 0.0065 s | 1 / 0.0145 s | +4.4194e-2 |
| 32 | 419,359 / 3.53 s | 419,359 / 0.077 s | 1 / 0.019 s | +7.8125e-3 |
| 36 | 1,012,191 / 7.50 s | 1,012,191 / 0.170 s | 1 / 0.023 s | +2.7621e-3 |
| 40 | 864,388 / 6.77 s | 864,388 / 0.127 s | 1 / 0.025 s | +7.8125e-3 |
| 44 | aborted (6.2 M) | aborted (6.2 M) | 1 / 0.031 s | +1.9531e-3 |
| 64 | – | – | 2 / 0.017 s | +2.1579e-5 |
| 80 | – | – | 48 / 0.045 s | +6.7435e-7 |
| 92 | – | – | 25,318 / 0.087 s | +1.3487e-6 |
| 96 | – | – | 127,927 / 0.128 s | +1.4901e-8 |
| 100 | – | – | 399,308 / 0.245 s | +9.5367e-7 |
| 104 | – | – | aborted (4.2 M) | – |

**Frontier: 40 → 100 T gates** at the same term budget. The cause is
structural. For `t < n`, a branch `P·Q_k` acquires `x(Q_k)`, which is
generically independent of the earlier axes, so it is pruned at once; the
value is then a product of `cos θ` factors (peak 1). Real branching only starts
once `t > n` saturates the span, so the cost is about
`2^{c·(t−n)}` instead of `2^{c·t}`. The gain is specific to "many qubits,
T-count around n". For `t ≫ n` the exponent is unchanged. On the adder,
`Z_b0 Z_b_top` (value 0 by symmetry) is not helped at all: 32,768 peak terms at
8 bits on every engine, and it aborts at 16 bits.

### 4c. Per-term throughput and threads 1 → 4

`term visits` = Σ over rotations of the number of live terms (new
`PathStats::term_visits`). Interleaved, min of 3 (`threads.txt`, load ≈ 14):

| run | visits | 1 thread | 2 threads | 4 threads | 1→4 |
|---|---|---|---|---|---|
| frame-noprune, stab t = 40 | 1.73 M | 0.100 s (17 M visits/s) | 0.081 s | 0.079 s | 1.27× |
| frame, stab t = 100 | 2.40 M | 0.156 s (15 M visits/s) | 0.131 s | 0.117 s | 1.33× |

These times include compilation (the tableau over ~12–29 k gates). Thread
scaling **could not be measured honestly**: the load average was ~14 on 4
vCPUs, so there were no idle cores to scale onto. Re-run this on a quiet
machine before drawing a conclusion. For comparison, the legacy engine does
2.0 M rotation visits at stab t = 36 in 9.56 s, but its time is dominated by
the 10,692 Clifford passes over all terms, so a per-visit rate is not
meaningful for it.

## 5. Negative results / not done

* **Rotation merging** does nothing on the random family (axes never repeat)
  or on the Cuccaro adder (peak terms are identical with and without it). It
  matters only for circuits with cancelling T structure, e.g. `CCX·CCX`, which
  has a unit test.
* **Fusion** (rotate + project in one pass) was not separately timed at a
  meaningful load; it is exact and tested in all option combinations.
* **Thread scaling**: inconclusive (§4c).
* **CAMPS** (Clifford-augmented MPS, arXiv:2412.17209): not attempted. The
  rotation frame computed here (the axes `Q_j` and the final Clifford `C`) is
  exactly the input CAMPS needs, so `compile()` is a natural starting point
  for a CAMPS backend.
* Truncation (Pauli weight / small coefficients) is lossy and rejected.

## 6. Commands

```
cargo build --release
B=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh
$B ./target/release/qsim bench clifford-t --observable stab --engine frame --max-t 104 --step 4 --repeat 3 --time-limit 20
$B ./target/release/qsim bench clifford-t --observable stab --engine legacy --max-t 64 --step 4 --time-limit 15
research/data/pauli/ab.sh 5 stab 36 legacy frame-noprune frame     # interleaved A/B
RAYON_NUM_THREADS=1 research/data/pauli/ab.sh 1 stab 100 frame
$B ./target/release/qsim bench adder --engine frame --bits 2,4,8,16
```

Engine names: `legacy`, `frame`, plus `frame-` variants containing `noprune`,
`nomerge`, `serial`, `nofuse` or `nodrop`.

## 7. Recommendation

Merge. `pauli_path::expectation` now goes through the frame engine. The legacy
engine stays as `expectation_legacy`, the reference in the cross-check tests.
Without pruning the win is about 60× on the 64-qubit random family. With
pruning the frontier moves from 40 to 100 T gates on the non-degenerate
benchmark. The README's `<Z_0>` table should be read as a cost benchmark only.
Follow-ups: a relative `drop_below`, thread scaling on a quiet machine, and
CAMPS on top of `compile()`.
