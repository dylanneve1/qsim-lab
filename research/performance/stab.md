# Lab notebook: `stab` (stabilizer tableau speed)

Branch `exp/stab`. All timings: release build (portable, no `target-cpu=native`),
min over `REPS` runs, taken through the swarm's `bench.sh` flock on a **shared
4-vCPU AMD EPYC-Rome VM (AVX2, 7.7 GB RAM)** while other agents were compiling.
1-minute load average was **8–16** during every run (printed at the top of each
raw file). Absolute times are therefore noisy. The numbers to trust are the
**interleaved A/B ratios**: each repetition runs the frozen reference and then the
new tableau back to back, so both see the same load. Tables report the ratio of
the minimum times and the median of the per-repetition ratios. Raw outputs are in
`research/data/stab/`.

Reproduce:

```
cargo build --release --example stab_bench
B=target/release/examples/stab_bench
REPS=5 $B syndrome ab 5,11,21          # rotated surface-code syndrome rounds
REPS=5 $B mixed ab 100,500,1000        # random Clifford layers + measurements
REPS=5 $B ghz ab 1000,5000,10000,20000 # GHZ prep + Tableau::measure_all
REPS=5 $B ghzprep ab ...               # GHZ preparation only
STAB_TRACK=0 $B ghzprep ab ...         # same, sign tracking switched off
REPS=3 $B ghzseq ab 1000,2000,5000     # GHZ + qubit-by-qubit measurement
```

The `outcomes` column hashes every measurement outcome (same seeds). It reads
`same` in every row of every run.

## Baseline and profile

The original tableau (frozen as `stabilizer::reference::RefTableau`, which is not
to be optimised) stores the four `n x n` blocks bit-packed. It keeps them
**qubit-major** for gates and transposes them to **generator-major** for
measurements (lazy, in 64x64 blocks). The cost model per operation is:

| operation | reference cost |
|---|---|
| 1q/2q gate | `O(n/64)` words, after a full `O(n^2/64)` transpose if the last op was a measurement |
| deterministic measurement | full transpose (if needed) + up to `n` row products: `O(n^2/64)` |
| random measurement | transpose + `O(n)` row products: `O(n^2/64)` |

Syndrome extraction alternates "a few gates" with "a few measurements" in every
round, so every round pays two full-tableau transposes. Deterministic ancilla
measurements (the common case in a noiseless or low-noise memory experiment) also
pay `O(n^2/64)` each. Instrumented with `layout_switches()`: the reference does
2 transposes per round at d=21 (881 qubits), and these plus the deterministic
row products dominate. The previous agent's numbers were 97 ms per round, against
0.9 ms per round after the change.

## Change 1: inverse-tableau sign tracking, no layout switch (Stim-style)

*Hypothesis.* Read the other way round, the qubit-major lines are the rows of the
inverse tableau `T = C†`. If the signs of the inverse rows `C† X_q C` and
`C† Z_q C` (`sx`, `sz`) are also kept, then:

* a Z measurement of `a` is deterministic iff `T(Z_a)` has no X part
  (`xs[a] == 0`), and its outcome is the sign `sz[a]`. That is `O(n/64)`
  instead of `O(n^2/64)`.
* a random measurement can be done without transposing, by applying CNOTs, an H
  (or H_YZ) and an X to *generator indices* ("at the beginning of time", where
  they act trivially on `|0..0>`). That is one pass over the qubit-major lines
  (`fanout_cnot`).

So the layout never switches for gates or measurements. Only `sample`,
`measure_all` and `stabilizers` transpose. The cost is that S, CNOT and CZ now
also update `sx`/`sz`, which needs the phase of one or two Pauli products over
`n/64` words per gate.

*Result (interleaved A/B, `ab_syndrome_mixed_v5.txt`, load 12.8):*

| workload | ref min (s) | new min (s) | ref/new (mins) | median ratio |
|---|---|---|---|---|
| syndrome d=5 (49 q, 10 rounds) | 0.00272 | 0.00006 | 43 | 71 |
| syndrome d=11 (241 q, 10 rounds) | 0.04847 | 0.00074 | 65 | 68 |
| syndrome d=21 (881 q, 10 rounds) | 1.26424 | 0.00871 | **145** | **102** |
| mixed n=100, depth 50, 5 meas/layer | 0.01841 | 0.00146 | 12.6 | 13.4 |
| mixed n=500, 25 meas/layer | 0.20411 | 0.04370 | 4.7 | 2.6 |
| mixed n=1000, 50 meas/layer | 0.53927 | 0.34458 | 1.6 | 2.3 |
| mixed n=100, 20 meas/layer | 0.01645 | 0.00256 | 6.4 | 13.2 |
| mixed n=500, 100 meas/layer | 0.13307 | 0.09634 | 1.4 | 1.6 |

Qubit-by-qubit measurement of a GHZ state (`measure_all_sequential`, the README's
"CHP qubit-by-qubit" column), from `ab_ghzseq_v6.txt` at load 12.9:

| n | ref (s) | new (s) | ratio |
|---|---|---|---|
| 1000 | 0.0349 | 0.00003 | ~1000x |
| 2000 | 0.1835 | 0.00018 | ~1000x |
| 5000 | 2.4402 | 0.00051 | **~4800x** |

That is `O(n^3/64)` → `O(n^2/64)`: the first outcome is random (one pass), and
every later one is a sign lookup. The same applies to anything that measures
through `Circuit::run` with per-qubit `Measure` ops.

On mixed circuits with many *random* measurements at large `n`, the gain shrinks
to 1.4–2.3x. Each random measurement is still an `O(n^2/64)` pass
(`fanout_cnot` is serial, whereas the reference's row products use rayon). The
gain left comes from skipping the transposes and from the deterministic
measurements.

*Verdict:* **keep.** This is the main win.

## Accuracy gate

`tests/engines/stab_exact.rs` checks the new tableau against the frozen reference. The
two must agree exactly, not just in distribution: both consume exactly one
`random_bool(0.5)` per random outcome, and whether an outcome is random is a
property of the state. So with equal seeds:

* `identical_outcomes_and_states_on_random_programs`: n ∈ {1, 2, 3, 5, 8, 17, 31,
  63, 64, 65, 100, 127, 128, 129, 200, 300}, three measure/reset mixes each.
  Gates cover H, S, S†, X, Y, Z, Phase/Rz(kπ/2), CNOT, CZ, SWAP and CPhase(kπ). It
  checks identical outcome sequences and an identical **canonical stabilizer group
  with signs** (fully reduced row-echelon form, exact Pauli phases) at the end.
* `identical_outcomes_on_repeated_parity_checks`: repetition-code-style rounds
  with resets; rounds repeat.
* `identical_outcomes_with_sign_tracking_toggled`: the same, with tracking
  switched off and on at random points (see change 3).
* `untracked_measurements_switch_tracking_back_on`.
* `measure_all_lands_in_the_support`: forcing the drawn bits on the reference
  never contradicts a deterministic outcome.
* `prop_identical_outcomes` (proptest, 64 cases): n < 90, ≤ 400 steps, random
  measure/reset rates, toggles on half of the cases.

Mutation checks (run by hand, then reverted): dropping the phase term from CZ's
inverse-sign update, dropping the `|u&v|` term from the on-demand sign, or
dropping the prefix carry each make two of these tests fail.

## Change 2: making the per-gate sign tracking cheaper (GHZ regression)

*Observation.* Interleaved GHZ prep (H + n−1 CNOTs) was **2–2.7x slower** than the
reference at n = 1000–5000, where it is compute-bound in cache
(`ab_ghz_v0.txt`, and v1 rows in the run below). The portable build only has
SSE2. The reference CNOT loop vectorises (`movups`/`xorps`), but the
bit-sliced phase counter `Phase::mul` has a loop-carried `c1 → c2` dependency, so
LLVM left the whole fused CNOT loop scalar. (Checked with `objdump`: no packed
ops in `Tableau::cnot`.)

Tried:

1. *Four lane accumulators inside the fused update loop (v2):* still scalar, no
   gain. **Negative.**
2. *Separate read-only `product_phase` pass* over `chunks_exact(4)` with four
   independent accumulators, followed by the plain CHP update loop (v3). This
   vectorises. GHZ prep ref/new went from 0.39–0.45 → 0.53–0.57 at n=5000 and
   0.78–0.84 → 0.88–0.96 at n=20000 (process-interleaved v1/v3 runs, load 7.7).
3. *Update loop split per block (destabilizer block, then stabilizer block),* as
   the reference does: fewer concurrent memory streams (v5).

GHZ prep, `ab_ghzprep_v5.txt` (REPS=9, load 13.7):

| n | ref/new tracked (median) | ref/new untracked (median) |
|---|---|---|
| 1000 | 0.55 | 0.68 |
| 5000 | 0.86 | 0.96 |
| 10000 | 0.89 | 0.96 |
| 20000 | 0.88 | 0.96 |

(At n=1000 the whole prep takes 50 µs vs 70–100 µs, so per-call overhead
dominates.)

GHZ prep + `measure_all`, which is what the README benchmark measures
(`ab_ghz_v4.txt`, `ab_ghz46k_v5.txt`): 0.92–1.02 tracked, 0.96–1.17 untracked
for n = 5000–20000; 1.10 tracked and 1.02 untracked at n = 46,336. The tracked
regression on the end-to-end benchmark is at most about 8%, within noise at
these loads.

*Verdict:* keep v3 + v5. The regression is reduced from 2–2.7x to ~12% on prep
alone (tracked) and ~4% (untracked). It is negligible end to end.

## Change 3: optional sign tracking (`set_sign_tracking`)

Untracked mode: gates skip the inverse-sign phase work, so they cost the same as
plain CHP. A deterministic measurement recomputes only the one sign it needs
(`inverse_sign`). It conjugates `T(Z_a)` back through the forward tableau,
`Z_a = ± i^{|u&v|} (-1)^{u·rd+v·rs} Π_{i∈u} D_i Π_{i∈v} S_i`, and sums the
per-qubit phase of that ordered product in one qubit-major pass (`O(n^2/64)`,
still no transpose). Switching tracking back on recomputes all `2n` signs
(`O(n^3/64)`). That also happens automatically after `2n` untracked
deterministic measurements (ski rental), so a wrong choice costs at most about 2x.

*Why tracking is on by default, and why no online policy fixes GHZ:* the tableau
starts in `|0..0>`, where the signs are known for free. Re-enabling costs
`Θ(n^3/64)`, while the whole GHZ tracking overhead is only `Θ(n^2/64)`. Any
online rule that switched tracking off during a gate run would risk paying the
recompute (or `O(n^2/64)` per later deterministic measurement) for a saving
that is always smaller. Example: GHZ followed by per-qubit `Measure` ops is
~4800x faster *with* tracking. Only a caller who knows that no single-qubit
measurements follow (e.g. GHZ then `measure_all`) should switch it off. Lookahead
from `Circuit::run` could make that automatic, but it would touch the shared
`Simulator` trait, so it is left as a follow-up.

## Bug found: `Tableau::reset_qubit` on main forced outcome 0

Main's `reset_qubit` called `measure_with(a, Some(false))`, which *forces* a
random outcome to 0. That is not the reset channel. After `H(0) CNOT(0,1)`,
resetting qubit 0 must leave qubit 1 uniformly random (as the state-vector
backend does), but the forced version always leaves it in `|0>`. Fixed here to
measure-then-X, returning the outcome
(`reset_of_entangled_qubit_leaves_partner_random`). It is harmless for ancillas
whose outcome is already deterministic, but wrong for resets of entangled qubits.
The noise tests and the qec surface/repetition tests still pass. The fix
consumes one RNG bit per random reset, so seeded streams change. qec agent
notified.

## Pauli-frame sampling (`frame.rs`)

The timed-out agent left only a one-line placeholder. **Dropped** from this PR.
See SymPhase below for the many-shot sampling direction.

## Negative / neutral results

* The 4-lane accumulator fused into the update loop did not vectorise (no gain).
* Random measurements at large `n` remain `O(n^2/64)` passes. The new serial
  `fanout_cnot` beats the reference's rayon row products only 1.4–2.3x on
  measurement-dense random circuits at n = 500–1000. Parallelising
  `fanout_cnot` over qubit lines is a possible follow-up, but it is of limited
  value on a contended 4-vCPU box.
* No online on/off policy for sign tracking can beat "on" for GHZ-like circuits
  (argument above).

## Change 4: SymPhase-style symbolic-phase sampler (`stabilizer::symphase`)

*Hypothesis* (arXiv:2311.03906; Stim's frame trick, arXiv:2103.02202 §2.3).
With Pauli noise, a stabilizer simulation's X/Z structure is identical in
every shot; only the signs depend on the random choices. So the outcome
vector is affine over GF(2):

```
m = m_ref xor A v
```

The variables `v` are:
* fault variables (X/Y/Z flips; 2 bits per 1q depolarizing error; 4 bits per
  2q depolarizing error; measurement and reset flips), with their joint
  distributions;
* uniform *coins*, one per point where a qubit is in a known Z eigenstate
  (start, after each measure, after each reset). Applying Z there with
  probability ½ doesn't change the state, and it re-randomises what the
  reference sample fixed arbitrarily.

`A` comes from one pass of a **symbolic Pauli frame**. Per qubit, the frame's X
and Z components are bit sets over all variables. Clifford gates XOR/permute
them, faults and coins set bits, and a measurement row is the X bit set of the
measured qubit. Classically controlled Pauli gates XOR the controlling row into
the frame. Gate noise on a controlled gate is rejected, because a conditioned
fault is not affine. Then each batch of 64 shots costs:
* one random word per coin;
* geometric skipping over the (fault, shot) grid, so the cost is proportional to
  the faults that actually happen;
* one sparse GF(2) mat-vec, `nnz(A)` word XORs.

*Accuracy* (`tests/engines/symphase.rs`). Full outcome distributions are computed
**exactly** on both sides and compared to a relative 1e-9 (the tableau side
sums up to ~10^5 branch probabilities, so float rounding is ~1e-12). The
tableau side branches over every noise outcome and every random measurement
of `run_noisy`; the sampler side enumerates every variable assignment. The
comparison covers:
* 300 random small circuits (1–3 qubits, up to 15 ops: H/S/S†/Phase/CNOT/CZ/SWAP,
  measurements, resets, all flip/depolarizing channels, `c_if` Paulis), under
  gate, measurement and reset noise models;
* a 3-qubit parity-check circuit with 2-qubit depolarizing gate noise;
* targeted cases: H M H M re-randomisation, reset of an entangled qubit, both
  directions of the CZ frame rule;
* the detector/parity compilation.

On a d=3, 3-round surface code at p=0.01, the per-measurement marginals and
round-to-round parities of 20,000 bit-packed sampler shots agree with 5,000
`run_noisy` tableau shots within 5σ. Hand mutations (dropping the
post-measurement coin, dropping one direction of the CZ rule, not clearing the
frame on reset) are each caught. The first version of the random test used only
60 cases and missed two of these mutations. Raising it to 300 cases and adding
targeted circuits fixed that.

*Numbers* (`symphase_v2.txt`, load 12.2; `SurfaceCode::build_circuit`, rounds =
d, `NoiseModel::circuit_level(p, p)`, interleaved per repetition, min of 3).
"tableau" is shot-by-shot `run_noisy` on the **new** (already sped-up) tableau.

| d | qubits | meas | vars | nnz(A) | compile | tableau / shot | sampler / shot | speedup | detector nnz | detector sampler / shot |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 17 | 33 | 406 | 739 | <0.1 ms | 25 µs | 0.027 µs | 934x | 645 | 0.025 µs |
| 5 | 49 | 145 | 2166 | 6331 | 0.2 ms | 97 µs | 0.154 µs | 630x | 4027 | 0.125 µs |
| 7 | 97 | 385 | 6266 | 25103 | 0.6 ms | 342 µs | 0.495 µs | 691x | 12401 | 0.354 µs |
| 11 | 241 | 1441 | 25422 | 156619 | 4.7 ms | 1.63 ms | 2.66 µs | 613x | 53149 | 1.41 µs |
| 15 | 449 | 3585 | 65746 | 546391 | 23 ms | 5.67 ms | 8.17 µs | 695x | 140937 | 3.58 µs |
| 5 (p=0.01) | | | | | 0.2 ms | 96 µs | 0.385 µs | 249x | | 0.352 µs |
| 11 (p=0.01) | | | | | 4.7 ms | 1.65 ms | 5.22 µs | 315x | | 4.07 µs |

The compile cost is paid back after a handful of shots. Sampled ones per shot
agree between tableau and sampler (e.g. d=11: 413.6 vs 419.2 over 30 tableau
shots). `with_parities` (detector basis: each ancilla XORed with its previous
round) cuts `nnz` 3–4x and the per-shot time 1.1–2.3x. At p=0.01 the fault
sampling dominates instead of the mat-vec.

*Verdict:* **keep.** This is the right engine for the threshold sweeps on
`exp/qec`: exact circuit-level sampling with no hand-written DEM. The qec agent
had finished before this landed, so wiring it into `SurfaceCode` (and the
decoder's detector order) is a follow-up.

Possible follow-ups (not done):
* sort/blocked `A` for cache locality in `eval`;
* parallel batches with rayon;
* a dense 64x64 bit-transpose to hand per-shot detector lists to the decoder.
