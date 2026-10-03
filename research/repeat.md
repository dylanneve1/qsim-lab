# Exact exploitation of repeated blocks (`compile::repeat`)

Owner request (1 Oct): "turn the circuits into graphs and optimize repeating
parts." The circuit DAG IR (`src/dag.rs`, `research/dag.md`) already exists.
This experiment looks for `circuit = prefix · B^r · suffix` (also nested, also
with changing angles) and simulates `B^r` faster than `r` copies, exactly.

Code: `src/compile/repeat/{mod,cliff,exec,workloads}.rs`, plan-reuse API in
`src/blocked.rs` (`compile_kops` / `run_compiled`), pipeline option
`pipeline::simulate_with(.., &SimOptions { repeat: Some(..) })` (default off;
`simulate()` is unchanged and a test pins that). Tests: `tests/repeat.rs`
(20 tests, incl. 5 proptests). Bench: `examples/repeat_bench.rs`.
Raw data: `research/data/repeat/`.

Machine: shared 4 vCPU AMD EPYC-Rome VM, other agents running. Every timed run
went through `bench.sh`, with variants interleaved and min of 3 reported.
The load average at the start of each run is the first line of each data file
(2.9–5.4, one run at 8; compile jobs of other agents run at nice 19).
Timings are therefore noisy at the ±10% level; the big ratios below are not.

## 1. Detecting repetition

Two detectors run and the one that saves more gates wins:

1. **As written**: rolling hash of each op, tandem repeats `h[i] == h[i+p]`
   for every period `p <= 4096` (cost `O(N · pmax)`, capped), greedy choice by
   covered ops, then recursion into the body for nested repeats. Every
   candidate is re-verified op by op, so a hash collision cannot give a wrong
   block.
2. **Canonical layers**: each maximal unitary run is cut into ASAP layers and
   the gates in a layer are sorted by `(qubits, kind)`; measurements, resets
   and noise are barriers. The layer sequence is a topological order of the
   same dependency DAG, so a tandem repeat of layers is a real repeat of the
   circuit and commuting reorderings of a block match.

A second pass over the plain stretches matches **shapes** (same gates,
angles ignored): copies with equal angles become an ordinary repeat, the rest
a `Param` repeat that stores the angle list of every copy (QAOA).

Coverage (`research/data/repeat/detect.txt`; `covered` = gates inside a repeat,
`saved` = gates a simulator can skip, i.e. all copies but the first):

| circuit | gates | as written | canonical layers | + parameterised |
|---|---|---|---|---|
| Trotter TFIM n=12, 1000 steps | 45,012 | 99.97% | 99.97% | 99.97% |
| QAOA ring n=12, p=20 (angles differ) | 972 | 0% | 0% | **98.8%** (1 param repeat) |
| Grover n=7 (+5 ancillas), 200 its | 12,007 | 99.94% | 99.94% | 99.94% |
| rep-code QEC d=5, 1000 rounds | 8,005 gates (16,006 ops) | 99.84% | 99.84% | 99.84% |
| rep-code QEC d=9, 1000 rounds | 16,009 | 99.84% | 99.84% | 99.84% |
| brickwork n=20, B (4 layers) ×40 | 7,920 | 100% | 100% | 100% |
| random brickwork n=20, depth 200 | 9,900 | 0% | 0% | 100% (**parameterised**: the 2-layer shape is the same, only the random angles differ) |
| Trotter n=8, 200 steps, **each copy a random topological order** | 5,800 | **0%** | **98.5%** | 98.5% |

* The shuffled Trotter row is the point of the layer form: the as-written
  detector finds nothing, the layer form finds everything.
* Detection time: 0.13 s for 45k gates (as written), 0.0004–0.17 s for the
  others. The layered variant costs ~2× the as-written one (it sorts and
  hashes layers too).
* `to_circuit()` of every detected program is checked equal to the original
  circuit by state vector (<1e-12).
* The random-brickwork row shows that "parameterised repeat" is cheap to
  find but only useful if the executor can exploit it (below: it can't yet).

## 2. Exact fast paths

### (a) Clifford `B`: symplectic map power (`cliff.rs`)
`CliffordMap` = images of `X_j, Z_j` as `i^k X^x Z^z` bit vectors with signs
(`k` mod 4 keeps `Y` and signs exact). Gate tables are *derived numerically from
the gate matrices*, not typed in. `then` composes in `O(n^3/64)`; `pow` squares
`log r` times. `synthesize()` reduces the map to the identity column by
column (H, S, Sx, X, Z, CNOT, SWAP), giving an `O(n^2)`-gate circuit equal to
`B^r` **up to a global phase**.

*Exactness:* proptest over random Clifford bodies (n ≤ 6, up to 40 gates,
r ≤ 70, 48 cases): `M^r` equals the map of `r` literal copies **including all
signs**; the synthesised circuit has the same map; its state equals the
gate-by-gate state up to a global phase to <1e-12. Against the CHP tableau
(`Tableau`) with n = 20, 64, 70 and r = 1000, 513, 77: canonical
(row-reduced, signed) stabilizer groups are equal.

*Speed* (`clifford.csv`, unitary random Clifford block of 4n gates, time to
reach the final CHP tableau of `|0..0>`):

| n | r | plain `r·|B|` | power + synth + apply |
|---|---|---|---|
| 50 | 1e3 | 14.5 ms | 4.7 ms |
| 50 | 1e4 | 141 ms | 4.6 ms |
| 50 | 1e5 | 1.46 s | 5.1 ms |
| 50 | 1e6 | (≈14.5 s, extrapolated) | 5.2 ms |
| 200 | 1e3 | 73.9 ms | **190 ms (loses)** |
| 200 | 1e4 | 795 ms | 196 ms |
| 200 | 1e6 | (≈80 s, extrapolated) | 239 ms |

The cost is dominated by the fixed synthesis (`O(n²)` gates × `2n` rows, ~0.19 s
at n = 200), so power-by-squaring only pays when `r·|B|` exceeds that: the
crossover is r ≈ 2500 at n = 200 and r ≈ 300 at n = 50. The pipeline uses it
only when `r·|B| > 25 (k² + 8)` gates (`k` = support).
The squaring itself is negligible, so the asymptotic claim `O(n^3 log r)` holds;
the constant is in the synthesis, which I did not optimise.

*Amplitude requests never use it* (the global phase of `B^r` is unknown from the
map). `simulate_with(Request::Amplitudes)` goes to the dense executor.

### (a') Clifford rounds **with measurements**: steady-state skipping
A QEC memory round is not unitary (measure + reset), so there is no map to
exponentiate. Instead (`cliff::sample_program`): run copy 1..4 of the repeat on the
CHP tableau; if a copy had **only deterministic measurements** and left the
stabilizer group (row-reduced, signs included) **unchanged**, every further copy
is identical: no RNG is drawn, the state does not move, and the outcome block of
that copy is repeated for the remaining `r - k` copies. This is exact
by construction, never skips a round with a random outcome, and consumes the RNG
stream exactly like `Circuit::run` on a `Tableau`.

*Exactness:* bit-identical outcomes **and identical RNG state afterwards** versus
`Circuit::run` on a `Tableau` for the same seed: repetition code with `|0..0>` and
with `|+..+>` start (random first round), 1–40 rounds, 6 seeds each; a 300-round
`Program`; 30 random Clifford measuring/resetting blocks repeated 12× where
the pass must *not* skip (never wrong).

*Speed* (`qec.csv`, noiseless repetition code with explicit repeat node, 1 shot;
plain = the same ops streamed onto the tableau without materialising 10^6 rounds;
the repeat path still has to write the `r·(d-1)` outcome bits):

| code | rounds | plain | repeat | speedup |
|---|---|---|---|---|
| d=5, 9 qubits | 1e3 | 0.60 ms | 0.12 ms | 5× |
| d=5 | 1e4 | 6.3 ms | 0.21 ms | 30× |
| d=5 | 1e5 | 69.5 ms | 0.66 ms | 105× |
| d=5 | 1e6 | 639 ms | 4.3 ms | **147×** |
| d=9, 17 qubits | 1e3 | 1.20 ms | 0.13 ms | 9× |
| d=9 | 1e6 | 1.26 s | 4.0 ms | **317×** |

**Limits, stated honestly:** this needs a *noiseless* Clifford round. With noise
channels in the body (`XFlip`, `Depolarize*`) the round is random, `is_clifford_program` is
false and the ordinary path runs, no speedup. Noisy QEC is what the Pauli-frame
sampler is for; this pass does nothing for it. The steady-state check also
gives up after 4 non-steady copies (period-2 syndromes etc. are not caught).

### (b) `B` on `k ≤ 8` qubits: `U^r` by squaring (`exec.rs`)
Build the `2^k × 2^k` unitary (gates applied to the maximally entangled
state), `U^r` by repeated squaring (naive complex matmul), apply once to the
state at any positions of the `k` qubits. A cost model picks it; `max_small_k`
= 8 default.

*Accuracy vs gate-by-gate*, Trotter step on 6 qubits (`accuracy.txt`; the
reference itself accumulates rounding error):

| r | max amplitude difference | norm defect of `U^r|ψ>` |
|---|---|---|
| 10 | 9.4e-16 | 0 |
| 100 | 1.3e-14 | 4.7e-15 |
| 1000 | 7.4e-14 | 5.3e-14 |
| 10,000 | 6.7e-13 | 5.2e-13 |

Grows ~linearly in r (the squarings double the relative error each time, like
r sequential steps do). It stays below the 1e-10 gate for r ≤ 1e4; extrapolating
the slope gives ~1e-10 at r ≈ 1.5e5 for this block. Documented bound: use for
r ≤ 1e4 (the bound is block dependent; the unitarity defect printed is the
quantity to watch).

*Speed* (`trotter.csv`; baseline `plain-batch` is what the pipeline does today,
`apply_gates` over the whole circuit with the blocked executor; `gatewise` is
`apply_circuit`, one pass per gate):

| TFIM chain n | r | gatewise | plain-batch | per-copy `apply_gates` | repeat |
|---|---|---|---|---|---|
| 6 | 1e3 | 2.6 ms | 10.3 ms | 7.9 ms | 2.8 ms (cost model: no (b)) |
| 6 | 1e4 | 25.2 ms | 111 ms | 79.9 ms | **3.9 ms** (28× vs batch, 6.5× vs gatewise) |
| 10 | 1e4 | 355 ms | 392 ms | 394 ms | 323 ms (k=10 > 8: plan reuse) |
| 14 | 1e3 | 840 ms | 343 ms | 441 ms | 414 ms (plan reuse; loses to batch) |

So (b) wins big on few qubits with large r, and only there. At r = 1e3 and
n = 6 the matmul chain costs about what 1000 gate-by-gate steps cost.

### (c) diagonal `B`: fold into one phase layer
Phase polynomial of `B` times `r` (angles multiplied, reduced mod 2π in
double-double with fma, so `r·a` with `r = 1e6` loses nothing). Global phase
from `Rz` is tracked exactly, so amplitudes are right including the phase.
Param repeats whose copies are all diagonal sum their angles. Test: proptest
(n ≤ 7, ≤ 30 gates, r ≤ 200) amplitudes <1e-12 including phase; r = 1e5: 3.4e-12 to
gate-by-gate (whose error grows with r).

| n | r | gatewise | plain-batch | per-copy | repeat |
|---|---|---|---|---|---|
| 14 | 3e4 | 8.41 s | 0.50 s | 3.47 s | **0.31 ms** |
| 20 | 200 | 2.84 s | 15.0 ms | 0.56 s | **9.0 ms** |

Obvious and exact, but only helps for pure phase layers repeated back to back
(the blocked executor already merges diagonal terms inside one batch, which
is why `plain-batch` is only 1.7× behind at n = 20).

### (d) wide `B`: compile the blocked plan once (`blocked::compile_kops`)
*Negative result.* Compile time (fuse + stage plan + per-stage prepare) is
**38–73 µs** against 35–313 ms to *run* one copy (`brickwork.csv`,
`compile-vs-run`): 0.1% of the run time, so reuse saves nothing measurable.

| n | depth | r | gatewise | plain-batch | per-copy | repeat (reuse) |
|---|---|---|---|---|---|---|
| 20 | 4 | 20 | 2.17 s | 0.733 s | 0.721 s | 0.705 s |
| 22 | 4 | 10 | 5.54 s | 1.531 s | 1.588 s | 1.538 s |
| 22 | 8 | 6 | 6.89 s | 1.835 s | 1.859 s | 1.796 s |

Within noise of each other. Grover (n=7+5 ancillas, 200 iterations): batch 12.9 ms,
repeat 12.6 ms; (n=9+7, 100 its): 100 ms vs 115 ms (reuse slightly loses).
Plan reuse could only matter for circuits whose per-copy compile cost is
non-trivial (tiny registers, huge blocks); the blocked executor's compile is
already linear and cheap.

(The first run of this benchmark used `StateVector::apply_circuit` as "plain",
which is gate-by-gate, not the pipeline's blocked batch, and showed a
fake 3× win for reuse. That column is kept as `gatewise`; the honest baseline
is `plain-batch`. Worth knowing: `apply_circuit` is 3–4× slower than
`apply_gates`.)

### QAOA / parameterised repeats
Detection works (98.8% covered, one `Param` repeat) but the executor has no
parameter-rebinding fast path, so it applies all copies as one batch: no
gain (n=18, p=20: batch 90 ms, canonical order 89 ms, repeat path 103 ms
including 2.7 ms detection; n=12: 2.9 vs 4.3 ms, slightly slower). The structure
(stages) could be reused with new angles only if fusion decisions did not depend
on angle values (identity detection in `fuse_1q`); not attempted.

## 3. Wiring
* `compile::repeat::{detect, Program, rewrite, cliff, exec, workloads}`.
* `rewrite(&Program, allow_clifford)` turns diagonal repeats into folded
  `Phase`/`CPhase` gates (+ exact global phase) and Clifford repeats into the
  synthesised power; other repeats are expanded. The result is an ordinary
  `Circuit`, so the existing compile plan (peephole, tableau sampler, Pauli
  paths ...) runs on it unchanged.
* `pipeline::simulate_with(circuit, request, budget, &SimOptions{ repeat: Some(RepeatOptions{..}) })`
  - `Samples`, terminal measurements: rewrite, then the normal plan
    (distribution tested equal to `exact_outcome_distribution` of the original
    to <1e-12); if a repeat needs the dense engine (neither diagonal nor
    Clifford) and n ≤ 24: dense executor + `sample`.
  - `Samples`, mid-circuit measurement, Clifford only: steady-state sampler
    (bit-identical to the plain `Circuit::run` per-shot loop for the same seed).
  - `Amplitudes`: dense executor (diag fold / `2^k` power / plan reuse), exact
    phase; never the Clifford power.
  - `Expectation`: rewrite then the normal path.
  - No useful repeat (`saved_gates < min_saved_gates`, default 64), noisy or
    classically controlled circuits, or no dense room: ordinary `simulate`.
* Default off; `simulate()` is `simulate_with(.., SimOptions::default())`.

Caveat: samples from the steady-state sampler equal the plain *tableau
per-shot* loop bit for bit, but the ordinary `simulate` splits mid-circuit-measurement circuits into components and
consumes the RNG in another order, so seeded outputs of `simulate` and
`simulate_with` agree in distribution, not bit for bit, for random outcomes
(for deterministic circuits like the noiseless repetition code on `|0..0>` the
outcomes are identical).

## 4. Verification summary (`tests/repeat.rs`, all green)
* detection: Trotter, QAOA (param), Grover, QEC, nested repeats, shuffled
  copies; `to_circuit()` equals the original (state vector <1e-12);
* (a) proptest vs gate-by-gate (<1e-12 up to phase) and map equality with signs;
  CHP tableau equality at n = 20/64/70, r up to 1000; gate tables vs matrices;
* (a') bit-identical to `Circuit::run` incl. RNG state;
* (b)/(d) proptest: `2^k` power, plan reuse and plain executor all within 1e-12
  of gate-by-gate for n ≤ 8, r ≤ 40; error-vs-r table above;
* (c) proptest incl. global phase <1e-12;
* pipeline: amplitudes with repeat = without (<1e-12) on Trotter, QAOA, Grover,
  diag; exact outcome distributions of the rewritten circuit; default
  options give an identical `Simulation`;
* full `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`
  pass. (The generic `run_preflight` lane is for talon checkouts and does not
  exist in this repo.)

## 5. Prior art
* **Stim** `REPEAT n { ... }` blocks keep repetition in the circuit IR; its
  flattening is avoided in the frame/reference samplers, and the detector error
  model also keeps `repeat` blocks. Stim does *not* raise a Clifford block to
  a power; it simulates `n` rounds, relying on speed per round. The
  steady-state skip in (a') is not in Stim.
* **Tableau exponentiation**: a Clifford is its symplectic matrix over GF(2)
  plus a sign vector (Aaronson–Gottesman; Dehaene–De Moor 2003 for the
  group structure; Maslov–Roetteler and Bravyi–Maslov for `O(n^2)` synthesis);
  powers of a Clifford by squaring are classical group arithmetic. The
  synthesis here is a plain column reduction, not depth/size-optimal.
* **Repeated-squaring unitaries** for Trotter/Hamiltonian simulation on few
  qubits is standard ("matrix power instead of time stepping"); the error
  analysis above is empirical.
* **Phase polynomials** (Amy–Maslov–Mosca; Gosset et al.) are the general form
  of diagonal folding.
* Tandem-repeat finding on a hashed op sequence is the classical
  Main–Lorentz / run-length idea; the canonical layer form is the standard
  ASAP layering.

## 6. What is not done / next
* Parameterised repeat executor (rebind angles into a cached plan) - QAOA
  shows no gain today.
* Tableau steady-state check only detects period 1 (state fixed by one round).
* Clifford synthesis is `O(n^2)` gates with a large constant; lowering the
  crossover (r ≈ 2500 at n = 200) needs a cheaper synthesis or applying the
  map to the tableau rows directly instead of via gates.
* `2^k` power uses naive matmul; `k ≤ 8`. A faer matmul would allow `k ≈ 10-11`.
* A noisy QEC round (with `XFlip`/`Depolarize`) is untouched.
* The merge recommendation: (a), (a') and (c) are exact and cheap. Merge the
  detector and the options struct; keep (b) behind its documented `r ≤ 1e4`
  bound; (d) can be dropped or kept as an API (`compile_kops`) with no
  claimed speedup.

## 7. Audit, round 4 (3 Oct 2026, exp/repeat-r4)
Rebased onto main fb30f56 (clean textual rebase; one semantic fix:
`run_stage` gained a `simd` argument on main, so `CompiledKOps` now records
`cfg.simd && simd_available()` like the other executors).

Independent differential fuzz `tests/audit_repeat.rs` (naive reference SV of
`tests/audit_common`, branch-tree comparison in `tests/audit_r4/`):
`run_dense` on hand-built programs (reps 0/1/2/3/7/64/513/4097/100003,
nested repeats, `Param` nodes, diagonal / Clifford / general bodies, all six
`ExecOptions` paths forced), `detect -> to_circuit` and `rewrite` with
measurements, resets, classical control and Pauli flips inside blocks (the
whole instrument is compared, every branch's unnormalised state), and
`simulate_with` amplitudes / expectations / samples (exact support + 6σ),
including steady-state QEC-like rounds. Mutation check: four deliberately
broken variants (steady-state skip ignoring randomness, `Rz` phase sign,
`U^(r-1)`, dropped stabilizer signs) are each caught.

**Bug found and fixed:** `simulate_with(Request::Samples)` returned
`NotSupported("repeat::run_dense needs a unitary program")` when terminal
measurements formed a detected repeat (e.g. the same qubit measured many
times after a non-Clifford repeated block), where `simulate` succeeds.
`strip_measures` only stripped top-level `Ops` nodes; it now recurses and
`needs_dense` is decided on the stripped program
(`audit_repeat_repeated_terminal_measurements`).

Note: §2(b) recommends the `2^k` power only for `r <= 1e4`; the code does
not enforce a bound on `r` (the cost model decides). The error grows ~linearly
in `r` like gate-by-gate does, so this is a documentation caveat, not a bug.
