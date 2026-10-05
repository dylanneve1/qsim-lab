# Graph compiler: compile once, bind many; dedup; fuse; partition (`src/graph/`)

Branch `exp/graph-compiler` (stacked on `exp/integrate-dense-fusion`, which is main + dense
k-qubit fusion). Owner's idea: treat circuits the way a graph compiler for ML treats a model:
fold constants, fuse into faster operations, find repeated subgraphs and compile them once,
then re-parameterise, and partition the graph into subgraphs that each run on their best
engine.

Code: `src/graph/{param,compiled,observable,rewrite,fold,dedup,partition}.rs`, a few hooks in
`src/engines/blocked.rs` (stage preparation that reports where every op went, so numbers can be
patched in place; `schedule_diag_order`), `pipeline::SimOptions::graph`.
Tests: `tests/compiler/graph.rs` (12 tests; every path is checked against the independent reference
state vector of `tests/audit_common`, extended with textbook matrices for U/Sx/iSWAP).
Bench: `examples/graph_bench.rs`. Raw data: `research/data/graph-compiler/`.

## Summary

| technique | where it wins (Mac M1 Pro, min of 3) | coverage | default |
|---|---|---|---|
| **1. Parameterised compilation** (`CompiledCircuit::compile` once, `bind` patches numbers) | bind costs **2–7 µs** (a compile costs 0.1–0.5 ms); per evaluation **3.0–5.7×** vs recompiling at n = 8–10, **1.1–1.4×** at n = 12–16, ≈1.0× at n = 20 (the state-vector run dominates). Parallel sweeps (one state per worker) **2.0–8.4×** vs a serial loop over the plain executor, **1.0–2.4×** vs the same loop parallelised | every unitary circuit | it is an API (opt in by using it) |
| **2. Structural dedup** (canonical blocks up to qubit relabelling) | per-block compile work once per class: **28.6×** (Trotter), **23.7×** (ripple adder), **8.9×** (lookup windows), **2.9×** (Pauli Trotter), **2.2×** (QFT); recipe CSE at bind **1.9–2.3×** (Trotter, QAOA) | 99–100 % of ops of Trotter, adders, lookup windows, 90 % Pauli Trotter, 65 % QFT (k = 4); 0.4 % random brickwork, 4 % HEA (89–100 % as *shape* classes) | recipe CSE on |
| **3a. Phase-polynomial region rewrite** (monomial runs → gadgets on the inputs + simplified permutation) | **1.25–1.54×** on QAOA written as CNOT·Rz·CNOT and Pauli-gadget Trotter circuits (CNOT ladders cancel completely: 528 → 0 permutation ops), mostly by cutting blocked-executor stages (12 → 7, 19 → 11) | circuits with CNOT-sandwiched rotations | on, kept only if the plan is cheaper (fewer stages, then fewer executor ops) |
| **3b. Basis-state constant folding** | **3.6–4.2×** when an arithmetic block runs on classical inputs (1318 ops → 4); ≈1.0× once a control is in superposition | arithmetic on basis inputs | on (never adds ops) |
| **3c. k ≤ 5 dense fusion by cost model** | **not built**: under the measured kernel cost rule no block of 4 or 5 qubits qualifies in any of 8 families; k = 2–3 is the integrator's dense fusion (wins only on SU(4)-type blocks) | k = 4, 5: 0 % of ops | — |
| **4. Cross-engine partitioning** (controlled-gate cut, unitary insertions, WHT join, planner-priced) | 60-qubit "long chain + deep core": **1.2 s** for 64 exact amplitudes; state vector, HSF impossible (16 EiB / 16 GiB), exact MPS **aborted after 122 s**, sparse aborted: beats every single engine by > 100× | circuits whose interaction graph has a small controlled-gate cut between an MPS-friendly and a dense-friendly part | API (`partition::plan_cut` + `cut_amplitudes`) |

**What did not pay:** dense fusion beyond 3 qubits (no qualifying blocks); a kernel JIT
(nothing to specialise once 4–5-qubit dense blocks are out; the fixed-shape kernels are
already monomorphised for k = 1–3); rewriting at n ≤ 12 when the circuit is not
CNOT-heavy (in-cache CNOT passes are as cheap as the diagonal terms that replace them);
folding once inputs are superposed; dedup on random/ansatz circuits (no exact repeats);
and the compile-once path at n ≥ 20 (the run is ≥ 99 % of the time; the remaining gain
there comes from the lowering, not from skipping the compile).

## Machine and method

Apple M1 Pro (8 cores: 6 performance and 2 efficiency), macOS 26.6.2, rustc 1.93 release
build, default `BlockConfig` (aarch64: 1 MiB blocks, dense fusion k = 2 on).
Every timing ran under the swarm bench lock (`research/data/graph-compiler/locked.sh`,
≤ 140 s per locked run). The Mac is shared, so load averages were **8–19 on 8 cores**
(each log's first line). Variants are interleaved within a run and the **minimum of 3**
repetitions is reported.

Noise floor: an A/A pair (QAOA n = 16, folding on and off, which gives the *identical*
plan) came out 0.65× when the variants always ran in the same order, and 1.03× once the order
alternated (`fold2.log` vs `fold3.log`). Treat any ratio within about ±10% as noise, and any
ratio from the fixed-order benchmarks (`bind`, `rewrite`) within ±25% as uncertain. The
large effects (≥ 2×) reproduce across runs at loads of 10 and 18 (`bind2.log` vs `bind3.log`).

Correctness: `cargo test --release --test graph` (12 tests) also passes with
`QSIM_FUZZ_ITERS=10 QSIM_FUZZ_SEED=777`. Every test compares against `RefSv` (no shared
kernels) to 1e-10–1e-12, global phase included for amplitudes. Two
mutation checks (a sign flip in the Rzz global phase; the sign of a known-1 parity in the
region rewrite) made the tests fail, as they should, and the folding test caught a real bug
in the first version of §3b.

## 1. Parameterised compilation

```rust
let mut pc = ParamCircuit::new(n, 2 * p);          // n qubits, 2p parameters
pc.gate(Gate::H(q));                                // fixed gates
pc.rzz(a, b, Angle::param(2 * l));                  // angles: c0 + Σ c_i θ_i
pc.rx(q, Angle::scaled(2 * l + 1, 2.0));
let cc = CompiledCircuit::compile(&pc, Some(&hamiltonian), &GraphOptions::default())?;
let e: f64 = cc.bind(&theta)?.expectation()?;      // or cc.expectation_at(&theta)
let es: Vec<f64> = cc.sweep_expectation(&thetas)?; // parallel over parameter vectors
```

**What `compile` does once.** It runs basis-state folding (§3b) and the region rewrite (§3a),
keeping the rewrite only if the plan is cheaper. Then it takes the light cone with respect to
the observable (plus a diagonal suffix for Z-only observables) and splits the circuit into
independent components. Next it lowers to executor ops and fuses single-qubit runs: constant
runs are multiplied out at compile time, and parameterised runs become *recipes* (an ordered
factor list). It plans the stages and schedules the diagonals (both purely structural). Every
constant stage is prepared with all executor options (dense fusion, L1 tiling). Every
parameter-dependent stage is prepared once as a *template* that records where every op landed
(`blocked::prepare_stage_mapped` → `OpLoc`). Finally it runs the constant prefix (everything
before the first parameter-dependent stage) and keeps that state.

**What `bind` does.** It evaluates each *distinct* recipe once (§2): a few 2x2 products or one
`e^{iθ}`. Then it clones the templates and patches the numbers in place
(`PreparedStage::set_u1` / `set_phase`); the kernel kind (X, real, complex) is re-derived from
the bound matrix. Nothing structural is recomputed. Exactness: the plan never depends on a
value a parameter can change. A run is treated as diagonal iff every factor is diagonal for
*all* angles, and phases equal to 1 are kept as terms.

Per-bind costs (`bind2.log` (load 17), `bind3.log` (load 11); `compiled` = bind + run +
expectation; `recompile` = `CompiledCircuit::compile` + the same, every time; `baseline` = bind
to a plain `Circuit` and run the blocked executor, i.e. what the existing API does):

| workload | n | ops | bind | compiled | recompile | ×  | baseline | × | sweep (par) vs serial baseline |
|---|---|---|---|---|---|---|---|---|---|
| QAOA 3-regular, p = 3 | 10 | 85 | 1.8 µs | 0.071 ms | 0.212 ms | **2.99** | 0.161 ms | **2.27** | **6.7×** |
| QAOA | 16 | 136 | 4.0 µs | 5.05 ms | 5.62 ms | 1.11 | 7.57 ms | 1.50 | 4.5× |
| QAOA | 20 | 170 | 5.5 µs | 45.3 ms | 46.0 ms | 1.01 | 67.7 ms | 1.49 | 2.0× |
| HEA (Ry Rz + CNOT ladder), L = 4, TFIM ⟨H⟩ with X terms | 8 | 92 | 3.3 µs | 0.015 ms | 0.086 ms | **5.68** | 0.034 ms | **2.23** | 4.8× |
| HEA | 12 | 140 | 4.1 µs | 0.50 ms | 0.68 ms | 1.36 | 0.58 ms | 1.16 | **8.4×** |
| HEA | 16 | 188 | 6.7 µs | 6.30 ms | 6.71 ms | 1.06 | 6.36 ms | 1.01 | 3.6× |

* Before in-place patching, `bind` re-prepared the parameter-dependent stages and cost
  57–96 µs (more than the whole n = 10 evaluation): the compiled path was 0.86× the baseline
  at n = 10. Patching made bind 15–30× cheaper; that is the change that makes the
  compile-once path pay.
* The QAOA "baseline" column also includes a lowering difference: a plain `Circuit` has no
  `Rzz`, so it is `CNOT·Rz·CNOT`, and the compiled path makes it two diagonal terms (1.5× at
  n = 16–20). That is a real gain of the API, but not of compiling once; the `recompile`
  column isolates the compile-once effect.
* At n ≥ 16 the state-vector run is 90–99% of an evaluation, so compiling once is worth
  1.0–1.1× there. The sweep column is where large sweeps gain: binds run in parallel with
  one state per worker under a 1 GiB memory guard, on top of the executor's own parallelism.
  The baseline parallelised the same way is 1.3–2.4× slower than the compiled sweep
  (`baseline-par` in the logs).

**Not done:** checkpointing for parameter-shift gradients, where binds differ in one
parameter and could restart at the first stage that parameter touches. It would only matter
at n ≥ 16 (multi-stage plans); see "Next".

## 2. Structural deduplication

`dedup::analyse(pc, k)` cuts the circuit into blocks of at most `k` qubits along the wire
DAG. This is the greedy grouping a fusion pass uses: an op joins the open block of its qubits
while the union stays ≤ k; otherwise the blocks on its qubits close. Each block is
canonicalised by renaming its qubits in order of first use and recording gate kinds plus the
exact bits of every affine angle coefficient, then hashed. Blocks with equal keys form a
*class*; a second key without angles gives *shape* classes (same block, different angles).
Executing the blocks in order of their last op reproduces the circuit exactly (tested).

Coverage = fraction of ops in a class with ≥ 2 instances (`dedup1.log`, k = 4; the log has
k = 2…5):

| family | n | ops | blocks | classes | coverage | shape coverage | per-class block unitaries (compile) | recipe CSE (bind) |
|---|---|---|---|---|---|---|---|---|
| TFIM Trotter, 50 steps (param dt) | 12 | 1150 | 322 | 9 | **100%** | 100% | 322 → 9: **28.6×** | **1.92×** |
| QAOA p = 5 | 16 | 216 | 66 | 52 | 15.7% | 64.4% | 1.0× | **2.34×** |
| HEA L = 6 (all angles distinct) | 12 | 210 | 39 | 36 | 4.3% | 89.0% | 1.0× | 1.00× |
| Pauli-gadget Trotter, 4 steps | 12 | 1204 | 100 | 34 | 90.6% | 90.9% | 2.9× | 1.27× |
| Shor ripple adder, controlled U_a (8-bit) | 28 | 4746 | 1865 | 39 | **99.5%** | 99.5% | **23.7×** | — |
| Shor lookup windows, controlled U_a | 31 | 1410 | 625 | 38 | **99.6%** | 99.6% | **8.9×** | — |
| QFT | 16 | 144 | 52 | 24 | 64.6% | 89.6% | 2.2× | — |
| random brickwork, depth 20 | 16 | 790 | 80 | 78 | 0.4% | 90.1% | 1.0× | — |

* Analysis costs 0.05–1.5 ms. The repeats are found with no markers: Trotter steps, MAJ/UMA
  ripples on shifted qubits, and lookup windows. The existing `compile::repeat` detector needs
  tandem repeats on the *same* qubits.
* What a class saves is per-block compile work: its unitary, fused matrix, rewritten form, and
  bound numbers for one parameter vector. Measured on block unitaries (k = 4, computed by
  simulating the block on its 2^k basis states), the saving is proportional to blocks/classes.
  Inside `CompiledCircuit` the same idea is *recipe CSE*: all `Rx(2β_l)` of a mixer layer,
  and all `Rzz(γ_l)` phases of a cost layer, are evaluated once per bind (1.9–2.3× bind
  time; HEA has no repeats and gets 1.0×).
* **No run-time effect on the state-vector engine.** Every instance still has to be
  applied to the state. Dedup only removes compile- and bind-time work, which is small next to
  the run except for small registers and huge sweeps. Shape classes (same block, other angles)
  are what a parameterised template would exploit. Here that already happens structurally,
  because every bind reuses the whole plan.

## 3. Fusion to faster ops

### 3a. Phase-polynomial regions (`rewrite::phase_regions`)
A *region* is a maximal run of affine-permutation ops (X, CNOT, SWAP, the X of Y) and
diagonal ops (Z, S, T, Rz, Phase, CZ, CPhase, Rzz, gadgets). Every wire is tracked as an
affine parity of the region's inputs, and every diagonal op becomes parity rotations
`exp(-iα/2 (-1)^{p·x})`, merged per parity (this is phase folding). The region then equals
`N · D(x)`: gadgets on the *input* wires followed by the permutation `N`. `N` is re-emitted
as X's when its linear part is the identity (a ladder and its inverse cancel), as SWAPs when
it is a wire permutation, and as the original permutation ops otherwise. Global phases are
kept (`POp::Global`). Before the pass, the ops can be re-scheduled in any topological order
that puts ready region ops first. Both the original and the re-scheduled order are tried,
because greedy re-scheduling can pull half of the next gadget's ladder into a region. A
gadget of weight k runs as 2^(k-1) diagonal terms, batched by the executor's diagonal groups;
up to `GraphOptions::max_zstring` = 6, and wider ones as ladders.

`rewrite2.log` (load 10–12; `compiled` = no rewrite, `rewritten` = always, `auto` = the
default, which keeps the rewrite only if it gives fewer stages, then fewer executor ops):

| workload | n | perm ops | kops / stages | compiled | rewritten | × | auto chose |
|---|---|---|---|---|---|---|---|
| QAOA as CNOT·Rz·CNOT, p = 3 | 12 | 108 → 0 | 210/1 → 156/1 | 0.517 ms | 0.413 ms | **1.25** | rewrite |
| same | 20 | 180 → 0 | 350/12 → 260/7 | 98.5 ms | 63.9 ms | **1.54** | rewrite |
| Pauli-gadget Trotter (weights 2–4, X/Y ends) | 12 | 528 → 0 | 915/1 → 757/1 | 1.86 ms | 1.61 ms | 1.16 | rewrite |
| same | 20 | 476 → 0 | 826/19 → 774/11 | 155.8 ms | 108.3 ms | **1.44** | rewrite |
| TFIM Trotter (Rzz native) | 12 | 0 | unchanged | 1.33 ms | 1.34 ms | 1.00 | plain |
| HEA (CNOT ladder, no rotations between) | 12 | 44 → 44 | unchanged | 0.42 ms | 0.41 ms | 1.04 | plain |

The first version re-scheduled greedily only, and on the Pauli workload it left 268 of 528
permutation ops (0.96× at n = 12; `rewrite1.log`). Trying both orders fixed that. Where the
rewrite wins, it wins mostly by **removing blocked-executor stages**: CNOT targets must be in
the cache block, diagonal terms need nothing. Inside one cached block, a CNOT pass (X kernel)
costs about the same as the diagonal-group pass that replaces it, so at n ≤ 12 the gain is
only the 15–25% fewer executor ops.

### 3b. Basis-state constant folding (`fold::fold_basis`)
Wires start in `|0>`, and while a wire is in a known basis state the pass folds as follows.
X and Y flip it (Y adds ±i). Diagonal ops on known wires become global phases, which may be
parameterised. A known `1` turns CPhase into Phase and flips the sign of a gadget angle. A
CNOT or Toffoli with a known `0` control vanishes, and with a known `1` control it loses that
control. Known wires stay physically `|0>`: a known `1` is restored by an X only when an
unfoldable op touches the wire, and at the end. This is the parameter-aware, basis-only
sibling of `compile::stateprop`, which handles stabilizer states on fixed circuits. The first
version dropped the restoring X; the differential test caught it.

`fold3.log` (load 8, variants alternated):

| circuit | n | ops | run fold off → on | × |
|---|---|---|---|---|
| Shor ripple controlled-U_a, control `|1>`, x = 1 | 16 | 1318 → 4 | 0.208 → 0.050 ms | **4.15** |
| same, control `|+>` | 16 | 1318 → 1258 | 0.212 → 0.214 ms | 0.99 |
| Shor lookup-window controlled-U_a, control `|1>`, x = 1 | 22 | 786 → 4 | 14.0 → 3.86 ms | **3.63** |
| same, control `|+>` | 22 | 786 → 742 | 15.0 → 15.4 ms | 0.97 |
| TFIM Trotter (first Rzz layer acts on `|0..0>`) | 16 | 310 → 296 | 12.8 → 12.6 ms | 1.02 |
| QAOA (starts with H) | 16 | 136 → 136 | 7.74 → 7.52 ms | 1.03 (A/A) |

On classical inputs the whole reversible block becomes four X gates. As soon as one control
is superposed, only the ops before it fold. Exploiting the rest needs the sparse or monomial
engines, and the planner already routes such circuits there.

### 3c. k ≤ 5 dense fusion with a cost model: measured not to pay
Dense k-qubit fusion is the integrator's work (`research/performance/dense-fusion.md`). Its measured
kernel cost rule is that a k-qubit dense pass costs about 2^k single-qubit passes, so a group
is fused only if it holds ≥ 2^k dense single-qubit gates. Fused groups won 2.15× (f32) on
SU(4) blocks and lost or tied elsewhere. Before building a front-layer DP up to k = 5, I
counted how much of each family such a cost model could ever fuse. I used the dedup blocking,
which is exactly the greedy grouping (`fusable.log`):

| family | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|
| Pauli-gadget Trotter | 28.5% of ops | 20.2% | **0%** | **0%** |
| random brickwork | 9.1% | 10.6% | **0%** | **0%** |
| Trotter, QAOA, HEA, adder, windows, QFT | 0% | 0% | 0% | 0% |

No block of 4 or 5 qubits in any family holds 16 (32) dense 1q gates. A DP would not change
that, because it only chooses among the same groups. So the k = 4–5 kernels, the DP and a
JIT (cranelift) for per-shape kernels were **not built**: there is nothing for them to fuse
on these workloads. The fusions that do pay are the ones above that change the *kind* of
work: monomial runs into diagonal gadgets plus a cheaper permutation, and known-basis wires
into nothing.

## 4. Subgraph partitioning across engines (`partition`)

Cut the qubits into `A | B` so that only controlled gates (CZ, CNOT, CPhase) cross. For cut
gate `C-U` (control `c`, target `t`):

```
C-U = |0><0|_c ⊗ 1 + |1><1|_c ⊗ U_t = ½ Σ_{s,u∈{0,1}} (−1)^{su} Z_c^s ⊗ U_t^u
⟨x|ψ⟩ = 2^{−c} Σ_{s,u} (−1)^{s·u} ⟨x_A|A_s⟩⟨x_B|B_u⟩
```

Each side therefore runs `2^c` *unitary* circuits; no engine ever sees a projector, so any
engine works per side. The double sum is a Walsh–Hadamard transform of the B vector (c·2^c
per amplitude). Candidate cuts come from greedy min-cut growth of B from the 4
highest-degree qubits: each step adds the qubit with the most gates into B, breaking ties by
fewest gates out, and every prefix is a candidate. **The planner prices every candidate** as
`2^c × (planned A + planned B)` and compares it with its best single engine. Execution then
calls `planner::amplitudes` per side, so each side gets its own engine and its own
speculation.

Family **chain + core**: A is a 40-qubit 1D chain of depth 4 (random U + nearest-neighbour
CZ; low entanglement). B is a 20-qubit core of depth 16 (random U + random all-to-all CZ
pairing; volume law). They are joined by `c` CZs between random A and B qubits in the
middle. 60 qubits, ~720 gates, 64 random amplitudes (`part1-3.log`, load 11–29):

| c | planner's cut | predicted | partitioned (measured) | best single engine |
|---|---|---|---|---|
| 1 | \|B\| = 20, A: MPS, B: SV | 0.34 s | **0.23 s** | SV: needs 16 EiB; HSF: a 16 GiB block; sparse: aborted at the deadline |
| 2 | same | 0.67 s | **0.46 s** | same |
| 3 | same | 1.34 s | **1.20 s** | **exact MPS aborted after 122.6 s** (deadline 120 s; the planner had predicted 6.8 s) |
| 4 | same | 2.68 s | **1.89 s** | same |
| 6 | same | 10.7 s | **7.23 s** | same |

* At c = 3 the partitioned run is > 100× faster than the only single engine that could even
  start, and that engine never finished. The run time doubles per cut gate, as priced, and the
  planner's prediction is within 1.5× of the measurement.
* Planning (all candidates, priced by the planner) takes 0.11–0.13 s.
* At small sizes (na = 12, nb = 10) all engines finish, and every one agrees with the
  partitioned amplitudes to ≤ 1.4e-16. There HSF (the same idea with dense sides) is the best
  single engine. The new part is the *heterogeneous* sides, plus the planner pricing the cut.
* Finding: the planner's MPS model underestimates exact MPS on this family by > 18×
  (predicted 6.8 s; still running at 122 s), because the bond-replay bound does not see the
  SWAP routing of the core's all-to-all gates. Worth a planner fix.

## Python binding (for `python/` and `qsimlab`)

The Rust API was shaped so that a PyO3 wrapper is thin:

* `ParamCircuit` uses plain builder methods with `impl Into<Angle>`. In Python, angles are
  floats or `Param` objects with `+`, `*` by float (`Angle` is `c0 + Σ c_i θ_i`, so
  `2*theta[3] + 0.1` maps to `Angle { c0: 0.1, terms: [(3, 2.0)] }`).
  `ParamCircuit::from_circuit` wraps an existing fixed circuit.
* `Observable` is built from `(coef, "X0 Y3 Z5")` pairs, the same sparse-string format as
  OpenFermion and Qiskit `SparsePauliOp.from_sparse_list`.
* `CompiledCircuit` is `Clone + Send + Sync` and owns everything. The methods that need no
  borrowed handle are made for bindings: `expectation_at(&[f64])`, `statevector_at`,
  `amplitudes_at`, and `sweep_flat(&[f64])`, which takes a row-major `[batch × num_params]`
  buffer, i.e. a C-contiguous `numpy.ndarray` without a copy.

Proposed `qsimlab.compile` surface:

```python
theta = qsimlab.Parameters(2 * p)
pc = qsimlab.ParamCircuit(n, theta)
pc.h(range(n)); pc.rzz(a, b, theta[0]); pc.rx(q, 2 * theta[1])
cc = qsimlab.compile(pc, observable=[(0.5, "Z0 Z1"), (0.7, "X2")])  # GraphOptions as kwargs
e  = cc.expectation(np.array([...]))       # one bind
es = cc.sweep(params_2d)                   # (B, P) float64 -> (B,), GIL released, rayon
psi = qsimlab.compile(pc).statevector(theta_values)
```

The pipeline hook `SimOptions::graph` (opt-in) runs fixed-angle amplitude and Z-expectation
requests through the same path. The current `qsimlab.simulate` can expose it as an
`engine_options={"graph": True}` flag without a new class.

## Flags and defaults

`GraphOptions` (all of the compiler's switches): `light_cone`, `diagonal_suffix`,
`components`, `prefix_cache`, `patch_bind`, `dedup_recipes`, `fold_basis` (all on),
`rewrite: Some(..)` (on, kept only when the plan is cheaper), `max_zstring = 6`, `block:
BlockConfig`, `mem_bytes`. Every option combination in `tests/compiler/graph.rs::option_sets` is
differential-tested, including forced dense fusion with L1 tiling and both bind modes.

The compiler is not behind `compile::plan::PlanOptions`. `PlanOptions` is `Copy` and
configures the flat-circuit passes of `compile_*`; the graph compiler is a separate entry
point. On the existing pipeline it is reached through `pipeline::SimOptions::graph` (**default
off**), because the ordinary pipeline also has the stabilizer, Pauli-path and compressed
engines, which the dense graph path does not try. The partitioner is also API-only
(`partition::plan_cut`, `partition::cut_amplitudes`). Wiring it into `planner::plan` as an
extra candidate is the obvious next step, and it costs about 0.1 s of planning per circuit,
so it should only run when the best single engine is predicted slow.

## Next

1. Partition as a planner candidate (amplitudes and Z-expectations; expectations need
   `⟨A_s'|O|A_s⟩` cross terms, i.e. MPS–MPS overlaps), with a DFS over the cut, as `hsf.rs`
   does, so the side runs share their prefix before the first cut gate.
2. Parameter-shift checkpointing for multi-stage plans: restart at the first stage the
   shifted parameter touches.
3. A parity-phase executor kernel. Today a weight-k gadget is 2^(k-1) pattern terms, which is
   why the rewrite does not win inside a single cached block.
4. Feed the dedup classes to the dense fusion pass (fused matrices once per class), once
   fusion finds blocks worth fusing.
