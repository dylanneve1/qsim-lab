# qsim-lab architecture: phase 3 (owner directive: "bigger circuits, faster, beyond the research")

Author: Claudius (architect). Agents implement against this document. Deviations need a written reason in the PR.

## 0. Where we are (1 Oct 2026, main ceaef67+)
Seven exact engines exist, each strong in its own regime: blocked SV (beats qsim and Aer on this box), tableau (sign-tracking), SymPhase sampler (3–4× Stim, pending an identical-circuit check), rotation frame + adaptive (Clifford+T), sparse SV (Shor), HSF, MPS. Analysis exists in two places: the flat-Vec compile passes and the DAG.

What's missing is **one pipeline** that sends every circuit, and every *part* of a circuit, to the cheapest exact engine. The other missing piece is a dense engine that isn't capped by RAM.

## 1. The pipeline (single entry point)
```
qsim::run(circuit, Request{ amplitudes | samples(shots) | expectations(obs) }, Budget{mem, time})
  -> Dag::from_circuit                      (src/dag.rs; single source of dependency truth)
  -> passes on the DAG (in order): peephole(commutation) -> swap-elim -> state-prop
       -> light-cone(Request) -> components
  -> per component: Planner::choose(component, Request, Budget) -> Plan
  -> execute plans; combine (product of components; classical suffix)
```
- `src/compile/` passes get ported to the DAG. The old flat-Vec implementations stay only as A/B oracles in tests, with identical results required, as the DAG agent already showed for peephole.
- **The Planner is the novel piece.** It's a learned cost model, not hand rules:
  - features per component: n, gate counts by class (Clifford / monomial / diagonal / dense 1q / dense 2q / non-Clifford rotations), T-count, the active dimension d_k profile from the rotation frame (cheap), the min cut from KL partitioning, a light-cone width proxy for the MPS bond, and sparsity from state propagation;
  - candidates: Tableau, SymPhase (noisy Clifford sampling), Adaptive (frame → compressed SV), SparseSV, MPS (exact only if the predicted bond is under the cap), HSF, BlockedSV, OutOfCoreSV;
  - the cost model is fitted on benchmark runs we generate ourselves (log-time regression per engine), stored in `research/data/planner/`, and validated on held-out circuit families. That's the "phase diagram of exact simulability" research question, built straight into the product.

## 2. Dense engine roadmap (bigger and faster for ANY circuit)
Ordered by expected gain per effort on 4–8 cores:
1. **k-qubit dense fusion (k ≤ 4)** inside the blocked executor. Group gates by DAG front layers into ≤ 4-qubit unitaries. That raises arithmetic intensity, which is where qsim's advantage comes from on big machines. Use the SoA kernels, real/complex split.
2. **Monomial-segment fusion.** Maximal runs of monomial gates (X/CNOT/SWAP/Toffoli/diagonal) become one generalized permutation `x → φ(x)·π(x)`. Use an affine π (X/CNOT/SWAP) plus a phase polynomial whenever possible, and apply it in one pass. The honest risk: blocked execution already streams memory once per block, so the gain may be compute-bound. **Measure before investing**: build a micro-prototype first and kill the idea if it's under 1.3×.
3. **Out-of-core SV** with the same chunk/global-qubit-swap model as distributed simulators. Swap schedules come from DAG lookahead (choose which qubits are local per segment so the most gates in a row are local; a greedy plus beam search over the next K gates). This is the generic "bigger" lever. Then a 2-process prototype, then VPS + MacBook.
4. **Precision tiers.** f32 by default when the request's tolerance allows; f64 when asked. Report the error estimate by running the first block in both.

## 3. Sampling-heavy workloads (where SOTA lives)
- SymPhase + the circuit-derived DEM is the QEC path. The next step is a **batched GF(2) mat-vec with pre-transposed sparse A**, plus bit-sliced 256-shot words (AVX2), so we stay ahead of Stim on equal circuits.
- Generic sampling from a dense state: the alias method or sorted uniforms (exists), plus marginal sampling straight from out-of-core chunks.

## 4. Correctness architecture (non-negotiable)
- Every engine implements the same `ExactEngine` trait and is fuzzed against the reference SV through one shared differential harness (`tests/audit_common`), using the same circuit generators across all op types.
- Every Planner decision is reproducible: a debug mode runs the chosen plan *and* the reference on small instances.
- The auditor gates every merge on the exact commit. When the owner orders a merge first, the auditor runs on main afterwards and any bug is hotfixed.

## 5. Work packages for the next round
- WP1 (headline): out-of-core SV plus DAG-driven swap scheduling. Brief: generic-scale.md.
- WP2: k-qubit dense fusion, plus a monomial-fusion micro-prototype with a kill criterion. Brief: sv-monomial.md, amended by §2.1–2.2 here.
- WP3: Planner v0. Features plus a fitted cost model, wired into `qsim::run`. Ground-truth timings come from the existing engines.
- WP4: gate-level Shor at 10^6 with a ripple-carry oracle (shor-ripple.md).
- WP5: syndrome-schedule search (schedule-search.md).
- WP6: auditor. The identical-circuit Stim comparison comes first (export our surface-code circuit to .stim and time both on equal circuits, with matching detector statistics), then whatever lands.
