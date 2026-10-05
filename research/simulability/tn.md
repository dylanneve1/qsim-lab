# An exact tensor-network contraction engine

**Provenance.** Branch `exp/tn` from `main` @ 6b21728, 5 October 2026. Machine: shared Intel Xeon Gold 6548Y+
(Emerald Rapids) Hyper-V VM, 16 vCPU (8 cores × 2 HT), AVX-512, 31 GB RAM, Linux 5.15, Rust 1.93.1; other users'
load during this study was 8–50 (1-minute average), recorded next to every timing. Python baselines: quimb 1.11.2 and
cotengra 0.8.2 (kahypar 1.3.7, optuna 5.0.0, cotengrust 0.2.1) installed privately into `/dev/shm/qsim/ext/tn-py`
(numba replaced by a pass-through stub: quimb imports it unconditionally but does not use it on the contraction path).

Code: `src/engines/tn/` (`network.rs` construction and simplification, `path.rs` tree search, `exec.rs` execution,
`mod.rs` API), planner integration in `src/planner.rs` (`Engine::Tn`), tests in `tests/engines/tn.rs`, driver
`examples/tn_bench.rs`. Data and scripts: `research/data/tn/`.

## Headline

(to be written)

## 1. What the engine computes

`engines::tn` answers three requests of a unitary circuit `C` on `n` qubits, exactly (no truncation anywhere; the
only error is floating-point rounding):

| request | API | network |
|---|---|---|
| one amplitude `<x\|C\|0^n>` | `tn::amplitude(c, bits, &opts)` | `\|0>` vectors on every input wire, `<x_q\|` on every output wire |
| `2^k` amplitudes over open qubits | `tn::amplitudes(c, bits, open, &opts)` | the `k` open output wires stay open; entry `j` has bit `t` of `j` on qubit `open[t]` |
| `<0\|C† P C\|0>` for a Pauli string `P` | `tn::expectation(c, &[(q, Pauli)], &opts)` | gates outside the backward light cone of `supp P` are dropped (they cancel against their inverses); the doubled network `C_cone · P · C_cone†` is contracted as the amplitude `<0\|D\|0>` |

Every gate of `qsim_lab::Gate` is accepted (1-qubit gates, CNOT, CZ, SWAP, iSWAP, iSWAP†, CPhase, Toffoli);
measurements, resets, noise and classical control are refused with `SimError::MeasurementNotSupported`. Precision is
complex f64 (default) or complex f32. A memory budget `max_bytes` bounds every contraction: the tree is sliced until the
largest intermediate has at most `max_bytes / (8 · entry size)` entries, and the executor refuses
(`SimError::TooLarge`) if one slice's working set plus the slice-invariant cache would still exceed the budget
(`run_network` then slices four times harder and retries, up to six times).

## 2. Network construction and simplification

Every gate becomes a dense tensor over its output and input wire indices (dimension 2), with the input vectors and the
closed output vectors as rank-1 tensors. `Network::simplify` then runs four exact passes to a fixed point (at most 32
rounds), in quimb's ADCRS order:

1. **Column reduction (C).** If a tensor is non-zero for only one value `v` of a (non-output) index, the index is fixed
   to `v` in every tensor that holds it. A tensor that is identically zero makes the whole network zero.
2. **Diagonal reduction (D).** If a tensor vanishes wherever two of its indices differ, the second index is identified
   with the first everywhere (a tensor holding both takes its diagonal). This turns every diagonal gate (Z, S, T, Rz,
   CZ, CPhase, and the diagonal halves of CNOT and Toffoli) into a tensor on *hyperedges*: one index per wire segment,
   shared by every diagonal gate on it. It also finds that iSWAP and fSim(π/2, φ) are diagonal in the *crossed* pairs
   `(out_a, in_b)` and `(out_b, in_a)` (they are SWAP times a diagonal gate), so each such gate becomes a 2-index
   "edge" tensor between two wire segments. D must run before 1-qubit gates are absorbed (that would hide the
   structure); the first version of the engine had the order reversed and its Sycamore networks were 2–4× more
   expensive to contract (§7).
3. **Rank simplification (R).** Scalars are multiplied into a global factor; a tensor is merged into a neighbour when the
   merged tensor has no more indices than the larger of the two (this absorbs rank-1 boundary vectors and 1-qubit
   gates where they fit, and merges consecutive gates on the same wires).
4. **Split (S).** A tensor of rank 3–6 is split by an SVD along the index bipartition with the smallest matrix rank `r`,
   if `r` is below both sides' sizes and the two factors are no larger than the tensor (`r = 1`: an outer product).
   Factor entries at or below `1e-15 · max|T|` are set to exact zero so the other passes can see the structure.

Indices held by a single tensor that are not outputs are summed out at the end, and the ids are renumbered.

## 3. Contraction-tree search

The search works on the hypergraph of the simplified network (`Hypergraph`: index sets, log2 dimensions, outputs).
Costs follow cotengra's conventions so the numbers are directly comparable: one pairwise contraction costs the product
of the dimensions of every index either input holds (complex multiply-adds; ×8 for real flops), a tensor's size is
its number of entries, and a sliced tree costs (number of slices) × (cost of one slice).

* **Randomised greedy** (`path::greedy`): repeatedly contract the connected pair with the smallest
  `size(ab) − α (size(a) + size(b))` (log-scaled, plus Gumbel noise of temperature τ); disconnected remainders are
  joined smallest first.
* **Recursive multilevel bisection** (`path::bisection`): the tensors are partitioned into two halves minimising the
  total log2 size of the cut indices (hyperedges count once) under an imbalance bound ε, with heavy-edge matching
  coarsening, greedy-growing initial partitions (8 tries) and Fiduccia–Mattheyses refinement at every level;
  sub-networks of at most `cutoff` tensors are finished greedily.
* **Hyper-parameter search** (`path::search`): `trials` trees, half with random (α, τ, ε, cutoff), then two rounds in
  which half of the trials perturb the four best parameter sets so far (a small evolution strategy); trials run in
  parallel (rayon).
* **Subtree reconfiguration** (`path::reconfigure`): for every internal node (most expensive first) the subtree with
  `k = 8` frontier tensors is re-solved exactly by dynamic programming over subsets (`3^k` splits), minimising the
  summed cost under an optional size cap; repeated while it improves.
* **Slicing** (`path::slice_tree`): while an intermediate exceeds the target, slice the non-output index of an
  oversized tensor that minimises the total all-slice cost (`d·(T − S_i) + S_i` with `S_i` the cost of the
  contractions involving index `i`), then reconfigure under the new sizes.

The four best trees of the random phase are reconfigured, sliced with reconfiguration, and the cheapest is returned
with its statistics (`PathStats`: log10 cost unsliced and sliced, log2 largest intermediate, slices, overhead, trials,
seconds).

## 4. Execution

`exec::contract` compiles the tree once (`ExecPlan`) and runs it per slice:

* **Pairwise contractions.** For inputs A (the larger) and B, every index is batch (shared, kept), summed (shared, not
  kept), or free. The default *loop-over-GEMM* plan takes the largest run of A's free indices as GEMM rows, the largest
  run of summed indices that is contiguous and identically ordered in both inputs as the inner dimension, and the
  largest run of B's free indices as columns; every other index becomes an explicit strided loop (summed loops
  accumulate). Nothing is permuted, so a contraction of a big tensor with a small one is one pass over the big one.
  When the GEMM blocks would be tiny and numerous, the plan instead permutes the inputs into `[batch, M, K] × [batch,
  K, N]` (the larger input only if its index groups are not already contiguous). GEMMs go through faer
  (`private-gemm-x86`, AVX-512); blocks of at most 64 multiply-adds use a plain loop. `PairStrategy::{Loops, Permute}`
  forces either plan (both are differential-tested).
* **Slices.** Sub-trees whose leaves hold no sliced index are contracted once and cached; per slice only the dependent
  part runs. Slices run on parallel workers (sequential GEMMs) when that many working sets fit the budget side by
  side; otherwise one after another with parallel GEMMs and loops.
* **Memory.** The plan computes each slice's peak live entries (results waiting for their sibling, the output, sliced
  leaf copies) and the permutation scratch; the check is `cache + workers · (2 · peak + scratch) ≤ max_bytes`.
  Buffers are recycled through a pool capped at the live peak.

## 5. Exactness

(to be written)

## 6. Planner integration and the cost model

(to be written)

## 7. Sycamore-pattern circuits against cotengra + quimb

(to be written)

## 8. IBM 127-qubit kicked Ising through light cones

(to be written)

## 9. Caveats and negative results

(to be written)

## 10. Reproduction

(to be written)
