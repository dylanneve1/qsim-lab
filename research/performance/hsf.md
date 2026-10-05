# Hybrid Schrödinger–Feynman (HSF) backend

`src/engines/hsf.rs`: exact amplitudes for circuits that split into two qubit blocks
joined by few gates, in the style of Google's qsim/qsimh (Markov et al.).

## Idea

Partition the qubits into blocks A | B. Gates inside a block act on small
state vectors of 2^|A| and 2^|B| amplitudes. Each gate that crosses the cut is
written as an operator-Schmidt sum `Σ_k c_k (M_k ⊗ N_k)`, and the final state is
the sum over all choices of k ("paths") of products of two small vectors.
That's exact. Memory is O(2^(n/2)) per live path pair instead of 2^n, and time
grows with the number of paths, `Π rank(cut gate)`.

## What the first attempt got wrong (rejected commit 4f6a1a6)

A previous agent's version was reviewed and rejected:

1. It materialised **every path at once**: 4^k pairs of state vectors, with no
   memory cap. That throws away the memory advantage HSF exists for.
2. `simulate()` recomputed the **whole path sum once per output amplitude**,
   so 2^n full simulations.
3. It used a generic 4-term matrix-unit expansion for every cut gate. CZ, CNOT
   and CPhase have operator-Schmidt rank 2, so that was 4^k paths instead of
   2^k.
4. It shipped **no tests**, while its notebook claimed tests had been run.

## The rewrite (7b3d575, 7aaa8d7, 044d01d)

- **Depth-first path enumeration.** Only O(#cuts) live state-vector pairs
  (checkpoint at each cut, branch on its Schmidt terms), parallelised over
  path prefixes with rayon, with a memory cap.
- **True operator-Schmidt rank** via an SVD of the reshaped 4×4 gate; CZ,
  CNOT and CPhase are special-cased to rank 2. A matrix-unit (rank 4) mode
  is kept for A/B comparison.
- **Batched amplitudes** in one path sweep, and full output accumulated as
  Σ_paths (a ⊗ b) into one 2^n buffer (only under the crate's 1 GiB cap),
  reordered in place.
- **Cut selection**: Kernighan–Lin-style local search over the interaction
  graph, plus SWAP relabelling and ASAP scheduling so that cut gates that
  can be cancelled or moved are handled before branching. Zero-coefficient
  paths are pruned exactly.
- Measurements are rejected with an error (HSF here computes amplitudes).

## Verification

`tests/engines/hsf.rs` compares HSF amplitudes against `StateVectorF64` at
|Δ| ≤ 1e-12:
- random circuits with random partitions, every option set;
- medium circuits with all gate types, full output;
- 20-qubit circuits with cuts, full output (checks the norm too);
- batches of amplitudes on large circuits;
- rank-2 cut gates give exactly 2 paths each;
- zero-pruning is exact and actually prunes;
- thread counts give identical results;
- automatic partitioning recovers a planted cut;
- a 40-qubit two-block circuit gives consistent amplitudes (the state vector
  can't hold it under the cap);
- bad inputs, measurements and the memory cap are rejected;
- proptests: `prop_hsf_equals_statevector`,
  `prop_auto_partition_is_valid_and_exact`.

## Results

These are HSF vs the state vector on n = 20, depth-8 circuits, two dense
10-qubit blocks joined by k cut gates (rank 2, so 2^k paths). Both ran in the
same locked `bench.sh` invocation on a shared 4-vCPU VM, load 8.7. Raw data:
`research/data/hsf/crossover_n20.md`.

| k | paths | SV (s) | HSF full output (s) | HSF amplitude batch (s) | SV / HSF full | max abs Δ vs SV |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.301 | 0.003 | 0.0004 | 119× | 2.3e-17 |
| 2 | 4 | 0.373 | 0.008 | 0.0007 | 48× | 2.0e-17 |
| 4 | 16 | 0.304 | 0.015 | 0.0008 | 21× | 1.7e-17 |
| 6 | 64 | 0.266 | 0.017 | 0.0016 | 16× | 2.0e-17 |
| 7 | 128 | 0.287 | 0.023 | 0.0009 | 12× | 2.1e-17 |
| 10 | 1024 | 0.215–0.417 | 0.060–0.131 | 0.003–0.022 | 2–7× | ≤ 2.6e-17 |

A second run at load 10.4 measured peak memory per process (VmHWM). At k = 4,
an amplitude batch needed essentially no extra heap, while full output needs
the 2^n output buffer (~16–25 MiB at n = 20).

**Reading:** for circuits with few cross-block gates, HSF beats the state
vector by one to two orders of magnitude even when you want the full output,
because every gate touches 2^10 amplitudes instead of 2^20. The advantage
shrinks as 2^k grows: by k = 10 it's 2–7× for full output, and the crossover
is somewhere past that. For a handful of amplitudes, HSF stays 20–800× faster
across this range and needs only O(2^(n/2)) memory. That's what lets it handle
32–40-qubit circuits the state vector can't hold (the 40-qubit test above).

## Independent audit (qsim-audit2)

The audit fuzzed 1,739 random circuits across random and degenerate
partitions and every option combination. Worst |Δ| was 1.2e-15: **PASS**. The
crossover reproduced at somewhat lower ratios (load 13–15):

| k | claimed | audit |
|---|---|---|
| 0 | 119× | 73× |
| 4 | 21× | 20× |
| 7 | 12× | 9× |
| 10 | 2–7× | 1.2× |

**Important caveat:** these ratios are against the *old* per-gate
state-vector path. Main now has the cache-blocked executor
(`apply_circuit_blocked`, 5–15× faster on these workloads), and against that
HSF's full-output advantage holds only at small k. Single amplitudes and
memory (2^(n/2)) are where HSF stays clearly ahead. A fair comparison against
the blocked executor is the next measurement to make.

## Not done (agent hit its time limit)

- The 32–40-qubit timing and memory sweep. Only the 40-qubit correctness test
  exists.
- Crossover points for k beyond 10 and for other n.
- Integration with the `Op` variants added on main (Reset, noise, classical
  control). HSF should reject them like measurements.

Timings are from a shared VM with other agents compiling, so treat the ratios
as indicative. The exactness results don't depend on load.
