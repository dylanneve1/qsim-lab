# ZX simplification and general slicing for IBM's doped-Clifford circuit (issue 228): cost study

Agent agt_ce8091d1, 2026-10-07. VPS only. Every run was capped at 1.1–1.2 GB address space, nice, 1 thread.
Only path search and cost estimation; the only contractions were small validation instances.
Circuit: `nq70_depth70_checks27_doped.qasm` (70 qubits, open chain, NN brickwork CZ, depth 70, 468 rz(pi/4)).
Depth truncation: keep CZ layers <= D, plus the 1q layer right after layer D (`circ.load`). T count: D=40 108, D=48 128, D=56 159, D=70 468.
Cost unit **C** = cotengra contraction cost = sum over pairwise contractions of prod(index sizes), times the number of slices.
That is roughly complex multiply-adds, so real FLOPs ≈ 8C.

## TL;DR
* **ZX route (A): dead.** full_reduce shrinks 13.6k spiders to 516 (379 TN indices) at D=70, but the reduced graph is dense
  (13k Hadamard edges). Its treewidth lower bound (MMD+) is **104** at D=70 and 40 at D=40, against 35 and 20 for the chain sweep.
  Pivoting and local complementation destroy the 1D locality that makes the raw network cheap.
* **General slicing (B): dead.** To get below the chain-sweep width you have to slice about one index per cut along the chain.
  Even one bit of width costs about 2^20–2^35 in total flops. Best width-31 path found at D=70: **2^101.5 total** (2^61 slices).
  For comparison, the unsliced width-37 sweep costs 2^49.1.
* **Slice-dropping fidelity = fraction kept** holds *exactly* on this circuit for arbitrary sliced index sets: cotengra-chosen
  or random inner indices, up to 256 slices, with orthogonality and equal norms to 1e-16. A Haar-random-1q control on the same skeleton
  shows it only on average. It does not help, because the cost at fidelity f is ≥ f × (unsliced optimum), and slicing overhead ≫ 1/f.
* **Verdict: no route here is < 1e16 total flops.** Even an impossible zero-overhead slicing at f=0.1 needs ≥ 4.7e14 flops per
  amplitude, so ≥ 1.4e18 for 2900 samples. Exact sampling (F≈1, ~10–20 samples) needs ~5e16–1e17 flops *and* 256 GiB–1 TiB of memory
  (SUTD's route).

## A. ZX route
Pipeline (`circ.py`): gate list → pyzx Circuit (h→HAD, s→Z(π/2), t→Z(π/4), sx/sxdg→X(±π/2), CZ) → `apply_state('0'*n)`,
`apply_effect(x)` → `full_reduce` → TN. Each spider is a hyperindex; phase → vector (1, e^{iα}); Hadamard edge → H/√2 matrix;
X spiders get H on all legs; the scalar comes from `g.scalar`. Then quimb `full_simplify`, then cotengra.

Validation (`val.py`, `val.log`) against a numpy state vector: n=16 D=70 (59 T, 2935 → 73 spiders) and n=20 D=70 (69 T, 3721 → 85 spiders),
3 random bitstrings each: amplitude **and phase** match (|ratio| = 1.000000, phase diff 0). n=16 D=40, n=20 D=56 and n=10 D=12 have no T on those qubits, and the diagram collapses
to the exact scalar, also matching. (`teleport_reduce` returns the unreduced graph here, so it was dropped.)
pyzx's `scalar.to_number()` overflows at n=70; the scalar is irrelevant for costing.

| D | T | spiders/edges before | after full_reduce | TN indices | non-Clifford spiders | tw upper (min-fill) | **tw lower (MMD+)** | cotengra 60 s |
|---|---|---|---|---|---|---|---|---|
| 40 | 108 | 7614 / 8924 | 114 / 1708 | 106 | 107 | 62–65 | **40** | width 68, 2^73.2 |
| 48 | 128 | 9098 / 10684 | 137 / 2508 | 125 | 127 | 76–77 | **46** | – |
| 56 | 159 | 10590 / 12452 | 167 / 4081 | 155 | 158 | 94–102 | **58** | – |
| 70 | 468 | 13557 / 15902 | 516 / 13188 | 379 | 449 | 170–204 | **104** | – |

(interior_clifford_simp / clifford_simp alone: D=70 535 spiders, MMD+ 116. The spider-fused-only graph is the raw network.)
Every tensor in the ZX network has rank ≤ 2 on spider hyperindices, so the contraction width is ≈ the treewidth of the spider graph.
At D=70, width ≥ ~104 means slicing to 31 needs ≥ 2^73 slices, so ≥ ~2^104 total. **ZX is strictly worse than the raw network at every D.**

## B. General slicing (raw quimb network; full_simplify gives rank-2 tensors on degree-3 hyperindices)
Trees:
* *chain seed*: tensors sorted by (qubit, gate), accumulated linearly, plus subtree reconfiguration. Widths 22/26/30/37 at D=40/48/56/70.
  That is D/2+2: my network is slightly worse than qsim-lab's analytic split-CZ sweep (D/2).
* *kahypar*: cotengra HyperOptimizer (kahypar+greedy, optuna, minimize=flops). Budgets: D=40 300 s (14 trials), D=48 600 s (14),
  D=56 600 s (13), D=70 300 s (7) and **1500 s (19)**. The box was loaded (~50% of a core each). Unsliced bests: D=40 w28 2^37.0,
  D=48 w37 2^43.8, D=56 w39 2^48.7, D=70 w47 2^58.1. Generic search never matches the chain sweep's width.
* Slicing: `lean.py`. Slice one bit at a time (cotengra SliceFinder, max_repeats=2) and run `subtree_reconfigure` (size 6) after each step.
  This is cotengra's slice_and_reconfigure with step 1, bounded memory.
  cotengra's in-search `slicing_reconf` (HyperOptimizer) could not be used. Default settings hit MemoryError under the 1.1 GB cap.
  The lean variant ran 1800–3000 s without finishing a single D=70 trial. Its one D=40 result, after a single trial in 900 s, was width 18, 2^150.5.

Best total cost found (min over all trees), log2 C (log2 slices, tree):

| D | chain-sweep width / log2C unsliced | @ width ≤ 31 | @ width ≤ 30 | @ width ≤ 28 |
|---|---|---|---|---|
| 40 | 22 / 33.0 | 33.0 (0, chain) | 33.0 (0, chain) | 33.0 (0, chain) |
| 48 | 26 / 37.4 | 37.4 (0, chain) | 37.4 (0, chain) | 37.4 (0, chain) |
| 56 | 30 / 41.7 | 41.7 (0, chain) | 41.7 (0, chain) | 96.4 (60, kahypar) |
| **70** | 37 / 49.1 | **101.5 (61, kahypar)** | **109.1 (70, kahypar)** | **121.0 (84, kahypar)** |

Slicing overhead to go k bits below the chain width (log2 of best total / unsliced chain cost):

| D | w0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=7 |
|---|---|---|---|---|---|---|---|
| 40 | 22 | +20.5 | +45.2 | +60.6 | +67.9 | +86.4 | +114.2 |
| 48 | 26 | +33.0 | +46.3 | +53.3 | +60.4 | +69.3 | +93.0 |
| 56 | 30 | +30.3 | +54.7 | +56.4 | +63.2 | +70.2 | +89.4 |
| 70 | 37 | +35.3 | +36.9 | +42.9 | +46.0 | +49.2 | +60.0 |

The chain seed sliced directly is far worse (D=70: width 31 needs 2^327 slices, 2^370 total), which confirms PLAN.md.
The 300 s → 1500 s kahypar search moved D=70 @31 from 2^147.6 to 2^101.5, so the numbers are search-limited upper bounds.
They cannot fall below the unsliced optimum (2^49), and the k=1 data (+20–35 bits for a single bit of width) show the geometry.
The network is a 70 (chain) × D/2 (CZ bonds per cut) grid. Any balanced separator crosses ≥ D/2 bonds, and a sliced bond only
helps separators within a column or two of it. So you have to slice ~k indices per cut along the whole chain.

## C. Slice-dropping fidelity
Argument: an inner index of the simplified TN is the Z-basis value of one qubit wire on a segment between non-diagonal gates
(a CZ bond is the control's Z value). Fixing it inserts a projector, ψ_s = U_> Π_s U_< |0>. For one index, slices are always
orthogonal (Π_0Π_1 = 0), with norms equal to the Born probabilities (50/50 or deterministic for stabilizer-like states).
For several indices at different times, histories can interfere: H·Π_b·H·Π_a·H|0> gives parallel slices. So "fidelity = fraction kept"
is *not* automatic for general slicing. If the slices are orthogonal with equal norms 1/M, keeping K gives F = |Σ_K⟨ψ|ψ_s⟩|²/‖Σ_Kψ_s‖² = K/M exactly.

Test (`slicefid.py`, `slicefid.jsonl`): open-output state TN, contract every slice to a full vector, check that the slices sum
to the state-vector reference, then measure the Gram matrix and F for 40 random subsets per fraction.

| instance | sliced indices | M | max off-diag/mean norm | norm CV | F at 1/2, 1/4, 1/8 |
|---|---|---|---|---|---|
| IBM n=12 D=70 (48 T) | cotengra (5) | 32 | 1.4e-16 | 5e-16 | 0.5000, 0.2500, 0.1250 (σ 1e-16) |
| IBM n=12 D=70 | random inner ×2 | 32 | ≤ 4.3e-16 | ≤ 5e-16 | exact, same |
| IBM n=14 D=70 (53 T) | cotengra (8) | 256 | 5.4e-16 | 6.5e-16 | 0.5000, 0.2500, 0.1250 exact |
| IBM n=14 D=70 | random inner ×2 | 256 | ≤ 3.6e-16 | ≤ 6.1e-16 | exact |
| control: same skeleton, Haar 1q gates, n=12 D=30 | cotengra / random | 32 | 0.04–0.44 | 0.17–0.73 | 0.50–0.51, 0.25–0.27, 0.127–0.137 (σ 0.01–0.06) |

Relative linear XEB of the dropped-slice distribution tracks F (0.12–0.13 at 1/8).
So on IBM's circuit F = fraction kept holds exactly for *any* sliced index set (empirically, n ≤ 14; I have no proof). It holds only
on average for generic gates. **But it is not the bottleneck.** Dropping slices gives cost = f × M × C_slice ≥ f × C_unsliced(tree)
≥ f × C_opt. It beats the exact sweep only if the slicing overhead O < 1/f ≈ 2^3.3, and the measured O is ≥ 2^20 for a single bit.

## Feasibility (f = 0.1, N ≈ 9/(f−0.044)² ≈ 2900 samples; M1 Pro 1.5 TFLOPS, H100 25 TFLOPS effective c64)
| path | real FLOPs / amplitude (8C) at f=0.1 | × 2900 | M1 Pro | H100 |
|---|---|---|---|---|
| best width-31 path (2^101.5 C) | 2.9e30 | 8.5e33 | 5.7e21 s | 3.4e20 s |
| best width-28 path (2^121.0 C) | 2.1e36 | 6.2e39 | – | – |
| ZX, width ≥104 lower bound, sliced to 31 (≥ 2^104 C) | ≥ 1.6e31 | ≥ 4.7e34 | – | – |
| *ideal* zero-overhead slicing of the 2^49.1 sweep (impossible, per B) | 4.7e14 | 1.4e18 | 310 s/amp, 250 h | 19 s/amp, 15 h |
| exact width-35/37 sweep, F≈1, N≈10–20, needs 256 GiB–1 TiB | 4.7e15 (f=1) | 5e16–1e17 | memory-infeasible | memory-infeasible (80 GB) |

Batch tricks: in a chain sweep, leaving the last k chain-end output bits open costs +k width only over the last k columns.
That gives 2^k correlated amplitudes per contraction, and frugal rejection sampling over the batch then yields ~1 sample per contraction.
This removes the "amplitudes per sample" factor, but not the cost per contraction or the memory.

## D. Verdict
No. Neither ZX simplification (width ≥ ~104 at D=70) nor hyper-optimised general slicing (≥ 2^101.5 C at width 31, best found
with a 25 min kahypar search plus slice-and-reconfigure) comes anywhere near 1e16. Slice-dropping fidelity is exact here, but cost(f) ≥ f × 2^49 C
bounds every slicing route from below at ≥ 1.4e18 flops for 2900 samples at f=0.1, even with zero overhead. The only sub-1e17 route
is still the exact width-35 sweep with ~10–20 exact samples, and that needs ≥ 256 GiB. These are the same memory walls PLAN.md already lists.

## Files
`circ.py` (loader, state vector, raw/ZX TN builders), `val.py` (ZX/raw vs state vector), `build.py` (n=70 networks → net_D*.pkl),
`tw.py`/`tw2.py` (ZX treewidth bounds), `search.py` (cotengra HyperOptimizer), `seed.py`/`trend.py` (chain seed),
`lean.py` (slice-and-reconfigure per bit), `pipe.sh` (search → lean), `slicefid.py` (slice-fidelity test), `table.py`.
Results: `results.jsonl` (searches), `lean.jsonl` (all slicing curves), `slicefid.jsonl`, `trend.jsonl`, `val.log`, `*.log`.
