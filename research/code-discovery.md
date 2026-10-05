# Code discovery: exhaustive search of weight-6 two-block (BB / GB / coprime-BB) codes, n ≤ 300

Branch `exp/code-discovery`.
- Code:
  - `src/qec/bicycle.rs`: two-block codes over `Z_l × Z_m`, GF(2) rank for k, exact distance, ISD upper bound.
  - `src/qec/bb_search.rs`: enumeration up to equivalence.
  - `src/qec/bb_circuit.rs`: depth-7 syndrome circuits and schedule validity.
  - `examples/bb_codes.rs`: CLI with `params`, `search`, `schedules`, `schedsearch`, `cdist`, `ler`.
- Tests: `tests/bicycle_codes.rs` pins published parameters. There are also unit tests in all three modules, including a brute-force check that the enumeration is complete.
- Data and scripts: `research/data/code-discovery/`. `literature.md` is a 341-entry table of published BB/GB/2BGA codes from 21 papers. It was compiled by a sub-agent, with every polynomial copied verbatim.
- Compute: all runs were on the Mac (M1 Pro) with ≤ 2 workers and no GPU. The Mac was shared, with load 8–16.

**Question.** Can a search over code families, scored with our fast exact tools, find quantum LDPC codes that beat known ones per physical qubit under circuit-level noise? The baselines are IBM's bivariate-bicycle codes and the 2024–26 code tables.

**Answer.**
- **Code capacity: no code in this space beats the literature's best k·d²/n at any n.** The space is every weight-6 two-block group-algebra code over every abelian group of rank ≤ 2, with n ≤ 300 — 147,107 inequivalent classes with k > 0.
  - Wherever our search decided the frontier, it matches the maximum k·d²/n of the published tables. The published frontier is mostly Liang et al. 2503.03827, IBM, and Lin–Pryadko.
  - It also reproduces every published entry in the space that it decided.
  - Two Pareto points are not in any table we found and are not matched by direct sums of published codes: **[[168,14,10]]** and **[[300,16,14]]** (polynomials below). Both have lower k·d²/n than their published neighbours.
  - A third code, **[[112,12,8]]**, has the same parameters as two copies of the published [[56,6,8]].
- **Circuit level: the same parameters do not mean the same performance.**
  - Setup: depth-7 CNOT schedules (IBM's generalised), uniform circuit noise, BP+OSD-CS (order 10 unless stated), Z memory.
  - The connected **[[112,12,8]] has 4–10× lower logical error per logical qubit than the published [[56,6,8]]** (that is, than two copies of it, which is the same qubit overhead n/k = 9.3 and the same d):
    - p = 0.3%: 3.9×;
    - p = 0.2%: 5.3–6.0× at OSD order 10/40/100;
    - p = 0.15%: 9.4–9.6×.
  - The ratio holds at every OSD width tried, although the absolute rates fall about 6× from order 10 to order 100.
  - **No depth-7 schedule of this shape keeps d_circ = 8 for [[56,6,8]].** 75 of the 936 valid schedules do for [[112,12,8]].
  - [[168,14,10]] has the same n/k = 12 as the gross code. At p = 0.3% it has 5.0e-5 failures per logical qubit per round against the gross code's 6.6e-5. The 95% intervals barely overlap. Its d is 10 against 12, so the gross code must win at lower p; we could not afford the shot counts to show the crossover. The published [[170,16,10]] does as well as [[168,14,10]] (4.8e-5) at a better n/k, so [[168,14,10]] is new but not better.
- **Tools.**
  - Exact distance via connected-cluster branch and bound, with the group's translation symmetry used to fix the root. It takes 21 ms for the gross code [[144,12,12]], 4.7 s for [[288,12,18]] and 32 s for [[294,10,20]].
  - Exact circuit distance uses the same search on the circuit DEM with translation orbits as roots. It proves d_circ = 6 for IBM's [[72,12,6]] circuit in 0.1 s (IBM gives ≤ 6) and d_circ ≥ 9 for the gross-code circuit (IBM gives ≤ 10).


---

## 1. Codes, conventions, equivalence

A two-block code is defined over a group `G` with `|G| = N` and has `n = 2N` qubits in two blocks, L and R.

**Matrices.** `H_X = [A | B]` and `H_Z = [Bᵀ | Aᵀ]`, where `A` and `B` are sums of group elements (monomials) acting as permutation matrices.
- On `Z_l × Z_m` we use `x = S_l ⊗ I_m` and `y = I_l ⊗ S_m`, with `S[i][i+1] = 1`. This is exactly the convention of Bravyi et al. and of their `decoder_setup.py`.
- Special cases:
  - BB codes: `m > 1`;
  - generalised-bicycle (GB) codes: `G` cyclic;
  - coprime-BB codes: `gcd(l, m) = 1`, so `G` is cyclic and they are GB codes.

**Search space.** Every abelian group of rank ≤ 2 is `Z_l × Z_m` with `m | l` (invariant factors), so each isomorphism class is listed once. Order 72, for example, gives Z72, Z36×Z2, Z24×Z3 and Z12×Z6.

**Equivalences used.** These maps preserve [[n,k,d]] and the Tanner graph:
- independent translations `A → A·t` and `B → B·s` (proof in `bb_search.rs`: relabel L by `t⁻¹` and the Z checks by `t`);
- any group automorphism `σ` applied to both `A` and `B`;
- the swap `(A, B) → (B, A)`.

**Enumeration.**
- `A` runs over canonical representatives of `{σ(A·t)}`, and `B` over all weight-3 sets containing 1.
- Every pair with `k > 0` is reduced to a canonical key (the minimum over the whole equivalence group) and deduplicated.
- The test `enumeration_finds_every_class` brute-forces all pairs of 3-sets for six small groups and checks that exactly the same classes come out.
- `Aut(G)` is computed by brute force over generator images; the test checks `|Aut|` against φ(n) and |GL(2,p)|.

**Decomposable codes.** If `⟨A ∪ B⟩ ≠ G`, the code is `|G : H|` disjoint copies of a smaller code. Such codes are reported but never claimed. Example: the published [[288,24,12]] is two gross codes.

## 2. Exact k and d

- **k** is `n − rank H_X − rank H_Z` over GF(2). For abelian `G`, `rank H_Z = rank H_X`, so the search uses `k = 2(N − rank[A|B])` with stack-allocated 512-bit rows.
- **d** comes from `qec::bicycle::min_weight_logical`, an exact branch and bound in the style of `qec::distance`, but over code qubits with `u128` logical masks:
  - Nontriviality is tested against `k` conjugate logicals, a basis of `ker H_Z mod rowspace H_X`.
  - A minimum-weight nontrivial logical has no proper non-empty subset in `ker H_X`. Its support is therefore reached by repeatedly picking an unsatisfied check, taking its allowed qubit with the fewest options, and branching. Already-tried siblings are banned, so the branches partition the solutions.
  - Two bounds prune the search: `|S| + ⌈|F| / colweight⌉`, and a greedy bound on check sets that pairwise share no qubit.
  - Translations act transitively on L and on R. It is therefore enough to root the search at `L0`, and then at `R0` with all of L banned.
- The test `symmetric_roots_match_plain_search` compares the two-root search with an all-roots search and with brute force on random small codes. `toric_code_as_bb` checks `[[2L², 2, L]]`.
- **Upper bound:** a randomized information-set search (Lee–Brickell, p ≤ 2) gives a witness. It costs about 0.15 ms per iteration at n = 216. With 200 iterations it is usually already tight.

**Reproduction of published parameters** (exact; times are one M1 core under load). Both CSS distances were checked with `BOTH=1`.

| code | group, A, B | our k, d | time | source |
|---|---|---|---|---|
| [[72,12,6]] | Z6×Z6, x³+y+y², y³+x+x² | 12, 6 | 11 ms | Bravyi et al. Table 3 |
| [[90,8,10]] | Z15×Z3, x⁹+y+y², 1+x²+x⁷ | 8, 10 | 9 ms | " |
| [[108,8,10]] | Z9×Z6, x³+y+y², y³+x+x² | 8, 10 | 8 ms | " |
| [[144,12,12]] | Z12×Z6, x³+y+y², y³+x+x² | 12, 12 | 21 ms | " |
| [[288,12,18]] | Z12×Z12, x³+y²+y⁷, y³+x+x² | 12, 18 | 4.7 s | " |
| [[72,4,10]], [[126,12,10]], [[154,6,16]], [[180,8,16]], [[210,14,12]], [[254,14,16]], [[294,10,20]] | cyclic GB, Liang et al. T5–8 | all exact matches | 5 ms – 32 s | 2503.03827 |
| [[174,4,18]], [[182,6,18]], [[204,4,20]], [[216,4,20]], [[222,4,20]], [[224,6,20]], [[228,4,20]] | GB / twisted-torus codes mapped to Z_N | all exact matches | 7 s – 17 min | 2503.03827 |

**Mapping twisted-torus codes to `Z_N`.** The twisted-torus codes `Z²/Λ` are cyclic whenever the gcd of the entries of Λ is 1. The isomorphism is `(a, b) → a·u + b·v mod N`. Example: `Z²/⟨(0,3),(29,1)⟩ → Z87` with `x → 2`, `y → 29`.

## 3. The search

`bb_codes search <Nmin> <Nmax> 3 3 <worker> <workers> <node_limit> [min_k]`, two workers, about 1.6 h of CPU in total.

**Search volume.** For `N = 4 … 150` (n = 8 … 300), 228 groups were covered:
- 41.9 M (canonical A, B) pairs were rank-tested;
- 147,107 inequivalent classes have k > 0;
- 4,964 were decided exactly;
- 141,913 were pruned: either provably unable to beat the best exact d of their (N, k), or, for k below `min_k`, given only an ISD bound;
- 37 classes have k > 128 (A = B type, d = 2) and were skipped.

**Pass 1 and pass 2.** Classes were processed per (N, k) in decreasing order of the ISD bound.
- If `ub ≤ best exact d so far`, the class cannot improve the frontier.
- Otherwise a single DFS level at `w = best` looks for a logical of weight ≤ best. If one is found, the class is pruned with a witness.
- Only if none exists is the exact distance computed, starting at `best + 1`.

**Exactness coverage.**
- N ≤ 113: every k.
- 114 ≤ N ≤ 139: k ≥ 6.
- 140 ≤ N ≤ 150: k ≥ 8.
- Smaller k in those ranges has only the ISD upper bound. Those codes have k·d²/n ≤ 10.7, below the best published value at those n, so they cannot change the per-qubit answer.
- 193 searches hit their node limit (5·10⁷–3·10⁸). The ones covering published entries were re-run with a much larger limit (§3.2).

### 3.1 Frontier vs literature

- `research/data/code-discovery/frontier_w6.md` has the full per-n frontier: every (n, k, d) not dominated at the same n, with polynomials and literature status.
- `frontier_w6.png`: left, the (n, d) frontier points, marker size ∝ k; right, the best k·d²/n per n, ours vs published.
- Every frontier point was compared with the 341 published entries **and with every direct sum of up to 5 of them** (`[[n₁+n₂, k₁+k₂, min d]]`).
- Decided frontier points split into three groups:
  - **Published, or dominated by a published code at equal or smaller n.** This is every decided frontier point except the ones below.
  - **Dominated by a direct sum of published codes, with equal parameters.** [[56,12,4]], [[98,18,4]] and [[112,12,8]] = 2×[[56,6,8]]; [[42,12,2]].
  - **New Pareto points:**

| code | group | A | B | k·d²/n | nearest published |
|---|---|---|---|---|---|
| **[[168,14,10]]** | Z42×Z2 | 1 + x + x⁵y | 1 + x²y + x³¹y | 8.33 | [[170,16,10]] (9.41), [[186,14,10]] |
| **[[300,16,14]]** | Z30×Z5 | 1 + y + x⁵y³ | 1 + x + x⁴ | 10.45 | [[288,12,18]] (13.5), [[294,18,10]] |

Both are verified exactly: k by rank, and d in both CSS sectors by the exact search, in 26 ms and 116 ms. Neither beats the k·d²/n of its published neighbours. They fill gaps in the (n, k, d) table we compiled:
- [[168,14,10]] is the smallest code there with k ≥ 14 and d ≥ 10 (the next is [[170,16,10]]);
- [[300,16,14]] is the only code there with k ≥ 16 and d ≥ 14 at n ≤ 300, counting direct sums.

**Best k·d²/n per n.** Wherever the frontier was decided, our best equals the literature's best, which is mostly Liang et al.'s twisted-torus and GB tables. Liang et al. claim optimality only inside the family `f = 1 + x + x^a y^b`, `g = 1 + y + x^c y^d`; this search extends that to all weight-6 abelian two-block codes. After the certification runs (§3.2), the only n where we have no decided value are n = 246, 258, 276 and 282. At those n the published best is a k = 4 code, which our run did not decide (k < `min_k`); see `frontier_w6.png`.

### 3.2 Literature upper bounds, certified

Published entries listed only as upper bounds, and our own node-limit aborts, were re-run with `params` and no practical node limit (3·10¹⁰ nodes). Raw data: `certify.jsonl`. The frontier files include these results via `certified.jsonl`.

| published entry | as a Z_N code (A ; B) | result | nodes | time |
|---|---|---|---|---|
| [[266,6,≤22]] (twisted torus) | Z133: 1+x⁵+x¹⁰⁹ ; 1+x¹⁹+x²⁵ | **d = 22 exactly** | 2.8·10⁹ | 503 s |
| [[280,6,≤22]] (twisted torus) | Z140: 1+x¹⁶+x³¹ ; 1+x⁵+x²² | **d = 22 exactly** | 2.2·10⁹ | 324 s |
| [[300,8,≤22]] (twisted torus) | Z150: 1+x⁴⁹+x⁹³ ; 1+x²+x⁹ | **d = 22 exactly** | 3.0·10⁹ | 399 s |
| [[234,4,≤22]] (GB, Liang T5–8) | Z117: 1+x¹³+x²⁹ ; 1+x+x²⁰ | **d = 22 exactly** | 2.8·10⁹ | 350 s |
| [[264,4,≤22]], [[288,4,≤24]] and the other k = 4 twisted-torus entries | — | not run (out of time) | | |

These four upper bounds of Liang et al. are tight. For the other classes at the same (n, k), our search left some undecided (ISD bound 22–24), so "no better code exists at that (n, k)" is not claimed. The twisted-torus codes were mapped to `Z_N` via `Z²/Λ ≅ Z_N` (each Λ has coprime entries). For example `Z²/⟨(0,7),(19,2)⟩ → Z133` with `x → 5`, `y → 19`, giving f = 1 + x + x⁻¹y⁻¹ → {0, 5, 109}. k was checked to match the paper in every case.


## 4. Circuit level

### 4.1 Syndrome circuits and schedules

- `bb_circuit` builds IBM-style memory experiments for any weight-6 two-block code:
  - one X ancilla and one Z ancilla per check;
  - 7 CNOT layers; in each layer an X-check term and a Z-check term act on opposite data blocks;
  - prepare and measure in each cycle;
  - final data measurement in the memory basis, with k observables (a basis of logicals of the memory type).
- **Noise ("uniform", as in Bravyi et al.):**
  - two-qubit depolarizing p after every CNOT;
  - single-qubit depolarizing p on every qubit idle in a CNOT layer;
  - preparation flip p and readout flip p;
  - the basis-change Hadamards are noiseless.
- **Schedule validity.** For every X-check/Z-check pair, the number of shared qubits that the X check touches first must be even. This is checked combinatorially (`schedule_valid`) and confirmed by the simulator: an invalid variant makes `circuit_dem` see random detectors (test `invalid_schedule_is_rejected_by_both_checks`).
- With "X idle in layer 0, Z idle in layer 6", every code we tried has exactly **936 valid schedules**, and IBM's (`-143502/350124-`) is one of them.

**Circuit distance.**
- `cdist` / `schedsearch` run the exact search on the decoded-sector DEM: Z-check detectors for Z memory, with all faults projected onto them.
- They use the circuit's translation symmetry: mechanisms are grouped into orbits under `detector (r, h) → (r, h + g)`, there is one root per orbit, and each root bans all earlier orbits. This is a 30× speed-up on [[72,12,6]].
- 3 rounds unless stated; both memory bases.

| code | IBM schedule d_circ (Z/X) | schedules with d_circ = d | distribution over all 936 |
|---|---|---|---|
| [[72,12,6]] | 6 / 6 (also 6 at 6 rounds) | 792 | (6,6) 792, (5,5) 60, (6,5)/(5,6) 48, (3,3) 24, (4,4) 12 |
| [[56,6,8]] | 6 / 6 | **0** (best is 7 / 7, 220 schedules) | (6,6) 591, (7,7) 220, … |
| [[112,12,8]] | 7 / 7 | **75** | (7,7) 467, (6,6) 167, (7,8)/(8,7) 202, (8,8) 75 |
| [[144,12,12]] | ≥ 9 (Z), proven at 10⁹ nodes; IBM state ≤ 10 | not searched | — |

Raw data: `sched_*.jsonl`.

### 4.2 Logical error rates

- **Setup.** Z memory, `rounds` cycles, uniform noise p. FastSampler shots are decoded by BP+OSD-CS:
  - min-sum with scale 0.625, 100 iterations;
  - OSD order 10;
  - Z-sector DEM, merged by (detectors, observable mask).
- **Failure** means any of the k observables is wrong.
- **Per-round rate:** `1 − (1 − p_L)^{1/rounds}`, with Wilson 95% intervals.
- Single-threaded runs on the Mac. Raw data: `ler.jsonl`. Plot: `ler_112_vs_56.png`, per-logical error vs p for both codes at OSD 10/40/100/250.

| code | schedule (d_circ Z/X) | basis | rounds | p | shots | fails | block p_L per round [95% CI] | per logical qubit per round | OSD calls | decode s |
|---|---|---|---|---|---|---|---|---|---|---|
| [[72,12,6]] IBM | IBM (6/6) | Z | 6 | 0.3% | 20,480 | 633 | 5.22e-3 [4.83, 5.64] | 4.35e-4 | 33% | 36 |
| [[72,12,6]] IBM | IBM (6/6) | Z | 6 | 0.2% | 20,480 | 156 | 1.27e-3 [1.09, 1.49] | 1.06e-4 | 22% | 30 |
| [[56,6,8]] published | -034521/452013- (7/7, best) | Z | 8 | 0.3% | 20,480 | 882 | 5.49e-3 [5.14, 5.86] | 9.15e-4 | 38% | 45 |
| [[56,6,8]] | -034521/452013- (7/7) | Z | 8 | 0.2% | 40,960 | 414 | 1.27e-3 [1.15, 1.40] | 2.12e-4 | 24% | 57 |
| [[56,6,8]] | -034521/452013- (7/7) | X | 8 | 0.2% | 40,960 | 398 | 1.22e-3 [1.11, 1.35] | 2.03e-4 | 24% | 54 |
| [[56,6,8]] | IBM (6/6) | Z | 8 | 0.2% | 40,960 | 671 | 2.06e-3 [1.91, 2.22] | 3.44e-4 | 23% | 49 |
| [[56,6,8]] | -034521/452013- (7/7) | Z | 8 | 0.15% | 81,920 | 348 | 5.32e-4 [4.79, 5.91] | 8.87e-5 | 18% | 74 |
| **[[112,12,8]]** | -034521/452103- (8/8) | Z | 8 | 0.3% | 20,480 | 454 | 2.80e-3 [2.55, 3.07] | 2.33e-4 | 58% | 148 |
| **[[112,12,8]]** | IBM (7/7) | Z | 8 | 0.3% | 20,480 | 435 | 2.68e-3 [2.44, 2.94] | 2.23e-4 | 57% | 142 |
| **[[112,12,8]]** | -034521/452103- (8/8) | Z | 8 | 0.2% | 20,480 | 72 | 4.40e-4 [3.50, 5.54] | 3.67e-5 | 41% | 107 |
| **[[112,12,8]]** | IBM (7/7) | Z | 8 | 0.2% | 20,480 | 67 | 4.10e-4 [3.23, 5.20] | 3.41e-5 | 39% | 102 |
| **[[112,12,8]]** | -034521/452103- (8/8) | X | 8 | 0.2% | 40,960 | 129 | 3.94e-4 [3.32, 4.68] | 3.29e-5 | 40% | 193 |
| **[[112,12,8]]** | -034521/452103- (8/8) | Z | 8 | 0.15% | 81,920 | 74 | 1.13e-4 [0.90, 1.42] | 9.41e-6 | 31% | 279 |
| [[144,12,12]] gross | IBM (≥9/–) | Z | 12 | 0.3% | 16,384 | 155 | 7.92e-4 [6.77, 9.27] | 6.60e-5 | 79% | 363 |
| **[[168,14,10]]** | IBM | Z | 10 | 0.3% | 16,384 | 114 | 6.98e-4 [5.81, 8.38] | 4.99e-5 | 79% | 460 |
| [[170,16,10]] published GB | IBM | Z | 10 | 0.3% | 16,384 | 126 | 7.72e-4 [6.48, 9.19] | 4.82e-5 | 79% | 421 |

**Decoder sensitivity.** BP+OSD-CS with a larger OSD combination width, at the best schedules, Z memory, 8 rounds:

| code | p | OSD order | shots | fails | block p_L per round [95% CI] | per logical qubit |
|---|---|---|---|---|---|---|
| [[56,6,8]] | 0.2% | 10 | 40,960 | 414 | 1.27e-3 [1.15, 1.40] | 2.12e-4 |
| [[56,6,8]] | 0.2% | 40 | 40,960 | 151 | 4.62e-4 [3.94, 5.41] | 7.69e-5 |
| [[56,6,8]] | 0.2% | 100 | 40,960 | 71 | 2.17e-4 [1.72, 2.73] | 3.61e-5 |
| [[56,6,8]] | 0.2% | 250 | 40,960 | 60 | 1.83e-4 [1.42, 2.36] | 3.05e-5 |
| **[[112,12,8]]** | 0.2% | 10 | 20,480 | 72 | 4.40e-4 [3.50, 5.54] | 3.67e-5 |
| **[[112,12,8]]** | 0.2% | 40 | 40,960 | 50 | 1.53e-4 [1.16, 2.01] | 1.27e-5 |
| **[[112,12,8]]** | 0.2% | 100 | 40,960 | 27 | 8.24e-5 [5.66, 12.0] | 6.87e-6 |
| [[56,6,8]] | 0.15% | 40 | 81,920 | 91 | 1.39e-4 [1.13, 1.71] | 2.32e-5 |
| **[[112,12,8]]** | 0.15% | 40 | 81,920 | 19 | 2.90e-5 [1.86, 4.53] | 2.42e-6 |

- The OSD-10 numbers above overstate both codes' error rates by about 6×. With order 100–250, [[56,6,8]] is nearly converged: order 250 gains only 15% over order 100.
- At equal decoder settings the per-logical ratio is stable:

| p | OSD order | ratio |
|---|---|---|
| 0.2% | 10 | 5.8× |
| 0.2% | 40 | 6.0× |
| 0.2% | 100 | 5.3× [2.9, 9.7] |
| 0.15% | 40 | 9.6× |

- Even [[56,6,8]] at order 250 is 4.4× worse than [[112,12,8]] at order 100.
- So the advantage of the connected code is not a decoder artefact at these settings. It could shrink with a much stronger decoder (an ML or tensor-network decoder was not tried), but it survives a 25× increase in OSD width.

## 5. Caveats

- **Search space.**
  - Weight 6 (3 + 3) only, abelian groups of rank ≤ 2 only, n ≤ 300.
  - Not searched: rank-3 groups (trivariate / tricycle codes), non-abelian 2BGA codes, weight-8 codes, lifted-product codes.
  - Weight-8 codes reach k·d²/n ≈ 19 in the literature ([[144,14,14]], Symons et al.), well above anything at weight 6.
- **Undecided classes.**
  - k = 4 for N ≥ 114 and k = 6 for N ≥ 140 have only ISD upper bounds.
  - 193 node-limit aborts, mostly k = 4/6 with d ≈ 18–24, are listed in `search_w6_all.jsonl.xz` with `d_lo < d_up`.
  - None of them can change the k·d²/n answer: at those n, k = 4/6 would need d ≥ 25 to reach the published best.
- **Literature table.** It was compiled from 21 papers (`literature.md`). "Not in any table we found" is not proof of novelty. The Lin–Pryadko 2BGA tables and Kovalev–Pryadko GB tables were only partly extracted.
- **Circuit-level noise.** IBM's model packs X preparation and Z measurement into the CNOT layers. Our cycle has separate reset and measure moments with no idle noise on data there, so absolute rates are not directly comparable with Bravyi et al.'s figures. All codes here share the same model and decoder, so relative comparisons are fair.
  - The decoder is BP+OSD-CS order 10 on the Z sector only, so Y correlations are not used. A better decoder could narrow the [[112,12,8]] vs [[56,6,8]] gap.
- **Schedules.** Only the "IBM shape" was searched (X idle first, Z idle last, depth 7). Coloration circuits and depth-8+ schedules were not tried.

## 6. Reproduce

```text
cargo build --release --example bb_codes
B=target/release/examples/bb_codes
BOTH=1 $B params 12 6 "x^3+y+y^2" "y^3+x+x^2"           # [[144,12,12]], 21 ms
$B search 4 150 3 3 0 2 300000000 6 > w0.jsonl          # worker 0 of 2 (even N)
python3 research/data/code-discovery/frontier.py w0.jsonl w1.jsonl --lit research/data/code-discovery/literature.md --out frontier
$B schedsearch 28 2 "1+x+x^3y" "1+x^2+x^20" 3 10        # all 936 schedules, d_circ both bases
$B cdist 12 6 "x^3+y+y^2" "y^3+x+x^2" ibm 3 12 1000000000
$B ler 28 2 "1+x+x^3y" "1+x^2+x^20" -034521/452103- 8 0.002 20480 12 1 10
```

