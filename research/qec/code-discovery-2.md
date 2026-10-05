# Code discovery 2: weight-6 two-block codes over every group of order ≤ 150, and coset codes

Branch `exp/qldpc-x` (base `main` 6b21728), 5 October 2026. Machine: the shared 16-vCPU Xeon (Emerald
Rapids) box; searches ran with at most 4 worker threads under `nice -n 15` while other users kept the
1-minute load at 15–50, so no timing here is a benchmark.

- Code: `src/qec/group_algebra.rs` (finite groups from multiplication tables, two-block group-algebra
  codes over any group, coset codes, enumeration up to equivalence, exact distance with automorphism
  roots); `src/qec/bb_circuit.rs` (the depth-7 syndrome circuits now take any group through the
  `TwoBlockLayout` trait); `examples/group_codes.rs` (`search`, `csearch`, `params`, `cparams`,
  `cdist`, `schedsearch`, `ler`).
- Tests: `tests/qec/group_codes.rs`.
- Data and scripts: [`research/data/code-discovery-2/`](../data/code-discovery-2/).
- Literature: [`literature.md`](../data/code-discovery/literature.md) now has a 2026-10-05
  supplement (489 rows from 17 more papers, non-abelian and coset two-block codes included).

**Headline.**

- An exhaustive search of weight-6 (3 + 3) two-block group-algebra codes over **every group of order
  ≤ 150** that the first study did not cover — 1000 groups from GAP's SmallGroups: all non-abelian
  ones and the abelian ones of rank ≥ 3 — finds **two codes beyond every published or publicly posted
  weight-6 code we could find**, both exactly certified:
  - **[[224,18,12]]** over C7 × ((C4 × C2) ⋊ C2) and C7 × (C4 ⋊ C4) (order 112). It strictly
    dominates the published [[294,18,10]], [[252,14,12]] and [[288,16,12]] codes; k·d²/n = 11.57.
  - **[[288,34,8]]** over A4 × A4 (and SmallGroup(144,193)). It strictly dominates the published
    [[288,32,6]] and [[292,18,8]]; k·d²/n = 7.56.
- It also finds, over ten different groups of order 144, the current weight-6 k·d²/n record
  **[[288,16,16]] (14.22)**, above every weight-6 code in published papers (best [[254,14,16]], 14.11),
  with **d = 16 proved exactly** for the first time. The same parameters had been posted (not
  published) to the Unitary Foundation qLDPC challenge on 2026-07-14 as a 2BGA code over C12 ⋊ C12,
  with d = 16 only an upper bound from randomized search. Likewise [[192,12,14]] (12.25),
  [[192,16,12]] and [[200,16,12]] turned out to be in Lin & Pryadko's public 2BGA dataset (2023,
  randomized distances); we prove their distances exactly.
- Every claimed distance is certified twice: the exact symmetry-rooted branch and bound in Rust
  (both CSS sectors), and an independent C program (`mwlogical.c`, all roots, no symmetry) that proves
  no nontrivial logical of weight < d exists in either sector, with k and weight-d witnesses checked
  by an independent Python script; the main codes are also rebuilt from their written presentations
  alone, or from permutation representations in the tests, and re-proved.
- The non-abelian space reaches the known weight-6 frontier at many (n, k) and, once the 2026
  literature, the Lin–Pryadko dataset and the challenge board are included, goes beyond it only at the
  two points above.

---

## 1. Why non-abelian groups, and what changed in the literature

The first study ([code-discovery.md](code-discovery.md)) searched every weight-6 two-block code over
every abelian group of rank ≤ 2 with n ≤ 300 and found nothing above the published k·d²/n frontier.
Three directions were left open: non-abelian groups, rank-3 abelian groups, and higher weights.

The literature check for this study (supplement in `literature.md`) found that 2026 papers have
started to use exactly the first two:

| code | family | source | k·d²/n |
|---|---|---|---|
| [[224,12,16]] | coset code, G = C7 × ((C4 × C4) ⋊ C2) (order 224), H = C2 not normal | Aydin, Tamo & Barg, arXiv:2606.17268 | 13.71 |
| [[280,12,16]] | two-block group-algebra (2BGA) code over C14 × D10 (order 140) | same | 10.97 |
| [[168,16,10]] | 2BGA over C14 × S3 (order 84) | same | 9.52 |
| [[216,12,14]] | 2BGA over Z2 × Z2 × ((Z3 × Z3) ⋊ Z3) or Z6 × Z6 × Z3 (order 108) | Hirasaki & Lee, arXiv:2607.28621 (polynomials not printed) | 10.89 |
| [[96,8,10]] | coset code, G of order 384, H = C8 | Aydin, Tamo & Barg | 8.33 |
| [[336,12,20]] | 2BGA over a quotient of Z84 ⋊ Z4 (order 168), n > 300 | Qian & Li, arXiv:2608.08996 | 14.29 |

So the published weight-6 frontier now contains non-abelian codes, and "new" has to be judged against
it. Two public sources outside papers matter as much (both found only after the search had produced
its first "new" codes, which they partly overturned — see §4):

- **Lin & Pryadko's dataset** for arXiv:2306.16400 (github.com/QEC-pages/2BGA-codes, 2023): every
  connected 2BGA code over every group of order ≤ 100 (abelian ≤ 50) with total weight ≤ 8, with
  randomized distances. The paper itself has no weight-6 table; the dataset has 30 353 weight-6 rows.
  Its best weight-6 code per (n, k) is in `lin_pryadko_2bga_w6.tsv`.
- **The Unitary Foundation qLDPC challenge board** (github.com/unitaryfoundation/qldpc-challenge,
  read at commit a101778, 2026-10-05): 460 CSS codes with check weight ≤ 6 and n ≤ 300, most with
  randomized upper bounds on d, many posted in 2026 by search campaigns (`qldpc_challenge_w6.tsv`).

All of these are part of the threshold T(n, k) below.

**Reproduction of three of these codes.** The cover codes of Aydin, Tamo & Barg are given as indices
into GAP's `Elements(G)`; `atb_codes.g` rebuilds them in our numbering and `group_codes params`
computes them exactly:

| code | group | k | d_Z / d_X (exact) | time | independent check (`verify_code.py`) |
|---|---|---|---|---|---|
| [[168,16,10]] | SmallGroup(84,13) = C14 × S3 | 16 | 10 / 10 | 0.36 s | no logical of weight ≤ 9 in either sector (325 k nodes each, 4.4 s) |
| [[280,12,16]] | SmallGroup(140,9) = C14 × D10 | 12 | 16 / 16 | 6.8 M nodes | — |
| [[112,12,8]] | SmallGroup(56,8) = C28 × C2 | 12 | 8 / 8 | 0.15 s | no logical of weight ≤ 7 (0.2 s) |

The [[112,12,8]] code over C28 × C2 is the same group and parameters as the "connected [[112,12,8]]"
of the first study, which therefore appeared in print (as a 2-cover of a [[56,12,4]] code) in 2026.
Their coset codes index GAP 4.14's `LeftCosets`, which GAP 4.15.1 (the version used here) does not
have, so those rows could not be rebuilt from their indices.

## 2. Codes, equivalences and the enumeration

**Convention** (`group_algebra.rs`). A group `G` of order N is a multiplication table with identity 0.
For 3-element subsets A, B of G the code has n = 2N qubits in two blocks L, R and

```text
X-check g:  L{g a : a in A}       R{b g : b in B}
Z-check h:  L{b^-1 h : b in B}    R{h a^-1 : a in A}
```

A multiplies from the right and B from the left, so both overlaps of X-check g and Z-check h count the
solutions of h = b g a and every pair of checks commutes. For abelian G = Z_l × Z_m this is exactly
the `qec::bicycle` convention (`H_X = [A|B]`, `H_Z = [Bᵀ|Aᵀ]`, test `matches_bicycle_on_abelian_groups`);
writing elements as their inverses gives Lin & Pryadko's left-A / right-B convention.

**Equivalences.** Each map below is a relabelling of qubits and checks (the proofs are one line each,
in the module docs), so it preserves [[n,k,d]]:

- two-sided translations of A and of B, *independently*: A → uAw, B → vBt;
- an automorphism of G applied to both;
- (A, B) → (B⁻¹, A⁻¹) (sectors kept), and (A, B) → (B, A), (A, B) → (A⁻¹, B⁻¹) (X and Z exchanged).

For non-abelian G the first item is much stronger than in the abelian case: A can be conjugated
without touching B. A *T-class* is a class of subsets under two-sided translation; its members that
contain the identity are the sets c(S s⁻¹) (c an inner automorphism, s ∈ S). There are about
N·|Z(G)|/6 T-classes of 3-subsets, against about N²/6 for an abelian group, so a non-abelian group of
order 144 has only a few thousand inequivalent weight-6 codes.

**Enumeration** (`Enumeration`): all T-classes, then the orbits of pairs of T-classes under the
automorphism generators (from GAP), the swap and the inversion, by union-find. The test
`enumeration_matches_brute_force_orbits` checks, for S3, D4, Z3 ⋊ Z4, Z7 ⋊ Z3, Z6 and Z3 × Z3, that the
number of orbits equals the number of orbits of *all* pairs of 3-subsets under the generators of the
equivalence group (an independent union-find over up to 1.8 M pairs) and that [[n,k,d]] is constant on
random pairs of one orbit; `equivalences_preserve_parameters` applies random equivalences to random
codes and checks (k, d_Z, d_X) transform as claimed.

**Which groups.** GAP 4.15.1 `SmallGroups` (`export_groups.g`): every group of order 6 ≤ N ≤ 150 that
is
- not a 2-group: for a 2-group F₂[G] is local, so an odd-weight element is a unit, H_X and H_Z have
  full rank N and k = 0;
- not abelian of rank ≤ 2 (done in the first study);
- generated by at most 4 elements (`SmallGeneratingSet`): otherwise ⟨A⟩⟨B⟩ ≠ G and the code is a
  disjoint union of smaller codes.

That is **1000 groups** (non-abelian ones and abelian ones of rank ≥ 3), 226 of them of order 96 and
191 of order 144.

**Distance.** k is n − rank H_X − rank H_Z over GF(2) (both ranks computed: for non-abelian G they need
not be equal a priori). Both CSS distances are computed, d = min(d_Z, d_X). The exact search is the
connected-cluster branch and bound of `qec::bicycle`. For abelian groups translations act transitively
on each block and two roots suffice; for non-abelian groups they do not. The code automorphisms used
are L x → t⁻¹xw, R x → vxu⁻¹ for uAw = A and vBt = B; every candidate is checked against the check
sets before use, and the search is rooted at one qubit per orbit (each root banning the earlier
orbits). `symmetric_roots_match_plain_and_brute_force` compares this with the all-roots search and
with brute force on random non-abelian codes.

**Threshold.** A code is *new* only if no known weight-6 code dominates it. `known_codes.py` collects
the published weight-6 frontier (the merged table of `literature.md` plus every weight-6 row of the
two-block sections of its 2026 supplement, taking listed upper bounds at face value), the best
weight-6 code per (n, k) of the Lin–Pryadko dataset, every weight-≤6 CSS code on the qLDPC challenge
board, the exact abelian frontier of the first study, and every direct sum of up to five of these,
and writes T(n, k) = max d′ over known [[n′, k′, d′]] with n′ ≤ n and k′ ≥ k. A code [[n, k, d]] is
beyond the known frontier iff d > T(n, k). The search itself ran with the threshold from the papers
and the first study only (the dataset and the board were found later); raising T can only turn a
`new` class into a tie, so every class reported `new` was re-evaluated against the full threshold
(`threshold.tsv` in the data folder is the full one).

**Classification** (`group_codes search`). For each inequivalent connected class with k > 0 (k ≤ 128):
a randomized information-set search (10 iterations per sector, then 90 more unless a logical of
weight < T was already found); if it finds a nontrivial logical of weight ≤ T the class is `below`
(< T) or `le_T` (= T, an upper bound only). Otherwise one exhaustive DFS level at weight T in both
sectors decides: a logical of weight ≤ T, or none, in which case the class is `new` and its distance is
computed exactly from T + 1. `undecided` means the DFS hit its node limit (2·10⁸ nodes per level, 2·10⁷
in the last runs). The first 325 groups ran with an earlier, slower rule that also proved ties exactly
(`tie` = exhaustive proof that d = T); its output is kept. The runs were restarted several times (to
change the rule, the thread count or the niceness); a group is in the output exactly once, from the
run that completed it (`done_*.txt` lists).

## 3. Coset codes over Z_m × K

Aydin, Tamo & Barg's coset codes generalise 2BGA codes: the qubits and checks are the cosets of a
subgroup H (non-normal, otherwise the code is the 2BGA code of G/H), A ⊂ G acts on one side and
B ⊂ N_G(H) on the other. In this module's convention (`CosetCode`; right cosets Hx):

```text
X-check Hg:  L{H g a}       R{H b g}
Z-check Hh:  L{H b^-1 h}    R{H h a^-1}
```

Both overlaps count the pairs with H b g a = H h, so the checks commute (b normalises H, so left
multiplication by b is well defined on right cosets). The anti-isomorphism x → x⁻¹ maps their
left-coset convention onto this one with the same A and B; checks with H = {1} reproduce `GroupCode`
and checks with H normal reproduce the 2BGA code of G/H (`tests/qec/group_codes.rs::coset_codes`,
which also compares symmetry-rooted and plain distance searches on random coset codes).

Most of their coset codes have the form G = Z_m × K with H ≤ K, e.g. the [[224,12,16]] code
(m = 7, K = (C4 × C4) ⋊ C2 of order 32, H = C2) and [[186,10,14]] (m = 31, K = S3). `group_codes
csearch` enumerates exactly this family: for every non-abelian K of order ≤ 32 (2-groups included,
since with m odd they can give k > 0), every non-normal cyclic H ≤ K of order ≤ 2 up to Aut(K), and
every m with N = m·[K:H] in range, all pairs (T-class of A in G, T-class of B in N_G(H)/H), with
equivalent pairs merged under the automorphisms of Z_m and the automorphism generators of K that fix
H. Pairs whose checks repeat a coset (weight < 6) are dropped. Classification as in §2.

## 4. Certification of new codes

A class is reported `new` by the search only after an exhaustive DFS proves that neither sector has a
nontrivial logical of weight ≤ T(n, k); its exact distance is then computed from T + 1. Every code
claimed below is then checked again by methods that share no code with the search:

1. **Rust exact distance** (`group_codes params`): k by GF(2) rank, d_Z and d_X by the
   symmetry-rooted branch and bound, with a minimum-weight logical of each type as witness.
2. **Independent lower bound** (`mwlogical.c`, C, written separately): all 2N qubits as roots in
   order, earlier roots banned, no symmetry, only the trivial counting bound; it proves that no
   nontrivial logical of weight ≤ d − 1 exists in each sector. Input files in
   [`certificates/`](../data/code-discovery-2/certificates/) (check supports in plain text, built by
   the Python script from the GAP table), run log `mwlogical_runs.txt`.
3. **Independent k and upper bound** (`verify_code.py`, Python): builds H_X, H_Z from the group table,
   checks that all checks commute, computes k by its own elimination, and checks that each Rust
   witness is in ker H and outside the row space of the other check matrix.
4. **Group-free reconstruction** (`tests/qec/group_codes.rs`): each code is rebuilt from a faithful
   permutation representation of its group (degree 10–15, from GAP's
   `SmallerDegreePermutationRepresentation`) and its [[n,k,d]] recomputed exactly; this checks that the
   claimed parameters do not depend on our GAP export or element numbering.

| code | group (GAP id) | k (Rust / Python) | d_Z, d_X (Rust, nodes) | C: no logical ≤ d − 1 (nodes per sector) | witnesses (Python) |
|---|---|---|---|---|---|
| [[288,16,16]] | SmallGroup(144,167) = Z6 × (C3 ⋊ D8) | 16 / 16 | 16, 16 (1.1·10⁷) | ≤ 15: 3.7·10⁸ / 3.8·10⁸ (41 s / 46 s) | weight 16 / 16, nontrivial |
| [[192,12,14]] | SmallGroup(96,17) = C3 ⋊ (Q8 ⋊ C4) | 12 / 12 | 14, 14 (1.0·10⁶) | ≤ 13: 2.9·10⁷ / 2.8·10⁷ | 14 / 14 |
| [[192,16,12]] | SmallGroup(96,12) = C3 ⋊ ((C4 × C4) ⋊ C2) | 16 / 16 | 12, 12 (1.1·10⁵) | ≤ 11: 3.0·10⁶ / 2.8·10⁶ | 12 / 12 |
| [[200,16,12]] | SmallGroup(100,13) = D10 × D10 | 16 / 16 | 12, 12 (1.1·10⁵) | ≤ 11: 2.8·10⁶ / 2.7·10⁶ | 12 / 12 |
| [[224,18,12]] | SmallGroup(112,20) = C7 × ((C4 × C2) ⋊ C2) | 18 / 18 | 12, 12 (9.8·10⁴) | ≤ 11: 3.5·10⁶ / 3.6·10⁶ | 12 / 12 |
| [[288,34,8]] | SmallGroup(144,184) = A4 × A4 | 34 / 34 | 8, 8 (1.3·10³) | ≤ 7: 4.2·10⁴ / 4.3·10⁴ | 8 / 8 |

### 4.1 The [[288,16,16]] code, explicitly

G = Z6 × K with K = SmallGroup(24,8) = C3 ⋊ D8:

```text
K = < a, b, c | a^3 = b^4 = c^2 = 1,  b^-1 a b = a^-1,  c a c = a^-1,  c b c = b^-1 >,   Z6 = < z >
A = { 1,  z^4 c,  z^2 a b^3 c }
B = { 1,  z^4 a b^2,  z^5 b }
X-check g:  L{g x : x in A},  R{y g : y in B}        Z-check h:  L{y^-1 h : y in B},  R{h x^-1 : x in A}
```

Every element is z^i a^j b^k c^l, and (z^i a^j b^k c^l)(z^i' a^j' b^k' c^l') =
z^(i+i') a^(j + (−1)^(k+l) j') b^(k + (−1)^l k') c^(l+l').
`code_288_from_presentation.py` builds the code from these lines alone (no GAP, no table): k = 16
(both ranks 136), all checks commute, and `mwlogical.c` finds weight-16 logicals of both types and
proves that none of weight ≤ 15 exists (3.98·10⁸ / 4.04·10⁸ nodes). In GAP's numbering
(`export_groups.g`) the same code is SmallGroup(144,167), A = {0, 9, 83}, B = {0, 51, 90}.
The same parameters occur over SmallGroup(144,64), (144,151), (144,153) and (144,154) (seven
inequivalent pairs (A, B) in all; whether these codes are isomorphic was not checked).

**Novelty.** A separate literature check (`novelty_288.md`, 25 papers incl. the four arXiv ids not
extracted for the supplement) found no weight-6 code *in a paper* with n ≤ 300 and k·d²/n > 14.11.
The parameters [[288,16,16]] are, however, not new: (i) at weight 6 they were posted to the Unitary
Foundation qLDPC challenge board on 2026-07-14 (`codes/288-16-16.json`, a 2BGA code over the metacyclic
C12 ⋊ C12 found by simulated annealing, d = 16 as an upper bound from 600 k randomized trials); (ii) at
weight 9 there is a quantum Tanner code (Leverrier, Rozendaal & Zémor, arXiv:2512.20532, QDistRnd
estimate), and at weight 7 Qian & Li's [[288,16,18]] (arXiv:2608.08996) is better. What is new here is
the exact distance (two independent proofs), the presentation above, and the fact that the search
finds no weight-6 2BGA code over any group of order 144 with k = 16 and d > 16 (all twelve classes
with d > 12 have d = 16 exactly). We also ran the C verifier on the challenge's own code
(`cert/ch288`, see §4).

### 4.2 The [[192,12,14]] code, explicitly

G = SmallGroup(96,17) = C3 ⋊ (Q8 ⋊ C4):

```text
G = < a, i, j, t | a^3 = t^4 = 1, Q8 = <i, j> (i^2 = j^2 = (ij)^2, i^4 = 1), <t> ∩ Q8 = 1,
                   t^-1 i t = i^-1, t^-1 j t = i j, i a = a i, t a = a t, j^-1 a j = a^-1 >
A = { 1,  j t^2,  a t }
B = { 1,  a i,  a^2 i^-1 t^3 }
```

`code_192_from_presentation.py` rebuilds it from these lines (quaternion arithmetic, no GAP): k = 12
(ranks 90 / 90), all checks commute, and `mwlogical.c` proves d_Z = d_X = 14 (no logical of weight
≤ 13; weight-14 logicals of both types found). In GAP's numbering: SmallGroup(96,17), A = {0, 1, 15},
B = {0, 20, 85}. k·d²/n = 12.25 exceeds the gross code's 12.0, the best weight-6 value with n ≤ 192 in
*papers*, and the code dominates the 2026 [[216,12,14]] lift of Hirasaki & Lee. But a second check
(`novelty_192.md`) found it in Lin & Pryadko's public dataset for arXiv:2306.16400
(github.com/QEC-pages/2BGA-codes, `nonabelian.zip`, SmallGroup(96,17), a = [2,16], b = [21,86], with
a randomized distance; reposted to the qLDPC challenge on 2026-09-18). That dataset enumerates every
non-abelian group of order ≤ 100 at total weight ≤ 8; its best weight-6 code per (n, k) is now part of
our threshold (`lin_pryadko_2bga_w6.tsv`), and with it [[192,16,12]] and [[200,16,12]] are not new
either. The exact distances proved here are.

## 5. Results of the search

(final numbers pending: the search was still running when this was drafted)

RESULTS_TABLES_PLACEHOLDER

**Where the new and record codes live.** All codes beyond the 2023–2025 frontier come from groups of
order 96–144 with a large abelian or direct-product part: C3 ⋊ (Q8 ⋊ C4) and C3 ⋊ ((C4 × C4) ⋊ C2)
(order 96), D10 × D10 and C10 × D10 (100), C7 × ((C4 × C2) ⋊ C2) and C7 × (C4 ⋊ C4) (112), and for
[[288,16,16]] ten groups of order 144, all of the form (C3 × C3) ⋊ (2-group) or C3 × (…) or
C6 × (C3 ⋊ D8). Most non-abelian groups give nothing competitive: with small centre the
two-sided-translation classes are few and the codes either have k = 0 or are disconnected (⟨A⟩⟨B⟩ ≠ G
for three quarters of the k > 0 classes).

**Rediscoveries** (the search finds the published non-abelian 2BGA codes in its space): the
[[168,16,10]] cover code of Aydin–Tamo–Barg in SmallGroup(84,13) (as a class reaching T), the
[[216,12,14]] lifts of Hirasaki & Lee in four groups of order 108, the [[112,12,8]] of the first study
in two non-abelian groups of order 56, and Lin & Pryadko's dataset codes [[192,12,14]], [[192,16,12]]
and [[200,16,12]] (their groups exactly: SmallGroup(96,17), (96,12), (100,13)).

## 6. Circuit level

**Schedule.** The depth-7 schedule of Bravyi et al. (`-143502/350124-`, X ancillas idle in the first
CNOT layer, Z ancillas in the last) is valid for [[288,16,16]] and [[192,12,14]]: `schedule_valid`
accepts it (every X-check/Z-check pair has an even number of shared qubits that the X check touches
first) and the circuit's detector error model builds without non-deterministic detectors (the DEM
builder panics otherwise). The memory circuits use one X and one Z ancilla per check (4N qubits:
576 for n = 288), uniform circuit noise (two-qubit depolarizing p after every CNOT, single-qubit p on
idle qubits, preparation and readout flips p), and BP+OSD-CS (min-sum, scale 0.625, 100 iterations,
OSD order 10) on the Z sector, exactly as in the first study.

(results pending)

## 7. Caveats and negative results

- **Weight 6 only, and "beyond the frontier" means beyond the tables we compiled.** The threshold
  T(n, k) uses the 21 papers of the first study, the 17 of the 2026-10-05 supplement and the first
  study's own codes; a separate check for [[288,16,16]] read 25 more (`novelty_288.md`). Published
  upper bounds were taken at face value (this can only make "new" harder to reach). Papers that give
  codes only in figures or ancillary files could still contain one of these codes. At higher check
  weight the same [[288,16,16]] parameters are published (weight 9), and weight 7 does better.
- **Small margins.** [[288,16,16]] beats the weight-6 [[254,14,16]] by 0.8 % in k·d²/n, and
  [[192,12,14]] beats the gross code by 2 %. They are new points of the weight-6 trade-off, not a
  change of scale.
- **`le_T` is an upper bound.** Under the fast rule a class whose best logical found has weight T is
  only known to have d ≤ T; whether d = T was proved only for the first 325 groups (`tie`).
- **Undecided classes.** Classes where the DFS at weight T hit its node limit are listed in §5 with
  their bounds; some of them (k = 8, n = 288, d ∈ [20, 22]; k = 2, d ∈ [23, 26]) could still be new
  points.
- **Symmetry used for root fixing** is the translation-type automorphisms only (outer automorphisms
  of G are ignored): this costs time, never correctness, and the independent C check uses none.
- **Enumeration.** Classes are merged under the automorphism *generators* GAP returns; a missing
  generator could only produce duplicate classes, never miss one. The search does not decide whether
  codes over different groups (e.g. the [[288,16,16]] codes over five groups of order 144) are
  isomorphic.
- **Coset codes** were implemented and tested (§3) but, on this loaded machine, not searched at scale.

## 9. Reproduce

```text
# groups (GAP 4.15.1 with SmallGroups; ~1 min): one file per order N, 1000 groups
gap -q research/data/code-discovery-2/export_groups.g      # after: ExportOrder(N, dir) for N = 6..150
python3 research/data/code-discovery-2/known_codes.py > threshold.tsv     # T(n, k)
cargo build --release --example group_codes
G=target/release/examples/group_codes
$G search threshold.tsv 4 200000000 groups/*.txt > w6.jsonl 2> w6.log    # all 1000 groups
python3 research/data/code-discovery-2/analyze.py --out summary w6.jsonl w6.log

# the headline code, three ways
$G params groups/144.txt 167 0,9,83 0,51,90                                # Rust: [[288,16,16]], 4 s
python3 research/data/code-discovery-2/code_288_from_presentation.py p288 # presentation only: k = 16
gcc -O2 -o mwlogical research/data/code-discovery-2/mwlogical.c
./mwlogical 15 < p288_Zlogicals.txt; ./mwlogical 15 < p288_Xlogicals.txt  # "none <= 15", ~1 min each
./mwlogical 16 < p288_Zlogicals.txt                                        # finds a weight-16 logical
cargo test --release --test group_codes new_code                           # group-free pins, 7 s
```

The certificate inputs for all five new codes are in `certificates/` (run `mwlogical d-1 < file`).
