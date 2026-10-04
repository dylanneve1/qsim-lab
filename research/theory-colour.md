# Theorems on single-auxiliary syndrome extraction for the triangular colour code

Branch `exp/theory-colour` (from `main` = e7e102d). Code: `tests/theory_colour.rs`; scripts and data: `research/data/theory-colour/`.

This turns the empirical colour-code findings of `research/colour-global.md` and `research/qec-r4.md` Part 2 into statements for every odd distance d. Setting: triangular 6.6.6 colour code with the Kishony–Fowler (K–F, arXiv:2603.28852) layout, Z memory, noisy-CNOT noise. The X-memory statements follow by the code's X/Z symmetry: exchange the two halves of the round.

## Summary

| # | Statement | Status |
|---|---|---|
| **T1 (corner lemma)** | If one corner plaquette's X-check is measured by a single auxiliary with sequential CNOTs, then **d_circ ≤ d − 1**. This holds for any order, depth, collision pattern, number of rounds and X/Z coupling, and for every odd d ≥ 3. One corner is enough. | **Proved** (§3). Tested for d ≤ 41 (combinatorial) and at circuit level for d = 5, 7, 9. |
| **T2 (boundary lemma)** | The same holds for **every** one of the 3d − 6 boundary-touching plaquettes, for every CNOT order. For trapezoids, every weight-2 hook is malign. For a boundary hexagon, every pair avoiding its inward vertex is malign. | **Proved** (§4–5). Tested for d ≤ 41; exact for d = 5, 7, 9. |
| **C3a (corollary)** | d_circ = d needs all 3d − 6 boundary checks measured *without* a bare single auxiliary. This explains and sharpens the certified "≥ 3d − 6 hook-free plaquettes" (one fewer is UNSAT). It also explains why a 3 + 3 cat split fails on boundary hexagons while 2 + 2 is fine on trapezoids. | **Proved** |
| **T3 (residual criterion)** | If every single fault leaves a data residual equivalent (mod stabilizers) to weight ≤ 1, then d_circ = d, for any schedule and any number of rounds. With T2, the boundary checks must satisfy this condition, or some other gadget property, while interior checks may keep hooks. | **Proved** (§6). Tested on real circuit DEMs. |
| **E (interior)** | Exactly 48 of the 720 orders of an interior hexagon have no malign *single* hook; boundary plaquettes have 0. The malign-pair graph is K5 minus one edge, plus one free vertex. | Exact for d = 7, 9 (all weight-d logicals enumerated). Not proved for general d. |
| **C2 (optimum in K–F's space)** | d − 1 at d = 3, 5, 7, 9; **d − 2 at d = 11**. | No general formula proved. The bounds that hold are **≤ d − 1 for all d** (T1) and ≤ d − 2 at d = 11 (DRAT). d = 13 is being run (§7). |
| **Correction (K–F)** | K–F remark that, in their circuit, "no single hook error alone reduces the distance, except at the corners". That is false: every trapezoid hook is malign (T2). At d = 9 the K–F trapezoid diagonal hook plus 7 data errors is a weight-8 logical. The circuit's d_circ is unaffected, since the corners already force d − 1. | Shown at circuit level (test `boundary_lemma_circuit_level_*`). |
| **Correction (ours)** | `colour-global.md` lists K–F's d_circ at d = 13 as 10. Their formula d − ⌊(d+3)/6⌋ gives **11**. The open d = 13 question in K–F's space is therefore D = 12 = d − 1, not D = 11. | Arithmetic. A circuit check of K–F at d = 13 is running (§7). |

## 1. Setting and notation

**Lattice coordinates.** Map the generator's coordinates (x, y) to u = (x − 2y)/4, v = y. Then:
- The sites are the triangle T_L = {(u, v) ∈ ℤ² : u, v ≥ 0, u + v ≤ L}, with L = 3m and d = 2m + 1.
- A site is a **plaquette** iff u − v ≡ 2 (mod 3). Its colour is G, B or R according to v mod 3 = 0, 1, 2.
- All other sites are **data** qubits.
- The six offsets a..f of `color.rs` become the six triangular-lattice neighbours a = (−1, 1), b = (0, 1), c = (1, 0), d = (1, −1), e = (0, −1), f = (−1, 0). The test checks this map against `ColorCode` for every d.
- Every neighbour of a plaquette site is data, so supp(P) = N(P) ∩ T_L.
- A data qubit (u, v) belongs to the plaquettes at offsets {f, b, d} if u − v ≡ 0, and at offsets {c, e, a} if u − v ≡ 1 (intersected with T_L).

**Sides.**
- A = {v = 0} (bottom, the Z logical `y = 0`), B = {u = 0}, C = {u + v = L}.
- Each side carries d data qubits. For example A = {(u, 0) : u ≢ 2 (mod 3)} has 3m + 1 − m = d points.
- Corners: (0, 0), (L, 0), (0, L).

**Rotation.** ρ(u, v) = (v, L − u − v) is a code automorphism:
- it maps T_L to T_L;
- it preserves u − v mod 3, because u + 2v − L ≡ u − v when 3 | L;
- it permutes the six offsets.

So it maps plaquette supports to plaquette supports (test `rotation_is_a_code_automorphism`), and ρ(A) = B, ρ(B) = C. A logical maps to a logical: k = 1, and the stabilizer group is preserved.

**Boundary plaquettes.** These are the plaquettes containing a qubit of degree < 3: there are 3d − 6 = 6m − 3 of them. The bottom side has:
- trapezoids t_j = (3j + 2, 0), j = 0..m − 1. The last one is the bottom-right corner. Each has support {a, b, c, f};
- the corner (0, 1), with support {b, c, d, e};
- hexagons h_j = (3j, 1), j = 1..m − 1.

The other two sides are the ρ- and ρ²-images. Corners are counted on two sides: 3 · 2m − 3 = 3d − 6 (test).

**Circuit class 𝒞.** A Z-memory circuit with any number of rounds, in which some round r has the following properties.
- (i) The X-checks of round r are measured in a block of CNOTs that contains no CNOT of a Z-type check.
- (ii) Every data qubit takes part in at least one CNOT of that block.
- (iii) The X-check of the plaquette p under study uses **one auxiliary**, prepared in |+⟩, coupled by sequential CNOTs aux → data in some order o1, …, ow, and measured in X.

The noise model must include X ⊗ I and I ⊗ X after CNOTs (noisy-CNOT, uniform depolarizing and SI1000 all do).

This covers K–F's design space and the `kf`/`sep`/`free` spaces of `colour-global.md`: any depth, any collisions, X/Z decoupled, other plaquettes measured in any way.

**Lemma 0 (fault accounting).** In 𝒞, suppose a weight-d X-logical L and a *hook residual* H = {o_{k+1}, …, o_w} of p (2 ≤ |H| ≤ w − 2) satisfy H ⊆ L or supp(p) ∖ H ⊆ L, and call that subset Q. Then the circuit has an undetectable logical fault set of size 1 + |L ∖ Q| = 1 + d − |Q|. If |Q| = 2, this is **d − 1**.

*Proof.* Take two kinds of fault:
- the X ⊗ I component after p's k-th CNOT, an X on the auxiliary. It propagates through CNOTs k + 1..w to X_H on data and does not flip the X measurement;
- for each q ∈ L △ Q, the I ⊗ X component after q's last CNOT of the block.

An X on a CNOT target never propagates to the control. So the whole set acts as the data error X_H · X_{L △ Q} at the end of the block, with no other effect:
- if Q = H, this is X_L;
- if Q = supp(p) ∖ H, it is X_L · X_{supp(p)}, i.e. X_L times a stabilizer.

X-type errors flip no X-check. A logical commutes with every later Z-check, and the telescoped Z detectors are therefore unchanged. The final Z data measurement flips Z_A.

The count is 1 + |L △ Q| = 1 + d − |Q|, because Q ⊆ L. ∎

So "malign" below means: some weight-d logical contains H or its complement. The test `boundary_lemma_circuit_level_*` builds exactly these fault sets out of the real circuit's DEM mechanisms, for the K–F, tri-optimal and d = 9 global schedules. It checks that each mechanism exists and is pure X-type, that their XOR is empty with the observable flipped, and that there are d − 1 of them.

## 2. Two families of weight-d logicals

All sets below are sets of data sites. A set S is an X-logical iff every plaquette meets S evenly and |S ∩ A| is odd. Write t_j for the trapezoid (3j + 2, 0).

**Family I_k (k = 0, …, m − 1)**: a red string from the bottom up to the left side, plus the top of the left side. With c = 3k + 1:

I_k = {(c, 0), (c, 1)} ∪ ⋃_{t=0}^{k−1} {x_t = (3(k − t), 3t + 3), y_t = (3(k − t) − 1, 3t + 4)} ∪ V,  where V = {(0, v) : 3k + 3 ≤ v ≤ L, v ≢ 1 (mod 3)}.

**Family II_j (j = 0, …, m − 1)**: the right part of the bottom side, plus a green string from t_j up-left to the left side.

II_j = {(u, 0) : 3j + 3 ≤ u ≤ L, u ≢ 2} ∪ ⋃_{t=0}^{j−1} {p_t = (3j + 1 − 3t, 3t + 1), q_t = (3j − 3t, 3t + 2)} ∪ {(1, 3j), (0, 3j)}.

**Proposition 1.** For every odd d ≥ 3, every I_k, II_j and all their ρ-images are X-logicals of weight d.

*Weights.*
- |I_k| = 2 + 2k + [2(m − k − 1) + 1] = d.
- |II_j| = [2(m − j − 1) + 1] + 2j + 2 = d.

Each count of the form 2(…) + 1 counts a side segment of length 3(…) + 1 minus its plaquette sites.

*Odd overlap with A.*
- I_k ∩ A = {(c, 0)}.
- For j ≥ 1, II_j ∩ A is the bottom segment, of odd size 2(m − j − 1) + 1. II_0 = A.

*Even overlap with every plaquette.* For each point of the set, take its at most 3 plaquettes (offsets {f, b, d} or {c, e, a}, by type) and check that each such plaquette contains exactly two points of the set. Every other plaquette meets the set in 0.

For I_k:
- (c, 0) and (c, 1) share t_k = (c + 1, 0) and the blue hexagon (3k, 1). The third plaquette of (c, 1) is the red r_0 = (c, 2). It also contains x_0 = (3k, 3) if k ≥ 1, or (0, 3) ∈ V if k = 0.
- x_t and y_t (types 0 and 1) share the green (3(k − t) − 1, 3t + 3) and the blue (3(k − t), 3t + 4). Their remaining neighbours lie in rows ≡ 2 (mod 3) or are not in the set.
- The red plaquettes r_t = (3(k − t) + 1, 3t + 2) contain exactly x_t and y_{t−1}, via offsets a and d.
- The last one, r_k = (1, 3k + 2), contains exactly y_{k−1} (or (c, 1) when k = 0) and (0, 3k + 3) ∈ V. Its other neighbour on B, (0, 3k + 2), is not in V.
- Along V: a blue (0, v₀) with v₀ ≡ 1 contains (0, v₀ ± 1), both in V when v₀ ≥ 3k + 4 and neither when v₀ ≤ 3k + 1. A red (1, v₀) with v₀ ≡ 2 and v₀ ≥ 3k + 5 contains exactly (0, v₀) and (0, v₀ + 1).

For II_j:
- Each bottom point shares a trapezoid t_i and a blue (3i, 1) with its partner. The pairs are {3i + 1, 3i + 3} for t_i, i ≥ j + 1, and {3i, 3i + 1} for (3i, 1), i ≥ j + 1.
- t_j contains (3j + 3, 0) and p_0 = (3j + 1, 1). For j = 0 it contains (1, 0) instead of p_0.
- The blue (3j, 1) contains p_0 and q_0.
- p_t and q_t share the blue (3(j − t), 3t + 1) and the red (3(j − t) + 1, 3t + 2).
- The greens g_t = (3j + 2 − 3t, 3t) contain exactly q_{t−1} and p_t.
- g_j = (2, 3j) contains q_{j−1} and (1, 3j).
- The pair (1, 3j), (0, 3j) shares the blue (0, 3j + 1) and the red (1, 3j − 1). The remaining neighbours of all of these lie in rows that contain no other point of the set.

ρ is an automorphism, so the images are logicals too. ∎

Test `families_are_weight_d_logicals` checks every member and every rotation for d = 3..41. `research/data/theory-colour/famtest.py` does the same independently in Python. The minimum-weight logicals are numerous: there are 7, 36, 140 and 464 of weight d at d = 3, 5, 7, 9 (exhaustive enumeration in the test). The families are a small explicit subset.

**Geometry.** These are the Y-shaped string-nets of the colour code. One leg is degenerate along a side, so the junction sits on the boundary. By Viviani's theorem, every such string-net of straight legs has length d. A weight-d logical can therefore pass through a boundary plaquette in many ways, and that is the source of T2.

## 3. Theorem 1 (corner lemma)

**Theorem 1.** Let d ≥ 3 be odd and suppose the X-check of one corner plaquette is measured as in 𝒞 (single auxiliary, sequential CNOTs, any order). Then d_circ ≤ d − 1.

*Proof.* By ρ-symmetry, take the corner plaquette p = (0, 1). Its support is {c₀ = (0, 0), a₀ = (1, 0), b₀ = (0, 2), e₀ = (1, 1)}. Here c₀ is the corner qubit, a₀ ∈ A, b₀ ∈ B, and e₀ is interior.

A weight-4 plaquette has exactly one multi-qubit hook class: H = {o3, o4}, the X on the auxiliary after the 2nd CNOT, with complement {o1, o2}. One of the two pairs contains c₀, so it is {c₀, x} with x ∈ {a₀, b₀, e₀}. Each choice of x lies in a weight-d logical meeting p in exactly that pair:
- {c₀, a₀} ⊆ A;
- {c₀, b₀} ⊆ ρ(A), which is the left side B;
- {c₀, e₀} ⊆ A · S_{t_0}. Here t_0 = (2, 0) has support {(1, 0), (3, 0), (1, 1), (2, 1)}, and A · S_{t_0} = (A ∖ {(1, 0), (3, 0)}) ∪ {(1, 1), (2, 1)}, of weight d.

Lemma 0 with |Q| = 2 gives a logical of d − 1 faults. ∎

**Remarks.**
- This is the "corner lemma" of `colour-global.md` §2, now with a proof for every d. One corner is enough, and a single pairing argument covers every order.
- K–F note that "on the corner plaquettes, all hook errors are malign". T1 makes that precise and turns it into a bound on *every* single-auxiliary circuit, which K–F do not state.
- K–F's design space is therefore capped at d − 1, and that is tight at d = 3, 5, 7, 9 (§7).

## 4. Theorem 2 (boundary lemma)

**Theorem 2.** Let d ≥ 5 be odd and let p be any of the 3d − 6 boundary-touching plaquettes. If p's X-check is measured as in 𝒞, with any CNOT order, then d_circ ≤ d − 1. More precisely, call a pair Q ⊂ supp(p) *malign* if some weight-d logical meets supp(p) exactly in Q.
- (a) If p is a corner or a trapezoid, all three pairings of supp(p) contain a malign pair.
- (b) If p is a boundary hexagon, all 10 pairs inside supp(p) ∖ {ι(p)} are malign. Here ι(p) is p's *inward vertex*: offset a = (−1, 1) for bottom hexagons, and its ρ-images on the other sides.

*Why (a) and (b) suffice.*
- A weight-4 plaquette has the single hook class {o3, o4} ~ {o1, o2}, i.e. one pairing.
- A weight-6 order o1..o6 has the hooks {o5, o6} (after the 4th CNOT) and {o3, o4, o5, o6} ~ {o1, o2} (after the 2nd). The disjoint pairs {o1, o2} and {o5, o6} cannot both contain ι(p), so one of them is malign by (b).
- Lemma 0 then gives a logical of d − 1 faults.

*Proof of (a) and (b).* By ρ, only the bottom side is needed. The corners are Theorem 1. Each entry below is a weight-d logical. The products L · S_P keep weight d because |L ∩ supp P| = |P|/2. Each meets the plaquette in exactly the stated pair.

**Trapezoid t_j, j = 0..m − 2.** Support: a = (3j + 1, 1), b = (3j + 2, 1), c = (3j + 3, 0), f = (3j + 1, 0).

| pairing | malign pair | logical |
|---|---|---|
| {cf ∣ ab} | cf | A |
| {af ∣ bc} | af | I_j (its bottom edge {(3j + 1, 0), (3j + 1, 1)}) |
| {ac ∣ bf} | ac | II_j if j ≥ 1 (string start p_0 = a, segment start c); ρ(I_{m−1}) if j = 0 |

**Hexagon h_j = (3j, 1), j = 1..m − 1.** Support: b = (3j, 2), c = (3j + 1, 1), d = (3j + 1, 0), e = (3j, 0), f = (3j − 1, 1), and the inward vertex a = (3j − 1, 2).

| pair | logical, 1 ≤ j ≤ m − 2 | logical, j = m − 1 |
|---|---|---|
| de | A | A |
| df | A · S_{t_{j−1}} | same |
| ce | A · S_{t_j} | same |
| cf | A · S_{t_{j−1}} · S_{t_j} | same |
| cd | I_j | same |
| bc | II_j | same |
| ef | ρ²(II_{m−j}) | same |
| bd | II_j · S_{t_j} | I_{m−1} · S_{(3m−2, 2)} |
| be | ρ²(I_{m−1−j}) | ρ²(I_0) · S_{(3m−2, 2)} |
| bf | ρ²(I_{m−1−j}) · S_{t_{j−1}} | ρ(II_{m−1}) |

The table entries are read off from explicit coordinates. For example, ρ²(u, v) = (L − u − v, u) maps I_k to:
- the left bottom segment {(u, 0) : u ≤ L − 3k − 3, u ≢ 2};
- the column u = L − 3k − 3 at heights 2, 3, 5, 6, …, 3k − 1, 3k;
- the two points (L − 3k − 1, 3k + 1) and (L − 3k − 2, 3k + 1) on and next to the right side.

For k = m − 1 − j the column is u = 3j, so ρ²(I_{m−1−j}) ∩ h_j = {b, e}. The remaining rows are checked the same way.

Tests:
- `corner_and_boundary_lemmas_hold_with_explicit_certificates` builds every entry and its two rotations for every d = 3..41. For each one it asserts: a logical; weight d; meets the plaquette exactly in the claimed pair; every one of the 3d − 6 boundary plaquettes has **zero** safe orders.
- `malign_hooks_exact_d5_d7` and `malign_hooks_exact_d9` compare this with the exact answer from all weight-d logicals. ∎

**Corollary C3a (necessity of 3d − 6).** If d_circ = d, then none of the 3d − 6 boundary X-checks, and for X memory none of the Z-checks, is measured by a bare sequential single auxiliary. In particular:
- Any scheme that makes at most 3d − 7 plaquettes hook-free has d_circ ≤ d − 1, because some boundary plaquette is then bare. This is the certified `d*_hf{8,14,20}` UNSAT, now for all d. The theorem is stronger: it names *which* plaquettes.
- Three corners hook-free is not enough for d ≥ 5 (certified `d7_D7_hf3corners`).
- A **two-auxiliary cat split** on a boundary hexagon (3 + 3) fails. Each auxiliary's residual after its first CNOT is the last pair of its triple, and the triple that avoids ι(p) yields a malign pair. On trapezoids, 2 + 2 is hook-free. This explains the certified `*_split_all` UNSATs.

**Comparison with K–F's remark.** K–F choose trapezoid hooks "along their diagonals" and state that their circuit "does not admit any single hook error that alone reduces the circuit-level distance, except at the corners". By T2(a) the diagonal pairing {ac ∣ bf} is malign too.

The K–F green trapezoid order is b, f, c, a (steps 1, 2, 4, 5). Its hook is {c, a}, contained in II_j. At d = 9 the trapezoid (5, 0), at (20, 0) in generator coordinates, gives a K–F-circuit logical of 1 hook + 7 data errors = 8 faults. The test assembles it from the K–F circuit's own DEM.

This does not change K–F's d_circ, because the corners force d − 1 anyway. It does mean the boundary deficit is not "fractional-only".

## 5. Interior plaquettes (exact for d = 7, 9; general d open)

Enumerating all weight-d logicals (7, 36, 140, 464 for d = 3..9) gives the exact malign hook classes (`hookD_d*.json`, `mal.py`; the test recomputes them).

- **No weight-3 hook is ever malign.** A weight-3 malign hook would need a weight-d logical meeting a hexagon in 3 qubits.
- An **interior** hexagon's malign pairs form K5 minus one edge on five of its vertices, plus one vertex in no malign pair:

| colour | free vertex | non-malign pair among the other five |
|---|---|---|
| B | a | df |
| G | e | bd |
| R | c | bf |

  Hence exactly **48 of the 720 orders** are safe, i.e. neither {o1, o2} nor {o5, o6} is malign. Equivalently, {o1, o2} and {o5, o6} are {free, x} and the non-malign pair, in some order.
- For **boundary** hexagons the non-malign pair becomes malign (all ten pairs of the five; T2b), so the number of safe orders drops to 0.

This classification matches K–F's description of malign hooks in the bulk ("the set of malign pairs is the same for all plaquettes of a given colour"). Its bulk part is not proved for general d here.

## 6. Theorem 3 (residual criterion): a sufficient construction

**Theorem 3.** Consider a memory circuit: R ≥ 1 rounds of Z- and X-check measurements of any form, final Z data measurement, and detectors = consecutive differences of each Z-check (round 0 against the deterministic |0⟩ preparation, last round against the data). Suppose every single fault f has a data X-residual e_f that is equivalent, modulo X-stabilizers, to an error of weight ≤ 1. (Here e_f is the X part it leaves on the data at the end of the circuit; auxiliary errors are reset or measured away.) Then d_circ = d.

*Proof.* Let F be an undetectable fault set that flips the observable. Sum all Z detectors of one plaquette: the sum telescopes to the syndrome of the final data error E = ∏_{f∈F} e_f. So s(E) = 0.
- If E were a stabilizer, the observable Z_A would not flip.
- So E is a nontrivial logical, and |E|_min ≥ d.
- E ≡ ∏ w_f with |w_f| ≤ 1, so |F| ≥ |E|_min ≥ d.

d single data errors on A meet the bound. ∎

**Check (test `hook_free_residuals_give_full_distance`).** In the real K–F circuit DEMs (d = 3, 5, 7; 2–5 rounds), drop every mechanism whose time-projected (syndrome, observable) is not that of an error of weight ≤ 1. Partial and time-like errors are kept, as are weight-(w − 1) hooks, which are ≡ weight 1. The exact distance of what remains is d. The unfiltered K–F circuit has d − ⌊(d+3)/6⌋ (qec-r4 §2.3).

**Construction and resource count.**
- **Fully hook-free.** Measure every check with a gadget satisfying the hypothesis, for example a verified cat state, or any gadget whose faults have residual ≤ 1 after flag post-processing (the idealised `--hookfree` relaxation). This gives d_circ = d for **every** schedule. It costs a gadget on all (3d² − 3)/8 plaquettes.
- **Boundary-only.** T2 shows the boundary set of 3d − 6 cannot be avoided. The cheapest candidate is boundary gadgets plus bare single auxiliaries in the interior, with interior orders from the 48 safe ones. This is **not** covered by T3, because interior hooks have weight-2 residuals. Whether it reaches d is decided by combinations of interior hooks ("fractional" hooks).
  - Certified constructions: d = 5 and 7 with 6 + 6 layers, d = 9 with 7 + 7 layers (`colour-global.md` §6).
  - d = 11, space-only and free orders: §8.

### 6.1 Why boundary-only fails at d = 11: interior middle hooks

The peer branch `exp/colour-flags` certifies `d11_D11_hfbnd_free` (DRAT VERIFIED). With all 27 boundary plaquettes hook-free and free orders at any depth, D = 11 is UNSAT.

We extracted a deletion-minimal unsat core of its 14,262 cut logicals with `cg_core.py`, giving 151 logicals in `core_d11_hfbnd_free.txt`. We decoded each mechanism into a single data error or an interior hook class (`core_an.py`). The findings:

- **The obstruction is purely spatial.** Every mechanism of every core logical is a clean data error of one layer. Partial (Z-half) errors and time-like errors play no role.
- **96 of the 151 core logicals use exactly one interior hook.** These cut the orders whose single hook is malign. They are what restricts each interior hexagon to the 48 safe orders of §5.
- **The other 55 use 2–4 interior hooks, and every one of them includes a weight-3 "middle" hook** (X on the auxiliary after its 3rd CNOT), often an alternating triple {a, c, e} or {b, d, f}.
  - A weight-3 hook is never malign alone (§5), because no weight-d logical contains three qubits of a plaquette.
  - With a neighbouring plaquette's weight-2 or weight-3 hook it is: the extra qubits cancel, and 2 hooks + 8 data errors = 10 faults.
  - These are interior analogues of K–F's "fractional hook errors".
  - At d ≤ 9 the interior is too small for such pairs to sit on a weight-d logical. At d = 11 it is not.

**Removing middle hooks restores d (space-only check).** In addition to the 27 boundary plaquettes, allow two-auxiliary (cat-split) measurement of interior plaquettes. A 3 + 3 split has only weight-2 residuals, so no middle hook. Then D = 11 is **FOUND** in the space-only model: `cg_sat.py 11 11 1 free --space --hookfree 27 --hookfree-set boundary --split 45`. The solver used splits on 10 of the 18 interior hexagons; this count was not minimised. The 1-round spacetime run is reported under "Status" below.

So, empirically: the boundary needs fully hook-free measurement (T2), and from d = 11 on, interior hexagons need their middle hook removed. Neither interior requirement is proved for general d.

## 7. C2: the optimum in K–F's design space

Let OPT(d) be the largest d_circ over K–F's design space: one auxiliary per plaquette, one collision-free 6-step table for both halves.

| d | K–F formula d − ⌊(d+3)/6⌋ | OPT(d) | evidence |
|---|---|---|---|
| 3 | 2 | **2 = d − 1** | T1 |
| 5 | 4 | **4 = d − 1** | T1 + K–F |
| 7 | 6 | **6 = d − 1** | T1 + K–F |
| 9 | 7 | **8 = d − 1** | T1 + found schedule (certified) |
| 11 | 9 | **9 = d − 2** | DRAT UNSAT for 10 (`d11_D10_kfT6`) |
| 13 | **11** (not 10) | 11 ≤ OPT ≤ 12 | T1; K–F (to be confirmed, §8); D = 12 SAT run in progress |

What is proved for all d: **OPT(d) ≤ d − 1** (T1), with equality for d ≤ 9.

What is *not* proved:
- any general lower bound. K–F's formula is their numerical observation for d ≤ 13, and no proof is known;
- whether OPT(d) ≤ d − 2 for all d ≥ 11.

The d = 11 obstruction is not a single-plaquette effect: every single hook is at worst d − 1 by T1/T2. It must come from interactions between hooks, partial errors and the shared 6-layer table near the boundary.

A natural conjecture is OPT(d) = d − 2 for every d ≥ 11 (a local boundary obstruction that can be repeated along long sides). That would beat K–F's formula from d = 17 on, where the formula gives d − 3. The alternative is OPT(d) = d − ⌊(d+3)/6⌋ for d ≥ 11. At d = 13, OPT = 12 would refute the d − 2 conjecture and OPT = 11 fits both. d = 17 is out of reach of the current CEGAR. **We state it as open.**

## 8. Runs (this branch)

- d = 13, K–F space, `cg_sat.py 13 12 1 kf --warm --sym` (D = d − 1): running on the Mac (`~/qsim-tc/cg`, log `~/qsim-wt-logs/tc-d13-D12.log`). The D = 11 run is kept as a sanity check: it should be FOUND if K–F's 11 holds. K–F d = 13 circuit distance: `color_search distance 13 1 kf` running.
- d = 11, boundary-only hook-free, free orders, space-only (`cg_sat.py 11 11 1 free --space --hookfree 24 --hookfree-set boundary`): running on the VPS.

Results are filled in under "Status" at the end of this file.

## 9. Literature: known vs new

- **Known.**
  - Hook errors and distance halving for uniform single-auxiliary colour-code circuits: Beverland–Kubica–Svore (PRX Quantum 2021); Lee et al. "tri-optimal"; K–F §1.
  - Flag qubits for colour-code hooks: Chamberland–Kubica–Yoder–Flammia (NJP 2020, "triangular colour codes on trivalent graphs with flag qubits"); Baireuther et al. (NJP 2019).
  - Gidney–Jones superdense and middle-out circuits.
  - K–F: corner hooks all malign (stated, not proved); bulk malign-pair classification (stated); fractional boundary hooks; d − ⌊(d+3)/6⌋ (numerics to d = 13).
  - Code distance and string-net picture of the triangular code: Bombin–Martín-Delgado.
- **New here.**
  - T1 as a bound on every single-auxiliary circuit, with proof for all d.
  - T2: every boundary plaquette is individually malign, with explicit weight-d logical certificates (families I and II); this corrects K–F's "only fractional at the boundary" remark.
  - C3a: necessity of hook-free measurement of exactly the boundary set, for all d, explaining the certified 3d − 6 threshold and the cat-split failure.
  - T3's clean residual criterion (standard in spirit, stated and tested here for the colour-code round structure).
  - Exact interior counts (48 safe orders) for d = 7, 9.
  - The d = 13 arithmetic correction.

## 10. Reproduce

```bash
export CARGO_TARGET_DIR=/tmp/tc-target CARGO_INCREMENTAL=0
cargo test --release --test theory_colour -- --include-ignored     # ~25 s
cd research/data/theory-colour
python3 famtest.py          # families I, II and rotations are weight-d logicals, d = 3..41
python3 assign.py 5 41      # boundary-lemma certificates, d = 5..41
python3 hookD.py 5 7 9 && python3 safe.py 5 7 9   # exact malign classes (MILP) and safe-order counts
```
