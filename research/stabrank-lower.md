# Stabilizer-rank lower bounds: state of the art, exact small-n values, and a plateau lemma

Branch `exp/stabrank-lower`. Code: `research/data/stabrank-lower/stabrank.rs`, a standalone
std-only Rust program (build with `rustc -O stabrank.rs`). Checks: `tests/stabrank_lower.rs`.
Raw outputs are in `research/data/stabrank-lower/*.out`.

**Summary.** I did not prove a super-polynomial lower bound for χ(|T⟩^{⊗n}). That is still a famous
open problem, and nothing here comes close to it. What this note does contain:

1. **New exact small-n values, computer-verified with a completeness proof.**
   * χ(|F⟩^{⊗5}) ≥ 4, where F is the face ("Bravyi–Kitaev T-type") magic state. The previous
     state of knowledge was 3 ≤ χ ≤ 6 (Labib–Russo 2026).
   * Complete lists of all optimal decompositions: H^{⊗2}: 1, H^{⊗3}: 16, F^{⊗2}: 3, F^{⊗3}: 72,
     F^{⊗4}: 9 (a single symmetry orbit). H^{⊗4}: see §3.
   * H^{⊗5}: the 4-term test is in §3.
2. **A structural theorem (the plateau lemma).** Suppose χ(ψ^{⊗n}) = χ(ψ^{⊗(n−m)}). Then every term
   of every optimal decomposition of ψ^{⊗n} is an *m-uniform* stabilizer state, i.e. a pure
   [[n,0,m+1]] code state. Moreover every product-stabilizer restriction of m qubits maps an optimal
   decomposition to an optimal one. Consequences:
   * χ(ψ^{⊗3}) ≥ 3 for **every** non-stabilizer single-qubit ψ.
   * Plateaus of n ↦ χ(ψ^{⊗n}) are short: their length is below the maximum distance of a pure
     self-dual qubit code.
   * A complete "gluing" algorithm builds optimal n-qubit decompositions from (n−1)-qubit ones,
     plus a one-step extension past a plateau. This algorithm is what made the n = 5 computations
     feasible.
3. **An exponential lower bound in a restricted model.** Decompositions whose terms are tensor
   products of stabilizer states over a fixed partition into blocks of size ≤ b (the form of every
   block-product construction in the literature) need (1 − p_b)^{−n/b} terms. For b = 1, 2, 3 this
   is 2^{0.263n}, 2^{0.183n}, 2^{0.159n}. The method is elementary, and the exponent decays like
   1/b, so this is a weak, honest result. It is not a step towards the general problem.

---------------------------------------------------------------------------------------------------

## 1. State of the art (October 2026)

Notation: χ(ψ) is the stabilizer rank, the least r with ψ = Σ_{i≤r} c_i φ_i over stabilizer states
φ_i. χ_δ is the approximate rank (‖ψ − Σ‖ ≤ δ). For qubits there are two Clifford orbits of
single-qubit magic states:
* the **H-type (edge) orbit** contains |H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩ and |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2,
  so χ(T^{⊗n}) = χ(H^{⊗n});
* the **face orbit** F contains cos β|0⟩ + e^{iπ/4} sin β|1⟩, with cos 2β = 1/√3 and Bloch vector
  (1,1,1)/√3.

Labib–Russo call the face orbit "T-type" (Bravyi–Kitaev naming). This note always writes F to avoid
the clash with |T⟩.

### Upper bounds (for context)
| construction | bound on χ(H^{⊗n}) |
|---|---|
| trivial | 2^n |
| Bravyi–Smith–Smolin 2016 (PRX 6, 021043): χ(H^{⊗6}) ≤ 7 | 2^{0.468n} |
| Bravyi–Gosset et al. 2019 (Quantum 3, 181): approximate rank | χ_δ ≤ O(2^{0.228n}/δ²) |
| Qassim–Pashayan–Gosset 2021 (Quantum 5, 606), via contracted cat states | 2^{0.396n} (= 3^{n/4}) |
| Kocia 2020 and others | similar exponents for related families |
| Labib–Russo 2026 (arXiv:2605.28586) | explicit χ(F^{⊗4}) = 3, so χ(F^{⊗n}) ≤ 3^{n/4} |

### Lower bounds
| paper | statement |
|---|---|
| Bravyi–Smith–Smolin 2016 | χ(H^{⊗n}) = Ω(√n) |
| **Peleg–Shpilka–Volk** (Quantum 6, 652 (2022); ITCS 2022) | χ(ψ^{⊗n}) = **Ω(n)** for single-qubit magic states (directional derivatives of quadratic forms over affine subspaces); χ_δ = Ω(√n/log n) for small constant δ (Razborov–Smolensky approximation, correlation with MAJORITY) |
| Labib (Quantum 6, 626 (2022), arXiv:2107.10551) | Ω(n) for qudit magic states, any prime d (higher-order Fourier analysis, non-classical quadratic phases) |
| **Lovitz–Steffan** (Quantum 6, 692 (2022), arXiv:2110.07781) | number-theoretic (refined Moulton theorem) and algebraic-geometric methods: simpler proofs of Ω(n) and approximate-rank bounds up to a log factor; explicit product states with *exponential* χ but O(1) χ_δ; first examples with multiplicative χ; generic stabilizer rank |
| **Mehraban–Tahmasbi** (STOC 2024, arXiv:2305.10277) | χ_δ(T^{⊗n}) = **Ω̃(n²)** for a wide range of δ (Haar-random states have large approximate rank, plus a step-by-step analysis of a teleportation protocol that samples Haar, plus LKS18 Clifford/T trading); also answers Williams' question with super-linear bounds for sums of quadratic exponentials |
| Kalra–Sinha (Quantum 10, 2179 (2026), arXiv:2503.04101) | via Barnes–Wall lattices: χ_δ(H^{⊗n}) = Ω(n/log n) even at exponentially small fidelity; new magic monotone ("Barnes–Wall norm") |
| Labib–Russo (arXiv:2605.28586, May 2026) | Ω(m/log m) for the qutrit Hadamard/Norrell orbits; exhaustive certificates χ(H^{⊗4}) = 4, χ(F^{⊗3}) = 3 (hence χ(F^{⊗4}) = 3) |
| conditional | under ETH / #ETH-type hypotheses, χ(H^{⊗n}) = 2^{Ω(n)} (Huang–Newman–Szegedy 2020, Morimae–Tamaki 2019) |
| analogues | super-polynomial approximate coherent-state rank of |1⟩^{⊗n} (Cottier–Chabaud, arXiv:2604.00766, via Raz's multilinear-formula bound); **exponential** approximate Gaussian rank for every non-Gaussian state, both fermionic and bosonic (arXiv:2610.02172, Oct 2026) |

**Exact small values known before this note.**

| m | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| χ(H^{⊗m}) | 2 | 2 | 3 | 4 (exhaustive, Labib–Russo 2026) | ≤ 6 | ≤ 6 |
| χ(F^{⊗m}) | 2 | 2 | 3 | 3 | ≤ 6 | ≤ 6 |

The best unconditional *exact*-rank lower bound for an explicit tensor-power magic state is still
**linear**. The best approximate-rank bound is quadratic.

### Barriers (why the methods stop at polynomial)
* **Counting and dimension.** There are 2^{n²/2+O(n)} stabilizer states, so dimension and counting
  arguments alone cannot beat ≈ n² for approximate rank. Mehraban–Tahmasbi's n² is essentially this
  ceiling, reached for an explicit state.
* **Algebraic/functional-representation barrier.** A rank-r decomposition writes the amplitude
  function as Σ c_i i^{q_i(x)} 1_{A_i}(x), a linear combination of quadratic-phase functions on
  affine subspaces. Super-polynomial lower bounds for *explicit* functions in this "sums of
  quadratic exponentials" model would imply new lower bounds of the kind Williams (2018) relates to
  ACC/THR-circuit frontiers. Mehraban–Tahmasbi's super-linear bound was new in exactly that model.
  So exponential χ(T^{⊗n}) is at least as hard as a circuit-lower-bound-flavoured statement about a
  very structured product function.
* **Tensor-power structure.** ψ^{⊗n} is a product state: every flattening has rank 1, so rank
  methods from tensor/matrix rigidity give nothing directly. The amplitudes t^{|x|} (t = √2 − 1)
  span only a 2-dimensional Q-space, so number-theoretic independence arguments à la
  Moulton/Lovitz–Steffan do not apply to H (they do give exponential bounds for products of
  *generic* states).
* **Restriction arguments saturate at linear.** PSV-style arguments restrict to subcubes, or take
  derivatives, and count how many terms die. A term dies only if the restriction hits a non-trivial
  local stabilizer, and 1-uniform (locally maximally mixed) stabilizer states are immune to every
  single-qubit restriction. The plateau lemma below makes this precise: whenever a restriction
  argument loses nothing, all the terms are highly uniform stabilizer states. Uniform stabilizer
  states are plentiful (they are code states), so a fundamentally different handle is needed.
* **Contrast with the Gaussian case** (2610.02172). Gaussian states form a continuous variety with
  spectral (Majorana) invariants that degrade multiplicatively under tensor powers. The stabilizer
  set is finite and closed under the Clifford group, which acts transitively. No analogous
  multiplicative invariant is known that is both small on every stabilizer state and large on T.
  Stabilizer extent and fidelity are multiplicative, but they bound only the *approximate* or
  ℓ1-weighted rank.

---------------------------------------------------------------------------------------------------

## 2. The plateau lemma

Throughout, ψ is a single-qubit **non-stabilizer** state and χ_n := χ(ψ^{⊗n}). Restricting qubit j
by a bra ⟨s| (s a single-qubit state) maps an n-qubit vector to an (n−1)-qubit vector.

**Fact 1 (restriction).** If φ is a stabilizer state and s a single-qubit stabilizer state, then
⟨s|_j φ is 0 or a non-zero multiple of an (n−1)-qubit stabilizer state. (Standard: apply a
single-qubit Clifford C with C|s⟩ = |0⟩ and project onto x_j = 0. The support of φ meets the
hyperplane in an affine subspace, on which the quadratic phase restricts to a quadratic phase.)

**Fact 2 (monotonicity).** χ_{n−1} ≤ χ_n. Indeed ⟨s|_n ψ^{⊗n} = ⟨s|ψ⟩ψ^{⊗(n−1)}, and ⟨s|ψ⟩ ≠ 0
because ψ ≠ |s^⊥⟩.

**Lemma 3 (kill criterion).** Let φ be an n-qubit stabilizer state with stabilizer group S and
M ⊆ [n], |M| = m. The following are equivalent:
(i) ⟨s|_M φ ≠ 0 for every product s = ⊗_{j∈M} s_j of single-qubit stabilizer states;
(ii) S_M := {P ∈ S : supp P ⊆ M} = {I}, i.e. the marginal ρ_M = 2^{−m}Σ_{P∈S_M}P is maximally mixed.

*Proof.* ⟨s|ρ_M|s⟩ = ‖⟨s|_Mφ‖². If S_M = {I} this equals 2^{−m} > 0. Otherwise pick I ≠ P ∈ S_M,
P = ±σ_1⊗…⊗σ_m, and let s be a product of eigenvectors of the non-identity σ_j with eigenvalue
product equal to −(sign of P). The elements Q ∈ S_M with ⟨s|Q|s⟩ ≠ 0 are those that are, up to sign,
products of the stabilizers of the s_j. They form a subgroup K ∋ P, on which Q ↦ ⟨s|Q|s⟩ ∈ {±1} is a
character, non-trivial because P ↦ −1. Hence Σ_{Q∈S_M}⟨s|Q|s⟩ = Σ_{Q∈K} = 0, so ⟨s|_Mφ = 0. ∎

**Theorem 4 (plateau lemma).** Suppose χ_n = χ_{n−m} = k and ψ^{⊗n} = Σ_{i=1}^k c_iφ_i. Then:
(a) every φ_i is **m-uniform** (every m-qubit marginal is maximally mixed), i.e. a pure
[[n,0,≥m+1]] stabilizer code state;
(b) for every M with |M| = m and every product stabilizer bra ⟨s|_M, the k restricted states
⟨s|_Mφ_i are non-zero, pairwise non-parallel and linearly independent, and after normalisation they
form an optimal decomposition of ψ^{⊗(n−m)} whose coefficients are uniquely determined.

*Proof.* ⟨s|_Mψ^{⊗n} = (Π_{j∈M}⟨s_j|ψ⟩)·ψ^{⊗(n−m)}, and the scalar is non-zero. So
ψ^{⊗(n−m)} = Σ_i c_i′⟨s|_Mφ_i. By Fact 1 each term is 0 or a multiple of a stabilizer state. If any
term vanished, or two were parallel (merge them), or the terms were linearly dependent (drop one in
the span of the others), ψ^{⊗(n−m)} would have a decomposition with < k = χ_{n−m} terms, which is
impossible. Linear independence gives unique coefficients. Every term being non-zero for every
(M, s) is condition (i) of Lemma 3 for every M, which is (a). ∎

**Corollary 5 (short plateaus).** If χ_n = χ_{n−m}, then an m-uniform n-qubit stabilizer state
exists. So m ≤ d_max(n) − 1, where d_max(n) is the largest distance of a pure [[n,0,d]] qubit
stabilizer code. In particular m ≤ ⌊n/2⌋, and asymptotically m ≤ n/3 + O(1) by Rains' shadow bound.
Concretely:
* There is no 2-uniform stabilizer state on 3 or 4 qubits (checked by enumeration in
  `tests/stabrank_lower.rs`). Hence **χ(ψ^{⊗3}) ≥ 3 for every non-stabilizer single-qubit ψ**: a
  plateau χ_3 = χ_1 = 2 would need one. By the same argument, χ_4 ≥ χ_2 + 1.
* d_max(7) = 3, so χ_7 > χ_4. Likewise χ_n > χ_{n−d_max(n)} for every n.
* Iterating, χ_n ≥ log_{3/2} n − O(1). Asymptotically this is **weaker than PSV's Ω(n)**; I state
  it only because it holds verbatim for every single-qubit magic state, has a two-line proof, and
  gives sharp small-n information.

**Corollary 6 (gluing algorithm).** Take m = 1, restrict the last qubit by ⟨0| and ⟨1|, and
suppose χ_n = χ_{n−1} = k. Then every optimal decomposition of ψ^{⊗n} has the form
φ_i = (|0⟩A_i + ω_i|1⟩B_i)/√2, where {A_i} and {B_i} are both optimal decompositions of ψ^{⊗(n−1)},
ω_i ∈ {±1, ±i}, and c_i is determined by the A-side coefficients a_i. Matching the B side forces
ω_i = (ψ_1/ψ_0)·b_{σ(i)}/a_i for a bijection σ, and so |b_{σ(i)}| = |a_i|·|ψ_0/ψ_1|. The algorithm
`glue()` enumerates pairs of optimal (n−1)-decompositions with matching coefficient magnitudes (a
hash on Σ|a_i|), every bijection σ, and keeps the combinations in which every φ_i is a stabilizer
state. **If χ_{n−1} = k and gluing returns nothing, then χ_n ≥ k + 1.** By the theorem this is a
complete search: it misses nothing.

**Proposition 7 (one step past a plateau).** Let k = χ_{n−1} + 1 and restrict the last qubit by a
stabilizer bra ⟨s|. The restricted k-tuple of a k-term decomposition of ψ^{⊗n} is either
* (I) independent with all terms non-zero (a *minimal* k-term decomposition of ψ^{⊗(n−1)}), or
* degenerate, in exactly one of three ways:
  * (II) one term vanishes and the rest form an optimal decomposition;
  * (IIIa) two terms are parallel, and merging them gives an optimal decomposition;
  * (IIIb) the terms span only χ_{n−1} dimensions; then some χ_{n−1} of them form an optimal
    decomposition D and the remaining term τ lies in span(D).

*Proof.* At most one term can vanish, else < χ_{n−1} terms would span ψ^{⊗(n−1)}. A vanishing term
leaves k − 1 = χ_{n−1} terms, which must be optimal. If none vanish but the tuple is dependent, the
span V has dimension r ≤ k − 1. Any maximal independent subset spans V ∋ ψ^{⊗(n−1)}, so r = k − 1
and that subset is optimal. Two parallel terms are the special case τ ∥ D_j. ∎

So a k-term decomposition is found either by (a) `glue()` applied to the *minimal* k-term
decompositions at n−1, if both the ⟨0|- and ⟨1|-restrictions of the last qubit are of type I; or
(b) by `degenerate_search()`, if some restriction (any qubit j, any of the six bras s) is
degenerate. In case (b) the symmetry group of ψ^{⊗n} (qubit permutations, together with the local
Cliffords and antiunitaries that fix ψ) moves (j, s) to j = n and s in a set of orbit
representatives. Given an optimal D, the coefficients are then fixed up to at most one free
parameter, and each term has ≤ 65 completions (for n = 5). This search is small.

### Validation of the algorithms
Every component was checked against an independent method wherever both are feasible:
* The direct exhaustive search (projective-hash search, below) agrees with naive enumeration of all
  k-subsets for H^{⊗2}, F^{⊗2} with k = 2, 3, 4 (counts 1, 788, 362380 and 3, 1071, 347112), and
  for n = 3 with k = 2, 3.
* With and without the symmetry pruning: identical totals (H^{⊗3}, k = 4: 42261; F^{⊗3}, k = 4:
  156384).
* Gluing reproduces the unique rank-2 decomposition of H^{⊗2}, the 3 of F^{⊗2}, finds none for
  H^{⊗3}/F^{⊗3} with 2 terms (agreeing with the exhaustive search), and produces exactly the 9
  rank-3 decompositions of F^{⊗4} that a direct n = 4 search finds (identical sets).
* `degenerate_search` equals the set of degenerate decompositions in a direct search, for H^{⊗3} and
  F^{⊗3} with 3 terms and several bras (4/16 and 15/72, sets equal).
* Non-plateau `glue` equals the doubly-non-degenerate decompositions of a direct search (H^{⊗3},
  k = 3: 8/16; k = 4: 10128/42261; F^{⊗3}, k = 3: 48/72).

### The exhaustive search (for the base cases)
To find all k-sets {φ_1..φ_k} (independent, with all coefficients non-zero) whose span contains ψ:
order states by symmetry-orbit index. WLOG φ_1 is the representative of the minimal orbit among the
terms, and φ_2 is minimal among the rest and minimal in its Stab(φ_1)-orbit. After choosing
φ_1..φ_{k−2}, put W = span(ψ, φ_1..φ_{k−2}). The remaining two terms must have *parallel*
projections onto W^⊥, because aφ_{k−1} + bφ_k ∈ W. The search hashes the projective class of every
candidate's projection (two fixed random functionals give a point on the Riemann sphere), sorts,
and verifies every near-collision by an exact least-squares check (residual < 1e-8, Gram
determinant > 1e-10, all |c_i| > 1e-8). True solutions collide to ~1e-13; the collision window is
1e-6. Projections of norm² < 1e-9 (the candidate lies in W) can only give dependent sets or
decompositions with < k terms, and are excluded by design.

---------------------------------------------------------------------------------------------------

## 3. Exact small-n table (this work)

(Filled in below from `research/data/stabrank-lower/*.out`.)

RESULTS_TABLE

---------------------------------------------------------------------------------------------------

## 4. An exponential lower bound for block-local decompositions

**Model.** Fix a partition P of [n] into blocks of size ≤ b. A *P-local* decomposition is
ψ^{⊗n} = Σ_i c_i ⊗_{B∈P} τ_{i,B}, with each τ_{i,B} a stabilizer state on block B. Every known
explicit construction of this shape (BSS's χ(H^{⊗6}) ≤ 7 tiled n/6 times, Labib–Russo's
χ(F^{⊗4}) = 3 tiled) is P-local with b = block size. Write R_P for the least number of terms.

**Theorem 8.** For a b-qubit block B, let μ be any probability distribution on b-qubit stabilizer
states σ with ⟨σ|ψ^{⊗b}⟩ ≠ 0, and let p_b(μ) = min_τ Pr_{σ∼μ}[⟨σ|τ⟩ = 0], the minimum taken over
all b-qubit stabilizer states τ. Then R_P(ψ^{⊗n}) ≥ (1 − p_b)^{−1} R_{P∖B}(ψ^{⊗(n−b)}). So if all
blocks have size b, R_P ≥ (1 − p_b)^{−n/b}.

*Proof.* Take an optimal P-local decomposition with k terms. Restrict block B by ⟨σ|, σ ∼ μ. The
target becomes ⟨σ|ψ^{⊗b}⟩ψ^{⊗(n−b)} ≠ 0, term i dies iff ⟨σ|τ_{i,B}⟩ = 0, and the survivors are
P∖B-local. The expected number of deaths is ≥ p_b k, so some σ kills ≥ p_b k terms, and
(1 − p_b)k ≥ R_{P∖B}. ∎

With μ uniform on {σ : ⟨σ|ψ^{⊗b}⟩ ≠ 0} (computed exactly, `stabrank pblock`):

| b | H: p_b | exponent −log₂(1−p_b)/b | F: p_b | exponent |
|---|---|---|---|---|
| 1 | 1/6 | 0.263 | 1/6 | 0.263 |
| 2 | 13/58 | 0.183 | 12/57 | 0.171 |
| 3 | 294/1044 | 0.159 | 291/1026 | 0.160 |

So, for example, **product-stabilizer rank** (b = 1) satisfies (6/5)^n ≤ R ≤ 2^n. Exhaustive
search gives product-stabilizer ranks 2, 3, 4 for n = 1, 2, 3 (both H and F), so the true growth is
invisible at small n. As b grows, p_b tends to a constant (the fraction of stabilizer states
orthogonal to a fixed one), so the exponent decays like 1/b. The theorem says nothing about
decompositions whose terms are entangled across arbitrary cuts, and that freedom is the whole
difficulty. Arguments of this kind are probably folklore; I found no statement of them.

---------------------------------------------------------------------------------------------------

## 5. Other observations

* **Galois symmetry of spans.** Stabilizer vectors (unnormalised) lie in Z[i]^{2^n}. If
  H^{⊗n} = Σc_iφ_i with independent φ_i, the coefficients lie in Q(ζ_8), and the automorphism
  √2 ↦ −√2, i ↦ i fixes every φ_i. So the same span also contains (|0⟩ − (1+√2)|1⟩)^{⊗n} ∝
  |H^⊥⟩^{⊗n}. For F the Galois orbit is larger. This constrains spans but yields only O(1) bounds by
  itself.
* **Structured states.** χ(W_3) = 2 (6 decompositions); χ(W_4) = 2 (15 decompositions).

---------------------------------------------------------------------------------------------------

## 6. Honest assessment

* The asymptotic problem is untouched. The exact-rank frontier remains Ω(n) (PSV/Labib/LS), and
  approximate rank Ω̃(n²) (MT).
* Genuinely new here, as far as I can tell:
  * the exact bound χ(F^{⊗5}) ≥ 4 (and the H^{⊗5} result in §3);
  * the complete lists of optimal decompositions for n ≤ 4 (for example, F^{⊗4} has exactly 9,
    forming one orbit);
  * the plateau lemma, with its corollaries (χ(ψ^{⊗3}) ≥ 3 for all magic ψ; plateau length
    < d_max(n));
  * the gluing and one-past-plateau algorithms, which turn an infeasible n = 5 search (≈ 10^{20}
    subsets) into minutes.
  The block-local bound is elementary.
* Soundness of the negative computations rests on: (1) the completeness proofs above; (2)
  floating-point tolerances that are generous relative to the ~1e-13 errors of true solutions; (3)
  the cross-validations listed in §2. They are not formal proofs in a proof assistant.
