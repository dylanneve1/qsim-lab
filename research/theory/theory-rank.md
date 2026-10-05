# Low stabilizer rank in algorithm circuits: theorems, a branching-rank invariant, and an exact engine

Branch `exp/theory-rank`. Author: qsim-theory-rank agent (round 4, 4 Oct 2026). Base: main 3919576.
Code: `src/engines/stab_rank.rs` (engine, ~1,300 lines), `examples/theory_rank.rs` (`profile`, `verify`,
`grover`), `tests/theory/theory_rank.rs` (9 tests). Data: `research/data/theory-rank/` (`families.jsonl`,
`scaling.jsonl`, `tables.md`, `grover_demo.txt`, `grover64.txt`, `grover128.txt`, driver scripts).

## Summary

The magic atlas found that Grover states look maximally magic to every measure it tracks (d = ν = n)
although they are sums of two stabilizer states. This note explains why and turns it into an engine.

1. **Why nullity misses low rank (Theorem R2).** For stabilizer states φ₁, …, φ_k and coefficients c,
   ν(Σ c_j φ_j) ≤ n − log₂|∩_j Stab φ_j|, with **equality for all c outside a finite union of proper
   subspaces**. For k = 2 and ⟨φ₁|φ₂⟩ ≠ 0 the right-hand side is exactly log₂(1/|⟨φ₁|φ₂⟩|²). So a rank-2
   state has nullity n whenever its two terms have overlap 2^{−n/2}, which is the Grover case
   (⟨w|s⟩ = 2^{−n/2}). The only general relations are **log₂ χ ≤ ν ≤ d** and **χ = 1 ⇔ ν = 0**
   (Theorem R1); low χ says nothing about ν beyond ν ≥ 1. For the branch states of theory-shor T2,
   **χ ≤ |S| · 2^{ν−d}** (one stabilizer term per coset of the group A_ψ of T2a).
2. **An exact per-term branching rule (Theorem R3).** Every non-Clifford gate in the atlas families is a
   *projector gate* U = I + (λ−1)Π (T, Phase, Rz, CPhase, Toffoli), with Π the joint +1 projector of m
   commuting Paulis. On a stabilizer term φ, let m_eff be the number of factors that are still
   random when the factors are applied one at a time. Then **Uφ is a stabilizer state iff m_eff = 0,
   or m_eff = 1 and λ ∈ {±1, ±i}, or m_eff = 2 and λ = −1**; in those cases the term is updated in
   place by a Clifford, and only otherwise does it branch into φ and (λ−1)Πφ. A T gate on a qubit that
   is a basis bit in a term does not branch that term; a Toffoli whose controls are fixed, or that
   acts on a term with ≤ 2 basis branches, does not branch at all.
3. **Branching rank r_k.** The engine keeps Ψ = Σ_j c_j φ_j in CH-form (Bravyi et al. 2019), applies
   R3, and after each gate merges terms that are the same ray (canonical signed stabilizer group as
   key) or whose sum is one stabilizer state (pair merge in the 2^s-dimensional joint eigenspace of
   their common stabilizer group, s ≤ 6). r_k = number of terms after gate k, so **χ(Ψ_k) ≤ r_k**. Cost
   is O(r·n²/64) per Clifford gate and O(r·n³/64) per branching gate.
4. **Where r_k ≪ 2^d** (atlas families, table in §4):
   - **Grover with the real Toffoli-ladder oracle: r = 2 at every iteration boundary, max r = n
     (n search qubits) inside the oracle**, for every n tested (6 … 128) and every iteration (up to
     804). d = 2n − 2 there, and ν reaches 2n − 2 inside the ladder.
   - **Reversible arithmetic on ≤ 2-branch inputs: r = 1 at every gate (Theorem R5)**: the windowed
     Shor oracle for a 62-bit modulus (256 qubits, 27,150 Toffolis) never branches. Cuccaro and Draper
     on classical inputs, Gidney (r ≤ 3 inside its T-gate ANDs), Draper on |+⟩|+⟩, QFT on |+ⁿ⟩ and
     Clifford-point Trotter also stay at 1.
   - **Coined quantum walk on a 2^m-cycle: r = 2 after 2 steps at every m (m = 3 … 32), r ≤ 15 after
     8 steps on 2^32 sites (64 qubits)**, where d = 62 and magic recycling had f_rec > 16.
   - The Shor oracle with x half-superposed (2^{nbits/2+1} branches) stays at r ≤ 17 up to 7 bits and
     fails at 8 bits.
   Everywhere else (QFT on basis or graph input, adders on superposed input, Trotter, QAOA, HEA,
   Heisenberg, QPE, HHL, random Clifford+T) r grows like 2^{#non-Clifford gates on superposed qubits}:
   r = 2^ν exactly for QFT|x⟩, r = 2^d for random Clifford+T, and **r > 2^n** for rotation-heavy
   circuits (e.g. HEA n = 6: r = 11,946 > 64). The terms are then distinct but linearly dependent, so
   r says nothing useful, and the compressed (2^d) or state-vector engines should be used.
5. **Beyond state-vector, exact** (checked against Grover's closed form). Mac = M1 Pro, 1 thread,
   under the bench lock, single run, Mac 1-min load 10–15 from other agents (`mac_timing.txt`). VPS =
   EPYC, 1 thread, `nice 15`, load 6–10. Both are indicative only, not benchmarks.

   | instance | qubits | gates (Toffolis) | max r | Mac | VPS | max error |
   |---|---|---|---|---|---|---|
   | Grover n = 20, **804 iterations (full search, P(w) = 0.99999976)** | 38 | 141,524 (57,888) | 20 | **43 s** | 123 s, 12 MB | 6.3e-13 rel. |
   | Grover n = 64, 8 iterations | 126 | 4,720 (1,984) | 64 | **15.6 s** | 55 s, 5 MB | 9.8e-16 rel. |
   | Grover n = 128, 1 iteration | 254 | 1,286 (504) | 127 | **22.7 s** | 83 s, 13 MB | 1.6e-16 rel. |
   | windowed Shor oracle, 62-bit N, x = 1 | 256 | 85,718 (27,150) | 1 | — | 3.2 s | (r = 1; theory-shor T2c) |
   | coined walk on 2^32 sites, 8 steps | 64 | 15,416 (15,376) | 15 | — | 47 s | (SV-checked at 2^8 sites) |

   ```
   cargo build --release --example theory_rank
   target/release/examples/theory_rank grover 20 804 5 400   # 38 qubits, full Grover search
   target/release/examples/theory_rank grover 64 8 1 400     # 126 qubits
   target/release/examples/theory_rank grover 128 1 1 400    # 254 qubits
   ```

What is known and what is new is discussed in §7. In short: stabilizer rank, CH-form and
sum-over-Cliffords simulation are Bravyi–Gosset 2016 and Bravyi et al. 2019, and Grover's rank 2 is
folklore. Coalescing sums of stabilizer states is in Garcia–Markov's stabilizer frames (Quipu,
2013/2015). The new parts, as far as I could find: R2 (generic nullity of a sum is the codimension of
the common stabilizer group, which explains d = ν = n on Grover), R3 (a complete per-term criterion),
the branching rank as an atlas-wide invariant, and exact gate-level Grover with the real ladder
oracle at 38 qubits through a full search and at 254 qubits for one iteration.

---------------------------------------------------------------------------------------------------

## 1. Definitions

n qubits; Stab ψ is the group of Hermitian Paulis P with Pψ = ψ (signs included); **ν(ψ) = n −
log₂|Stab ψ|** (stabilizer nullity, Beverland–Campbell–Howard–Kliuchnikov 2020); **χ(ψ)** = least k
with ψ = Σ_{j≤k} c_j φ_j, φ_j stabilizer states (Bravyi–Smith–Smolin 2016). d = active dimension of the
rotation frame (atlas §1, ν ≤ d). For a branch state ψ = Σ_{x∈S} α_x|x⟩, D = direction space of aff S,
d_aff = dim D, and A_ψ is the group of theory-shor Theorem 2a (ν = d_aff − dim A_ψ).

A **projector gate** is U = I + (λ−1)Π with |λ| = 1 and Π = Π_{P₁}⋯Π_{P_m}, Π_P = (I+P)/2, P_i
commuting Hermitian Paulis. The families use:

| gate | λ | factors |
|---|---|---|
| T, T†, Phase(θ) | e^{±iπ/4}, e^{iθ} | −Z_q |
| Rz(θ) | e^{iθ} (global e^{−iθ/2}) | −Z_q |
| CPhase(θ) | e^{iθ} | −Z_a, −Z_b |
| Toffoli(a, b; t) | −1 | −Z_a, −Z_b, −X_t |

(Rx, Ry, U, iSWAP, √X are lowered to Clifford + Rz first.)

## 2. Rank versus nullity

**Theorem R1.** (a) χ(ψ) ≤ 2^{ν(ψ)}. (b) χ = 1 ⇔ ν = 0. (c) For a branch state, χ(ψ) ≤ |S|/|A_ψ| =
|S|·2^{ν − d_aff}. (d) ν is not bounded by any function of χ: for every n ≥ 1 there are states with
χ = 2 and ν = n (Theorem R2). So between χ and ν only log₂χ ≤ ν and χ = 1 ⇔ ν = 0 hold in general
(and ν ≤ d from the atlas).

*Proof.* (a) Stab ψ has n − ν independent commuting generators. A Clifford C maps them to
Z₁, …, Z_{n−ν}, so Cψ = |0^{n−ν}⟩ ⊗ ψ′. The ν-qubit state ψ′ is a sum of at most 2^ν basis states, and
C^{−1} maps each term to a stabilizer state. (b) holds by definition. (c) A_ψ ⊆ T(S), so S is a union
of A_ψ-cosets. Fix a coset K = x₀ ⊕ A_ψ and let ψ_K be the restriction of ψ to K. For a ∈ A_ψ the Paulis
P_a = λ_a^{−1} X^a Z^{b_a} of theory-shor T2a stabilize ψ, so they commute with each other, and they
preserve K, so they stabilize ψ_K. For z ∈ A_ψ^⊥, Z^z acts on ψ_K as the constant (−1)^{z·x₀}. These
Paulis commute with each other (z·a = 0) and generate an abelian group of size |A_ψ|·2^{n − dim A_ψ} =
2^n. So ψ_K is a stabilizer state, and ψ = Σ_K ψ_K has |S|/|A_ψ| terms. (d) follows from R2 with
φ₁ = |0ⁿ⟩, φ₂ = |+ⁿ⟩. ∎

(c) is tight for two-branch states (|S| = 2: χ = 2^ν ∈ {1, 2}, which is T2c). It is useless for Grover
(S = everything, d = ν = n gives 2^n), where R2 is the relevant statement.

**Theorem R2 (nullity of a sum of stabilizer states).** Let φ₁, …, φ_k be stabilizer states,
I = ∩_j Stab φ_j (signed) and s = n − log₂|I|. For every c ∈ ℂ^k with ψ = Σ c_j φ_j ≠ 0:

  (a) I ⊆ Stab ψ, so **ν(ψ) ≤ s**;
  (b) **ν(ψ) = s for all c outside a finite union of proper linear subspaces of ℂ^k**;
  (c) for k = 2 and ⟨φ₁|φ₂⟩ ≠ 0: **|⟨φ₁|φ₂⟩|² = 2^{−s}**, so generically ν(αφ₁ + βφ₂) = log₂(1/|⟨φ₁|φ₂⟩|²);
      for ⟨φ₁|φ₂⟩ = 0, s ≥ 1;
  (d) the exceptional ratios really occur and can lower ν to 0 even when s = n: |00⟩ − |++⟩ ∝
      (1, −1, −1, −1)/2 = (−1)^{x₁∨x₂} is a stabilizer state (s = 2, ν = 0).

*Proof.* (a) is immediate. (b) For a Hermitian Pauli P ∉ I, V_P = {c : Σ c_j (P − I)φ_j = 0} is a
linear subspace. It is proper: if it were all of ℂ^k, then c = e_j would give Pφ_j = φ_j for every j,
so P ∈ I. There are finitely many P (2·4^n). Outside ∪_P V_P, Stab ψ = I. (c) Map I to +Z₁, …, +Z_{n−s}
by a Clifford: φ₁ = |0⟩ ⊗ a, φ₂ = |0⟩ ⊗ b, with a and b s-qubit stabilizer states with no common signed
stabilizer. If they shared an unsigned one (±P), then ⟨a|b⟩ = ⟨a|P|b⟩ = −⟨a|b⟩ = 0, which is excluded.
So the unsigned intersection is trivial, and the standard inner-product formula (|⟨a|b⟩| = 2^{−t/2},
t = number of generators of Stab a outside ±Stab b; Aaronson–Gottesman 2004, Garcia–Markov–Cross
2012) gives t = s. If ⟨φ₁|φ₂⟩ = 0, then φ₁ ≠ φ₂ forces s ≥ 1. (d) can be checked by direct computation. ∎

**Corollary (Grover).** Between iterations the Grover state is α|s⟩ + β|w⟩ (ancillas |0⟩), and
|⟨s|w⟩|² = 2^{−n}. So ν = n except at finitely many (α : β), and in particular at every iteration
count k < k_opt for n ≥ 3 (checked numerically: atlas ν = n_search at every iteration end). The exceptions
are where the search hits: for n = 2 one iteration gives |w⟩ (ν = 0). Inside the oracle ladder the state
is Σ_x |x⟩|x₀x₁, x₀x₁x₂, …⟩, which is a sum of j + 2 stabilizer states after j ladder Toffolis (group
the x by the length of their leading run of 1s). So **χ ≤ n inside the oracle while ν and d reach
2n − 2**. All of this is invisible to d, ν, f and f_rec, which see "maximal magic".

*Checks:* `thm_r2_rank_two_nullity` uses 300 random pairs of stabilizer states (n ≤ 4) and 10 ratios
each. The common signed stabilizer group is brute-forced over all 2·4^n Paulis, ν over all 4^n
expectations. The test checks ν ≤ s for every ratio, ν = s for the generic ratio, the overlap law
whenever ⟨φ₁|φ₂⟩ ≠ 0, and that special ratios do lower ν (they do).

## 3. The engine and the branching rank

**Representation.** Ψ = Σ_j c_j φ_j with c_j ∈ ℂ and φ_j = ω_j U_C U_H |s⟩ in CH-form (Bravyi, Browne,
Calpin, Campbell, Gosset, Howard, Quantum 3, 181 (2019), §4.1; ω_j exact as 2^{p/2}e^{iπe/4}). The
Clifford rules are a port of Qiskit Aer's `chstabilizer.hpp` to bitsets of any width (Aer is limited
to 63 qubits). Added operations: (I + i^c P)/√2 for any Hermitian P (projection for c = 0, the Pauli
π/4 rotation for c = 1, through Aer's `UpdateSvector`); the deterministic eigenvalue of P; Pφ; exact
amplitudes; and a **canonical key**: the signed stabilizer generators
U_C U_H (±Z_k or ±X_k) U_H U_C^{−1}, obtained from G^{−1} and F^{−1}, in fully reduced row echelon form
with phase-correct products. Cost O(n³/64).

**Theorem R3 (exact non-branching criterion).** Let φ be a stabilizer state and U = I + (λ−1)Π_{P₁}⋯Π_{P_m}
a projector gate. Apply the factors in order to a copy ψ ← φ. A factor that is deterministic on the
current copy either acts as the identity (eigenvalue +1; drop it) or gives Πφ = 0 (then Uφ = φ). A
factor that is random is *effective*: ψ ← Π_P ψ. Let m_eff be the number of effective factors. Then:

- m_eff = 0: Uφ = λφ, or φ if Πφ = 0;
- Uφ is a stabilizer state **iff** m_eff = 0, or m_eff = 1 and λ ∈ {±1, ±i}, or m_eff = 2 and λ = ±1;
- the Clifford cases are: m_eff = 1, λ = −1: Uφ = −P₁φ; m_eff = 1, λ = ±i: Uφ = e^{±iπ/4}(I ± iP₁)φ/√2;
  m_eff = 2, λ = −1: Uφ = e^{iπ/4} R(P₁) R(P₂) R(P₁P₂) φ with R(Q) = (I + iQ)/√2;
- otherwise Uφ = φ + (λ−1)Πφ, a sum of two stabilizer states (the term **branches**).

*Proof.* Factors that are deterministic with eigenvalue +1 act as the identity on every later copy.
So Πφ = Π_{Q_{m_eff}}⋯Π_{Q₁}φ, where Q_i are the effective factors. **Normal form:** consider the map
Stab φ → F₂^{m_eff}, g ↦ (commutation of g with Q₁, …, Q_{m_eff}). It is onto. Otherwise some
nontrivial product Q = Π_{i∈T} Q_i commutes with all of Stab φ, so Q ∈ ±Stab φ (maximality). Then, for
k = max T, Q_k = ±Q·Π_{T∖k}Q_i lies in ±Stab(Π_{Q_{<k}}φ) (Q commutes with every Q_i), so Q_k was
deterministic: a contradiction. Choose g_i ∈ Stab φ anticommuting exactly with Q_i, and a basis
h_{m+1}, …, h_n of the kernel. The pairs (Q_i, g_i) together with the h's extend to a symplectic basis,
so there is a Clifford K with KQ_iK† = Z_i, Kg_iK† = ±X_i and Kh_jK† = Z_j. Fix the signs with a Pauli
that commutes with every Z_i. Then Kφ = |+^{m_eff}⟩|0⟩ and Π becomes |0^{m_eff}⟩⟨0^{m_eff}|, so

  KUφ ∝ |0…0⟩·λ + Σ_{y≠0}|y⟩ (on the m_eff qubits).

This is a stabilizer state iff the support is affine with flat modulus and the phase is i^{linear}·
(−1)^{quadratic} (Dehaene–De Moor). With |λ| = 1: m_eff = 1 works for λ ∈ {±1, ±i}; m_eff = 2 needs
λ = ±1 (the phase (−1)^{(1−y₁)(1−y₂)} is quadratic, and i^{AND} is not allowed); m_eff ≥ 3 needs λ = 1
(an indicator of a point is cubic or higher). The explicit Clifford forms are identities on the
eigenvalues: f(p₁, p₂) = exp(iπ(1 + p₁ + p₂ + p₁p₂)/4) equals −1 at (+, +) and 1 elsewhere. ∎

*Check* (`thm_r3_branching_criterion_is_exact`): 1,500 random stabilizer states (n = 2…5) and random
commuting factor sets (W Z-strings W†, random signs, m = 1…3, often dependent), with λ ∈ {−1, ±i,
e^{iπ/4}, e^{0.3i}}. The test checks that **"engine branches" ⇔ "ν(Uφ) > 0"** in every unitary case and
that the engine's Uφ equals the dense one to 1e-10. (Cases seen: 676 branch, 325 Clifford,
499 diagonal.)

**Merging.** After a gate where some term branched or was changed individually:

1. *Ray merge.* Terms with equal canonical key are the same ray. Their coefficients are added, using
   the exact amplitude ratio at a support point. A merged term is dropped when its weight falls below
   1e-12 times the largest weight that contributed to it (exact cancellation up to rounding).
2. *Pair merge.* For a pair (i, j) in which at least one term is new or changed, compute the signed
   common stabilizer group I_ij and s = n − dim I_ij. An exact early-exit filter is used first: for
   Lagrangian subspaces, dim(L_i ∩ L_j) = n − rank of the n×n symplectic pairing matrix. If s ≤ 6,
   diagonalise I_ij by a synthesised Clifford V (V g V† = +Z_p). Both terms then live in |0⟩_pivots ⊗
   ℂ^{2^s}. Read the 2^s amplitudes of c_iVφ_i + c_jVφ_j; if the sum is (to 1e-13) one stabilizer state,
   synthesise it (`magic_atlas::stabilizer_synth`) and map it back with V^{−1}. Zero sums cancel both
   terms.

Because Clifford gates and the Diagonal/Clifford cases of R3 apply a unitary to every term, two distinct
rays stay distinct and two unmergeable terms stay unmergeable. So merges are only needed for terms
touched individually. That keeps the cost per gate at O(r) pair tests, not O(r²).

**Definition (branching rank).** r_k = number of terms after gate k under the rules above (initial
state |0ⁿ⟩, r₀ = 1).

**Theorem R4.** (a) χ(Ψ_k) ≤ r_k (every term is a stabilizer state). (b) A gate at which no term
branches has r_k ≤ r_{k−1}. A branching gate has r_k ≤ r_{k−1} + #branched terms ≤ 2r_{k−1}. So r_k ≤
2^{B_k}, where B_k is the number of gates up to k at which some term branched. (c) No bound of the form
r_k ≤ f(ν) or f(d) holds: rays can be distinct but linearly dependent, and r_k > 2^n occurs (HEA n = 6:
11,946 terms). The useful structural bound is **χ ≤ min(r_k, 2^{d_k})**, and a planner should run
whichever of the two engines is smaller.

*Proof.* (a) holds by construction. (b) Each branched term adds at most one term, and merges only
remove terms. (c) is shown by the data in §4. ∎

**Theorem R5 (reversible circuits on ≤ 2 branches never branch).** Let the input be a stabilizer state
with at most two basis branches (e.g. (|0⟩|u⟩ + |1⟩|v⟩)/√2, any relative phase in {±1, ±i}), and let the
circuit consist of X, CNOT and Toffoli. Then r_k = 1 at every gate.

*Proof.* The term φ always has ≤ 2 branches (X, CNOT, Toffoli permute basis states, and R3's Clifford
updates compute U φ exactly). Toffoli factors are −Z_a, −Z_b, −X_t. If the two branches agree on a,
then Z_a is deterministic. Otherwise projecting on it leaves one branch, and Z_b becomes deterministic.
X_t is random on any basis branch. So m_eff ≤ 2, λ = −1, and R3 says Clifford. ∎

This is theory-shor T2c ("stabilizer at every Toffoli boundary") restated as an operational fact,
and it extends to the inside of each Toffoli: the engine never lowers a Toffoli to 7 T, so the
borrowed magic of T2e is never created. It does not cover Gidney's T-gate AND (r = 3 inside, 1 after)
or wider superpositions.

**Proposition R6 (Grover with the Toffoli-ladder oracle; verified, not proved in general).** For the
atlas circuit (n search qubits, n − 2 clean ladder ancillas, marked w, diffusion with the same
ladder), r = 2 at every iteration boundary and max_k r_k ≤ n. Verified for n ∈ {4, 6, 9, 12, 16, 20}
(3 iterations, test `grover_rank_two_at_iteration_boundaries`), n = 20 for all 804 iterations, n = 32,
48, 64 (8 iterations) and 128 (1 iteration). *Mechanism:* the basis-state term (|w⟩ in the oracle, |0⟩
in the diffusion) never branches (R3, m_eff ≤ 1). The uniform term branches once per compute-ladder
Toffoli, giving the j + 2 prefix-pattern terms of the R2 corollary. During uncomputation every
branch is cancelled by an equal ray (ray merges: 118 per iteration at n = 64) or absorbed by a pair
merge (about 3 per iteration, around the middle CZ of each ladder). Without pair merges (`RANK_PM_S=0`),
r reaches 121 at n = 8 and is 62 after 2 iterations. Pair merges are essential.

## 4. The branching rank of the atlas families

`theory_rank profile <spec> 1 1024` (cap 1024 terms; `>1024` = stopped). d is the atlas active
dimension. ν is the ground truth at 20 checkpoints (n ≤ 12). Full table: `data/theory-rank/tables.md`.

| family | instances | max r_k | vs d, ν | verdict |
|---|---|---|---|---|
| **Grover, Toffoli ladder** | n_search = 6 … 128 | **n_search** (6, 8, 16, 24, 32, 47, 63, 127); **2 at every boundary** | d = ν = 2n − 2 | **r ≪ 2^d, χ structure found** |
| **Shor windowed oracle, x = 1** | 4 … 62-bit N (22 … 256 qubits) | **1** | d = n, ν = 0 | R5 |
| Cuccaro / Draper on basis input; Draper on \|+⟩\|+⟩ | up to 64 / 8 / 32 bits | 1 / 2^{bits−2} (transient, 1 at end) / 1 | Draper basis: r = 2^{ν_max} | Draper basis exponential mid-circuit |
| Gidney on basis input | 3 … 64 bits | 3 (inside each T-AND), 1 after | d ≈ 2n/3 | small |
| QFT on \|+ⁿ⟩; Ising at J·dt = h·dt = π/4 | n = 12 | 1 | ν = 0 | trivial |
| **Coined walk (2^m cycle)** | m = 3 … 32 | **2 (2 steps), 4 (4 steps), 15 (8 steps, m = 32, 64 qubits)**; 189 at 16 steps, m = 8 | d = 2m − 2, ν = 3 (m ≤ 6) | **r ≪ 2^d** (but r ≲ #basis branches: the sparse engine also works) |
| Shor oracle, x half-superposed | 4 … 8 bits | 7, 7, 17, 14, >1024 | d = n | small up to 7 bits, then fails |
| QFT on a basis state | n = 6 … 24 | **= 2^{ν} exactly** (16, 64, 256, 1024) | ν = n − 2 | exponential (but f = 1: use the factored engine) |
| Cuccaro, a = \|+⟩ | 3 … 16 bits | 4, 8, 9, 18, 36, 72, >1024 | ν = 2·bits − 1 | ≈ 2^{bits−1}: exponential, far below 2^ν |
| Cuccaro / Gidney / Draper, superposed | 5 / 3 / 5 bits | 49 / 629 / 869 | | exponential |
| random Clifford+T | t = 4 … 16 | 2^t while t ≤ d (16, 256, 1024 = 2^d at t = 12, d = 10) | | no gain (expected) |
| QPE, stabilizer eigenstate | t = 2 … 6 | 5, 14, 41, 122, 365 (≈ 3^t) | d = ν = t | **worse than 2^d** |
| Ising, Heisenberg, QAOA, HEA, QFT-graph, QPE-Trotter, HHL, full Shor | n ≈ 12 | >1024 within the first 10–40 % of gates | d ≈ n | r useless (r > 2^n at n = 6) |

So the families where r_k ≪ 2^d are exactly **Grover (amplitude amplification with a real oracle),
reversible arithmetic on ≤ 2 branches, and short coined walks**. Of these, only Grover was not already
handled by an existing engine of the repo (recycling handles the Shor oracle, sparse/branch engines
the walk and classical arithmetic). Where the circuit applies arbitrary-angle rotations to superposed
qubits, sum-over-Cliffords pays 2× per rotation, as theory predicts, and r is the wrong measure.

## 5. Exactness

`cargo test --release --test theory_rank` (9 tests, 36 s on the loaded VPS, 2 threads):

| test | what |
|---|---|
| `ch_form_random_clifford_exact` | 300 random Clifford circuits, n ≤ 7, 60 gates: amplitudes (with global phase) = independent reference SV (`audit_common::RefSv`) to 1e-10, r = 1 |
| `random_clifford_t_exact`, `random_universal_exact` | 200 + 200 random circuits from the audit's edge-biased generator (T, Rx/Ry/Rz/Phase with edge angles, CPhase, Toffoli, SWAP), n ≤ 7: amplitudes to 1e-9 (observed ≤ 1e-14) |
| `canonical_key_identifies_rays` | 400 pairs: key equality ⇔ ray equality (computed densely) |
| `thm_r3_branching_criterion_is_exact` | Theorem R3 (above) |
| `thm_r2_rank_two_nullity` | Theorem R2 (above) |
| `families_exact_vs_statevector` | 20 atlas families at n ≤ 12 (QFT ×3, Cuccaro ×3, Gidney, Draper ×2, Grover ×2, Ising, Heisenberg, QAOA, HEA, QPE ×2, walk, HHL, Clifford+T) against RefSv: max amplitude error 8.5e-15 (HEA, r = 11,946), with up to 57,202 terms (QPE-Trotter) |
| `grover_rank_two_at_iteration_boundaries` | Proposition R6 for n ≤ 20 |
| `thm_r4_two_branch_permutation_circuits_rank_one` | Theorem R5 on Shor (4, 8 bits), Cuccaro 16 bits and 100 random X/CNOT/Toffoli circuits on 2-branch inputs (SV-checked) |

An early version had a real exactness bug: `stabilizer_synth` accepts states within ~1e-9 of a
stabilizer state, so near-Clifford angles made pair merges approximate (error 1.2e-9). A replay guard
now requires agreement to 1e-13.

The beyond-SV demos are checked against Grover's closed form. Each iteration is −(2|s⟩⟨s|−I)(I−2|w⟩⟨w|),
so after k iterations ⟨w|Ψ⟩ = (−1)^k sin((2k+1)θ) and ⟨x|Ψ⟩ = (−1)^k cos((2k+1)θ)/√(N−1). The check
uses w and 200 random x ≠ w with clean ancillas, and 200 random strings with a dirty ancilla (must
be 0; observed exactly 0). The worst relative error is 6.3e-13, after 141,524 gates at n = 20.

## 6. Cost

Per Clifford gate O(r·n²/64) (CH-form left multiplication touches n columns). Per branching gate:
canonical keys O(r·n³/64), plus pair tests (early-exit pairing rank, then the 2^s reduction for
candidates). At n = 64, Grover costs about 7 s per iteration on the loaded VPS: 23 s in ray-merge keys
and 31 s in pair tests over 8 iterations (152k pair tests). At n = 128 it is 83 s per iteration.
From n = 64 to n = 128 the time per iteration grew 12× (about n^{3.6}, under varying load). The obvious speedups (incremental keys,
hashing a cheap invariant before full keys, parallel term updates) are not done. Memory is
r·3n²/8 bytes (12 MB at 254 qubits).

## 7. Literature: known vs new

**Known.**
- Stabilizer rank, and simulation cost ∝ χ: Bravyi, Smith, Smolin, PRX 6, 021043 (2016); Bravyi &
  Gosset, PRL 116, 250501 (2016) (approximate rank of |T⟩^{⊗t} ≈ 2^{0.23t}); Bravyi, Browne, Calpin,
  Campbell, Gosset, Howard, Quantum 3, 181 (2019) (CH-form, sum-over-Cliffords, stabilizer extent;
  CCZ has rank 2). Qiskit Aer's `extended_stabilizer` implements CH-form for ≤ 63 qubits with
  approximate sum-over-Cliffords sampling. Our CH code is a port of it.
- Lower bounds on χ (Peleg, Shpilka, Volk, Quantum 6, 652 (2022)); nullity (Beverland et al. 2020);
  overlaps of stabilizer states (Aaronson–Gottesman 2004; Garcia, Markov, Cross 2012).
- **Stabilizer frames (Garcia & Markov, "Quipu", ICCD 2013; IEEE TC 2015, arXiv:1712.03554):** a sum
  of stabilizer states sharing a frame, Toffoli and controlled-R handled by cofactoring, and a
  *coalescing* step that merges terms. This is the closest prior art to our engine. They report
  compact simulation of Cuccaro adders on full superpositions, where our pair merge (s ≤ 6) does not
  find a small decomposition (r = 49 at 5 bits). I did not reproduce their benchmarks.
- ZX-calculus stabilizer decompositions with simplification (Kissinger & van de Wetering, "Simulating
  quantum circuits with ZX-calculus reduced stabiliser decompositions", 2022; quizx) reduce T-count
  before decomposing. Not compared here.
- "Grover states have stabilizer rank 2" is folklore (also noted in the atlas). Generic Grover
  simulation by stabilizer methods is usually said to be exponential in the number of non-Clifford
  gates (e.g. Quantum Computing SE 15979), which is true for T-count-based decompositions.

**New here, as far as I found.**
1. Theorem R2: the generic nullity of Σc_jφ_j equals the codimension of the common signed stabilizer
   group (= log₂(1/|⟨φ₁|φ₂⟩|²) for two non-orthogonal terms), and ν ≤ that codimension always. This is
   the exact reason d = ν = n on Grover while χ = 2, and it says when low rank and high nullity
   coexist: small mutual overlaps.
2. Theorem R1(c): χ ≤ |S|·2^{ν−d} for branch states, which ties rank to the affine and branch
   structure of theory-shor T2.
3. Theorem R3: a complete, O(m·n²) per-term criterion for when a projector gate (T, Rz, CPhase,
   Toffoli) needs to branch, with explicit Clifford updates for the non-branching cases. The usual
   sum-over-Cliffords decomposes each gate blindly.
4. The branching rank r_k as an atlas-wide invariant, with the finding that it separates Grover,
   2-branch arithmetic and short walks (r ≪ 2^d) from everything else (r ≥ 2^d, often r > 2^n).
5. Exact gate-level simulation of Grover with the real Toffoli-ladder oracle (no oracle shortcut, no
   T-count blow-up: Toffolis branch at most once per term) at 38 qubits through a complete search
   (804 iterations, 57,888 Toffolis), 126 qubits for 8 iterations and 254 qubits for one iteration.
   A complex128 state vector stops at about 29 qubits on the 7.7 GB VPS and about 30 on the 16 GB Mac.

## 8. Caveats and negative results

- r_k depends on the merge heuristics. Ray merges are canonical; pair merges stop at s ≤ 6 and are
  skipped while r > 256 (`pair_merge_max_r`), so r is an upper bound on χ, not χ itself. For the
  Shor oracle with x half-superposed at 8 bits, r runs away after 90 % of the gates. Raising the r-limit
  to 4096 (`RANK_PM_MAXR=4096`) did not help: the run hit its 20-minute timeout without finishing
  (`shorhalf8.json`), because pair merging at r in the thousands is too slow.
- r can exceed 2^n (distinct but linearly dependent rays). For rotation-heavy circuits use 2^d or SV.
- QPE with a stabilizer eigenstate has d = ν = t but r ≈ 3^t: the engine is worse than the compressed
  state there.
- Grover's success amplitude is exact only to rounding. At n = 20 after 804 iterations the relative
  error on 4.8e-7-sized amplitudes is 6.3e-13. The global sign (−1)^k matches.
- Timings: VPS (4 vCPU EPYC, shared), 1 thread, `nice -n 15`, 1-min load 6–10 from other agents, so
  these are not benchmark claims (LOAD RULE). The Mac numbers for the three Grover demos are single
  runs under the bench lock with the Mac oversubscribed (load 10–15), not min-of-3. Read all times
  as upper bounds.
- The walk and the Shor oracle are also easy for the sparse/branch engines. Only Grover is a family
  where this engine is the first exact simulator in the repo beyond SV size.

## 9. Reproduce

```
cargo build --release --example theory_rank
B=target/release/examples/theory_rank
$B profile 'grover:n=16,it=2' 1 1024          # JSON: r profile, d, (ν if n<=12)
$B verify  'walk:m=8,steps=8' 1               # vs state vector
$B grover 20 804 5 400                        # full Grover search, 38 qubits
cd research/data/theory-rank
./families.sh ../../../$B 1024 > families.jsonl
./scaling.sh  ../../../$B > scaling.jsonl
python3 tables.py                             # tables.md
cargo test --release --test theory_rank
```
