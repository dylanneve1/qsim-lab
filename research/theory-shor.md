# Theorems behind the round-4 Shor observations (topic `theory-shor`, branch `exp/theory-shor`)

Author: qsim-theory-shor agent, 4 Oct 2026, base main = e7e102d.
Checks: `tests/theory_shor.rs` (12 tests, all green: `cargo test --release --test theory_shor`,
about 9 min on the loaded VPS with 2 threads) and `research/data/theory-shor/window_bounds.py`
(→ `window_bounds.out`). Each test fails if its theorem is false. Where the arithmetic can be
done exactly, it is: cyclotomic integers, integer divisibility, exact enumeration of measurement
trees.

**Summary.**

| claim (as stated in research/shor.md, shor-noise.md, magic-atlas.md) | verdict |
|---|---|
| T1 `|S_i| ≤ B_i = min(2^i, r/gcd(r,2^(t−i)))` | **true**, proved (Thm 1a) |
| T1 equality "except on a measure-zero set" | **false as worded.** The exceptional set is explicit (Thm 1b), has probability up to Θ(1/r_odd) per round, and is never empty in general: 131 of 45 267 exhaustively enumerated tree nodes. Corrected: P(round deficient) ≤ 4/r_odd (Thm 1c). |
| T1 `Σ B_i ≈ r_odd(2n − log₂ r) + r` | true, with an exact closed form and error term **−1 − 0.0861·r_odd ≤ E ≤ −1** (Thm 1d) |
| T1 peak support = `max(r_odd, r/2)` | true for `max_i B_i` (ν ≥ 1; it is `r` when r is odd). The real peak equals it except with probability ≤ 4/r_odd (Thm 1d) |
| T2 "stabilizer state at every Toffoli boundary" | **true for any permutation circuit**, not only controlled-U, *provided the two branches have equal weight and relative phase in {±1, ±i}* (Thm 2c). For a generic phase, ν = 1 at every boundary. |
| T2 "magic bounded by log₂(#branches)?" | **false** (ν can reach #branches − 1). The right bound is the **affine dimension of the support**, and there is an exact formula (Thm 2a) |
| T2 recycled register f_rec = 1–2 | explained: the nullity *inside* a lowered Toffoli is ≤ dim aff(support without the target) + 1, so **≤ 2 for two branches** (Thm 2e; observed max 2) |
| T3 "any fault in the last ν₂(r) rounds is harmless" | **true for any fault whatsoever** (any channel, any number of faults, readout errors), exactly: P(ok) = S₀ (Thm 3a). Sharp: a readout flip in round t−ν−1 gives P(ok) ≤ 1 − S₀. |
| T3 "Z faults harmless in rounds < ⌊t − 2 log₂ r⌋" | true *with a lower bound depending on the depth into the window*, for every fault that leaves the ancillas clean (Thm 3b). Not exactly 1, and the edge of the window is soft. |
| T3 "X/Y fatal outside the end window" | true **under a stated hypothesis** (the dirty ancilla dephases the control in every later round): P(ok) ≤ 1/r + r_odd·2^(i+1+ν−t) (Thm 3d). The measured 3.2 % start-window survival is above this bound, so about 3 % of X/Y faults are not dephasing ones; "fatal" is not a theorem for all X/Y faults. |
| T3 "Z faults succeed with probability ≈ 1/2 in the middle" | **not proved.** Reduced to an exact Fourier formula (Thm 3c, checked to 1e-10), which is all the analysis needs; the value ≈ 1/2 is a property of the oracle's bit statistics, not a theorem I can state in general. |

Notation throughout: N odd, a coprime to N, r = ord_N(a), ν = ν₂(r), r_odd = r/2^ν; n = number of bits
of N − 1 (so 2^n ≥ N > r, as in `shor::work_bits`), t = 2n rounds; round i = 0 … t−1 applies
controlled-U^(2^(t−1−i)) with U: x ↦ a·x mod N and records bit y_i of y (least significant
first); y_{<i} = Σ_{j<i} y_j 2^j. The oracle maps |c⟩|x⟩|0…0⟩ to |c⟩|a^c x mod N⟩|0…0⟩ for every
x < N (exhaustively tested in the repo: `windowed_controlled_ua_exhaustive_small`, the r4 audit).

---------------------------------------------------------------------------------------------------

## T1. The support law

### Setting

Round i (Griffiths–Niu semiclassical QFT with one recycled control, `shor::Instance::round`):
H on the control, controlled-U_i with U_i = U^(2^(t−1−i)), Phase(−π·y_{<i}/2^i), H, measure. The
noiseless state between rounds is |0⟩_c ⊗ ψ ⊗ |0…0⟩_anc (ancillas clean, checked by the engine).
**Support** S_i(y) = {x : ⟨x|ψ⟩ ≠ 0}: the work-register basis values with non-zero amplitude
before round i, given the recorded prefix y = y_{<i}. Since the ancillas are clean, |S_i| is also
the number of basis branches of the whole register (times 2 inside the round, one per control
value); a permutation gate cannot change the number of branches, so this is the number of branches
at every gate of round i.

**Lemma 1 (intermediate state).** Let g_i = a^(2^(t−i)) mod N. Then

    ψ̃_i(y) = 2^(−i) Σ_{m=0}^{2^i − 1} e^{−2πi·m·y/2^i} |g_i^m⟩,   P(y_{<i} = y) = ‖ψ̃_i(y)‖²,

and ψ_i(y) = ψ̃_i(y)/‖ψ̃_i(y)‖.

*Proof.* Induction on i. One round maps |0⟩ψ to ½|0⟩(ψ + e^{iφ}U_iψ) + ½|1⟩(ψ − e^{iφ}U_iψ), so the
unnormalised post-measurement state for outcome b is ½(ψ̃ + e^{i(φ+πb)}U_iψ̃). Writing
m' = 2m + c (c = the new control bit, U_i^c g_i^m = g_{i+1}^{2m+c} since g_i = g_{i+1}²), the phase of
the c = 1 term is φ + πb = −π y/2^i + πb ≡ −2π(y + 2^i b)/2^(i+1) (mod 2π), i.e. the phase
−2π m'·y'/2^(i+1) with y' = y + 2^i b, using −2π·2m·y'/2^(i+1) = −2πmy/2^i − 2πmb. ∎

### Theorem 1

Let R = R_i = ord(g_i) = r/gcd(r, 2^(t−i)), B_i = min(2^i, R).

**(a) Bound.** |S_i(y)| ≤ B_i for every prefix.

**(b) Exact support.** If 2^i ≤ R, |S_i(y)| = 2^i for every prefix. If 2^i > R write 2^i = qR + s with
0 ≤ s < R, and let ζ = e^{−2πi·R·y/2^i}. Then

    |S_i(y)| = R − (R − s)·[ζ ≠ 1 and 2^i | s·y] − s·[ζ ≠ 1 and 2^i | (R − s)·y],

where [ζ ≠ 1] ⇔ 2^i ∤ R·y. At most one bracket is 1, so |S_i| ∈ {R, R − s, s}; |S_i| = 0 only for
prefixes of probability 0.

**(c) Exceptions are rare but not measure zero.** For every round, P(|S_i| < B_i) ≤ 4/r_odd.

**(d) Sums and peak.** With L = ⌈log₂ r_odd⌉ (L = 0 if r_odd = 1),

    Σ_{i=0}^{t−1} B_i = r_odd·(2n − ν − L − 1) + 2^L − 1 + r
                      = r_odd·(2n − log₂ r) + r + E,   −1 − 0.0861·r_odd ≤ E ≤ −1,

(0.0861 = max_{δ∈[0,1]} (1 + δ − 2^δ)), and max_i B_i = max(r_odd, r/2) if ν ≥ 1, = r if ν = 0. The
peak of the real support, max_i |S_i|, is ≤ that value, with equality except with probability ≤ 4/r_odd.

**(e) Work.** The engine evaluates every gate of round i on 2|S_i| branches, so the counter is
exactly W = Σ_i 2·|S_i|·G_i (G_i = gates of round i); W ≤ Σ 2B_iG_i with equality iff no round is
deficient.

*Proof of (a), (b).* By Lemma 1 the coefficient of g^k (0 ≤ k < min(R, 2^i)) collects the m ≡ k (mod R)
with m < 2^i; there are J_k = ⌈(2^i − k)/R⌉ of them (J_k = q + 1 for k < s, q for k ≥ s), and the
g^k are distinct because R = ord(g). With ω = e^{−2πi/2^i}:

    c_k = 2^(−i) ω^{k y} Σ_{j<J_k} ζ^j.

So |S_i| ≤ min(R, 2^i) = B_i, which is (a). If 2^i ≤ R every J_k = 1 and nothing cancels. Otherwise
c_k = 0 ⇔ ζ ≠ 1 and ζ^{J_k} = 1. Now ζ^q = e^{−2πi(2^i − s)y/2^i} = e^{2πi·sy/2^i}, which is 1 iff
2^i | sy, and ζ^{q+1} = e^{−2πi(2^i + R − s)y/2^i}, which is 1 iff 2^i | (R − s)y. If both held, then
2^i | Ry, i.e. ζ = 1. ∎

*Proof of (c).* Write R = 2^e·r_odd (e = max(0, ν − (t − i)); R_i = r_odd·2^e). A deficiency needs
2^i > R, ζ ≠ 1 and ord(ζ) | J for some J ∈ {q, q+1}, so ord(ζ) ≤ q + 1 ≤ 2^(i+1)/R. Since
ord(ζ) = 2^(i − e − ν₂(y)), this forces 2^{ν₂(y)} ≥ r_odd/2, i.e. the first v = ⌈log₂(r_odd/2)⌉ recorded
bits are all 0 (v ≤ i because 2^i > R ≥ r_odd). By Lemma 1 that event has probability
P_v = 2^(−2v) Σ_k J_k(v)², with J_k(v) the class sizes at round v. If 2^v ≤ R_v then P_v = 2^(−v) ≤ 2/r_odd.
Otherwise P_v ≤ 2^(−2v)·R_v·(2^v/R_v + 1)² ≤ 4/R_v ≤ 4/r_odd. ∎

*Proof of (d).* For i ≤ t − ν, gcd(r, 2^(t−i)) = 2^ν and R_i = r_odd; for i > t − ν, R_i = r_odd·2^(i−t+ν).
Since 2^(t−ν) ≥ 2^(2n)/r > 2^n ≥ N > r ≥ r_odd, we get L ≤ t − ν, B_i = 2^i for i < L, B_i = r_odd for
L ≤ i ≤ t − ν, and B_i = r_odd·2^(i−t+ν) (because 2^i ≥ R_i) for i > t − ν. Summing:
(2^L − 1) + r_odd(t − ν − L + 1) + r_odd(2 + … + 2^(ν−1)) = r_odd(t − ν − L − 1) + 2^L − 1 + r.
With 2^L = r_odd·2^δ, δ = L − log₂ r_odd ∈ [0, 1): E = r_odd(2^δ − 1 − δ) − 1, and 2^δ − 1 − δ ∈ [−0.0861, 0]
on [0, 1]. Maximum: the last term B_{t−1} = r/2 if ν ≥ 2, = r_odd = r/2 if ν = 1, and B_i ≤ r_odd for
i ≤ t − ν. If ν = 0 the maximum is r_odd = r. The peak statement follows from (a) and (c), applied to
one round where B_i attains the maximum. ∎

**Checks** (`tests/theory_shor.rs`):

* `t1_support_closed_form_equals_exact_cyclotomic`: the closed form of (b) equals the support computed
  **exactly in Z[ζ_{2^i}]** (coefficients reduced with ζ^(2^(i−1)) = −1, then a zero test) for **all**
  prefixes y < 2^i, i ≤ 10, R ≤ 70 (146 290 cases, 1 721 deficient), plus 3 000 random cases up to
  i = 14, R < 400.
* `t1_support_law_on_gate_level_tree`: the **real gate-level engine** (bit-sliced, windowed w = 4 oracle,
  every gate evaluated) on whole measurement trees (exhaustive for t ≤ 12; 200 random paths for
  N = 65 … 143): at every node the stored state equals Lemma 1 (amplitudes to 1e-9, prefix probabilities
  to 1e-12), and the support equals the closed form of (b). 20 (N, a) pairs, 45 267 nodes, 131 with
  support < B_i. Byproduct: in all 131 the f64 engine cancelled **exactly** (no rounding residue was
  stored), so the engine's support counts are the exact ones.
* `t1_exception_probability_bound`: exact P(deficient) over all prefixes for every r < 2^n, n = 3…7,
  rounds i < 13: maximum r_odd·P = 0.977 (bound 4). The constant 4 is not tight; ≈ 1 is what is observed.
* `t1_sum_and_peak_closed_forms`: the exact closed form, the error interval of E and the peak formula, for
  **every** r < 2^n, n = 2…14.
* `t1_work_counter_identity`: `gate_branch_ops` = Σ 2·|S_i|_closed-form·G_i on seeded trajectories
  (N = 143, 221, 899, 4087).

**Consistency with the measured numbers.** shor.md reports |S_i| = B_i in 5 076/5 088 sampled rounds
(12 exceptions, e.g. N = 143, r = 20, i = 3, y = 4: R = 5, 2^3 = 8 = 1·5 + 3, ζ = e^{−2πi·20/8} = −1 ≠ 1,
8 | (5 − 3)·4, so the s = 3 classes k < 3 vanish and |S| = R − s = 2, as observed). Theorem 1(c) allows
up to 4·t/r_odd expected exceptions per run, which is 0 for all practical sizes (r_odd ≥ 10³) and of
order 1 for the 8-bit instances that produced the 12 exceptions.

---------------------------------------------------------------------------------------------------

## T2. Borrowed magic

### Setting

A **branch state** is ψ = Σ_{x∈S} α_x|x⟩ on n qubits, S = supp ψ. Write T(S) = {a : S ⊕ a = S} (the
translation group of S), D = span{x ⊕ y : x, y ∈ S} (direction space of the affine hull), d = dim D =
dim aff S. **Stabilizer nullity** (Beverland, Campbell, Howard, Kliuchnikov 2020): ν(ψ) = n − log₂|Stab ψ|,
Stab ψ = {Hermitian Pauli P : Pψ = ψ}; ν = 0 iff ψ is a stabilizer state.

### Theorem 2

**(a) Exact nullity of a branch state.**

    ν(ψ) = d − dim A_ψ,   A_ψ = {a ∈ T(S) : ∃ b ∈ F₂ⁿ, λ ∈ ℂ with α_{x⊕a} = λ(−1)^{b·x} α_x for all x ∈ S},

a subgroup of T(S). Consequently **d − log₂|T(S)| ≤ ν(ψ) ≤ d ≤ min(n, |S| − 1)**, and ν = 0 iff S is
an affine subspace and A_ψ = D (flat modulus with the Dehaene–De Moor phase structure).

**(b) The "log₂(#branches)" bound is false.** ψ = (|000⟩ + |001⟩ + |010⟩ + |100⟩)/2 has four branches and
ν = 3 (d = 3, T(S) = {0}). In general ν can be as large as min(n, |S| − 1). The affine dimension of the
support is the correct upper bound; the lower bound d − log₂|T(S)| ≥ d − log₂|S| shows it is tight up to
log₂|S|.

**(c) Permutation circuits on two branches.** Let π be any X/CNOT/Toffoli (indeed any reversible
classical) circuit and ψ = α|u⟩ + β|v⟩, u ≠ v. At every gate boundary the state is α|π_k u⟩ + β|π_k v⟩
and

    ν = 0  if |α| = |β| and β/α ∈ {±1, ±i},      ν = 1  otherwise.

In particular (|0⟩ + |1⟩)/√2 ⊗ |x⟩ ⊗ |0…0⟩ (control of a controlled-U, x a basis state) is a stabilizer
state at every Toffoli boundary of **every** reversible circuit, not only the controlled modular
multiplier.

**(d) Converse: nothing else survives all circuits.** Let ψ be a branch state with at least one ancilla
in |0⟩ (so |S| ≤ 2^(n−1)) and n ≥ 4. Then ν = 0 at every gate boundary of every X/CNOT/Toffoli circuit
iff |S| = 1, or |S| = 2 with equal weights and relative phase in {±1, ±i}.

**(e) Magic inside a lowered Toffoli.** Lower every Toffoli(a, b; c) to the standard 7-T network
(Nielsen–Chuang Fig. 4.9: 2 H, 6 CNOT, 7 T/T†). At any gate inside the network, with S the support at the
preceding Toffoli boundary and S_¬c = {x with bit c deleted : x ∈ S},

    ν ≤ dim aff(S_¬c) + 1   and   ν ≤ ν(boundary) + 6.

For two-branch inputs this gives **ν ≤ 2 at every point of the circuit**.

*Proof of (a).* X^aZ^b|x⟩ = (−1)^{b·x}|x ⊕ a⟩, so ψ is an eigenvector of X^aZ^b (with some eigenvalue)
iff S ⊕ a = S and (−1)^{b·x}α_x = μ·α_{x⊕a} on S, i.e. a ∈ A_ψ with λ = μ^(−1). For each such pair
exactly one sign makes the Hermitian Pauli ±i^{a·b}X^aZ^b stabilize ψ with eigenvalue +1. For fixed
a ∈ A_ψ, two admissible b, b' satisfy (−1)^{(b⊕b')·x} = const on S, i.e. b ⊕ b' ∈ D^⊥; so the admissible
b form a coset of D^⊥ (size 2^(n−d)). Hence |Stab ψ| = 2^(n−d)·|A_ψ| and ν = d − log₂|A_ψ|. A_ψ is a
group (compose the relations) inside T(S), and T(S) ⊆ D. Since S is a union of T(S)-cosets,
|T(S)| ≤ |S|, and |S| ≥ … ≥ d + 1 points span a d-dimensional affine hull, so d ≤ |S| − 1. If ν = 0 then
|A_ψ| = 2^d, so S ⊇ a coset of D of size 2^d while S ⊆ aff S of the same size: S is affine. ∎

*Proof of (b).* S = {0, e₁, e₂, e₃} spans F₂³ (d = 3) and S ⊕ a ≠ S for every a ≠ 0, so A_ψ = {0}. ∎

*Proof of (c).* π_k is a bijection, so the state is α|u'⟩ + β|v'⟩ with u' ≠ v'; d = 1, T(S) = {0, u'⊕v'}.
a = u' ⊕ v' ∈ A_ψ iff β = λ(−1)^{b·u'}α and α = λ(−1)^{b·v'}β for some b, λ, i.e. λ² = ±1 (choosing
b·(u'⊕v') freely) and β/α = ±λ, i.e. |β| = |α| and β/α ∈ {±1, ±i}. Then ν = 1 − dim A_ψ. ∎

*Proof of (d).* "If" is (c). "Only if": a two-branch state violating the phase/weight condition already
has ν = 1. Let k = |S| ≥ 3. There is a non-affine k-subset S' ⊂ F₂ⁿ: if k is not a power of 2 any
k-subset works; if k = 2^m (2 ≤ m ≤ n − 1), take a subspace V of dimension m, remove a v ∈ V∖{0} and
add a w ∉ V (an affine set of size 2^m containing 0 is a subspace; it would contain V∖{v}, which spans
V because 2^m − 1 > 2^(m−1), hence equal V, contradicting w). For n ≥ 4, X/CNOT/Toffoli circuits on n
wires generate the alternating group on F₂ⁿ (Shende, Prasad, Markov, Hayes 2003), which is
(2ⁿ − 2)-transitive, and k ≤ 2^(n−1) ≤ 2ⁿ − 2; so some NCT circuit maps S onto S'. At its last boundary
the support is not affine, so ν > 0 by (a). ∎

*Proof of (e).* In the network, the controls a, b only ever see CNOTs among themselves (b ↦ a ⊕ b and
back) and diagonal gates; the only non-diagonal gates on the target are H and CNOTs from a, b. Hence a
basis input |x⟩ is mapped, at any interior point, to e^{iθ_x}|w(x)⟩ ⊗ |τ_x⟩_c, where w is an affine
bijection of the non-target bits (identity or b ↦ a ⊕ b) and |τ_x⟩ a single-qubit state of the target.
So the state lies in span{|w⟩ : w ∈ w(S_¬c)} ⊗ ℂ²_c, whose non-target support has affine dimension
d' = dim aff S_¬c; every Z^z with z supported off c and orthogonal to its direction space stabilizes the
state up to sign, giving 2^(n−1−d') stabilizers, so ν ≤ d' + 1. For the second bound: the stabilizers of
the boundary state that act as the identity on {a, b, c} commute with the network, and they form a
subgroup of index ≤ 4³ = 2⁶ (the restriction map to 3-qubit Paulis mod phase). ∎

**Relation to the atlas's magic recycling (f_rec = 1–2).** magic-atlas.md asserts Σlive ≥ ν for its
recycled register. Theorem 2(c) says the windowed Shor oracle with control |+⟩ and x a basis state has
ν = 0 at every Toffoli boundary, so an exact recycler can return to an empty register after every
Toffoli; Theorem 2(e) says the magic in flight never exceeds 2 qubits. The observed f_rec ∈ {1, 2} is
therefore optimal up to the engine's choice of when it tests: nothing smaller than the in-flight nullity
is possible, and the in-flight nullity is at most 2. With x half-superposed (2^(h+1) branches) the
boundary nullity is d − dim A_ψ with d up to min(n, 2^(h+1) − 1); the atlas's "ν = 3, Σlive = 13" instance
is consistent with (a) (ν ≤ d), but I did not recompute that instance.

**Checks:**

* `t2_nullity_formula_for_branch_states`: 1 500 random branch states (n = 3…6, random or affine supports,
  flat / ±1 / powers of i / random complex amplitudes): ν by **brute force over all 4ⁿ Paulis** (one FWHT
  per X-part) equals d − dim A_ψ (computed from the support and GF(2) solvability, no Pauli enumeration);
  both bounds hold; 746 states have ν > log₂|S| (the refuted bound); the four-branch example has ν = 3.
* `t2_two_branch_inputs_have_zero_magic_at_toffoli_boundaries`: the 7-T network equals a Toffoli on all
  8 basis states; then 60 random NCT circuits on 6–7 qubits with lowered Toffolis and two-branch inputs:
  ν = 0 at all boundaries for weights equal and phase in {±1, ±i}, ν = 1 for generic phase or unequal
  weights (1 200 boundaries); **inside** every network ν ≤ dim aff(S_¬c) + 1, and the observed maximum is 2.
* `t2_only_equal_two_branch_inputs_are_universally_magic_free`: for 60 inputs with 3–8 branches on 5
  qubits (one ancilla), an NCT circuit reaching ν > 0 is found every time.

---------------------------------------------------------------------------------------------------

## T3. Noise windows

### Setting (hypotheses made explicit)

The circuit of shor-noise.md: ancillas and work register start ideal; the phase correction of round i is
computed from the **recorded** bits; faults are single-qubit Paulis after oracle gates, Pauli faults on
the control (Prep, H1, Phase, H2) and readout flips (`noisy::Site`). The success criterion is the
**peak** criterion:

    ok(y)  ⇔  ∃ s : |y/2^t − s/r| < 1/(2r²)  ⇔  dist(y·r/2^t, ℤ) < 1/(2r).

S₀ = P(ok) without faults. A fault is **clean** if, after its round and for every measurement outcome,
the register is |0⟩_c ⊗ φ ⊗ |0…0⟩_anc with φ ∈ span{|x⟩ : x < N} (ancillas clean, work register in the
valid domain). Every Z fault on any qubit is clean (Z changes no bit; the support never leaves the
noiseless one), and so are readout flips, Prep flips (= Z after H), Z faults at H1/Phase and X faults
after H2.

### Theorem 3

**(a) End window, any fault.** If every fault (of any kind: arbitrary channels on any qubits, any
number of them, readout errors, persistent dirty ancillas) acts in rounds i ≥ t − ν, then P(ok) = S₀
exactly. The window is sharp: a readout flip of bit t − ν − 1 gives P(ok) ≤ 1 − S₀.

**(b) Start window, clean faults.** If all faults lie in round i and are clean, then

    P(ok | faults) ≥ L(Δ_i),  Δ_i = 2^(t−i−2)/r²,
    L(Δ) = 1 − 1/(2(⌈Δ⌉ − 3)) if ⌈Δ⌉ ≥ 4;   L(Δ) ≥ 8/π² ≈ 0.81 if Δ ≥ 1.

So clean faults are harmless with probability → 1 deep inside the window i ≪ t − 2 log₂ r; the
guarantee starts at i ≤ t − 2 − 2 log₂ r and says nothing for the last two rounds of the empirical
window ⌊t − 2 log₂ r⌋.

**(c) Exact formula for Z faults (any round).** For Z faults on qubits q_1 … in round i (after gates
g_1 …), let f(c, w) ∈ F₂ be the parity of the values of those qubits on input |c⟩|w⟩|0…0⟩ at those
points of the gate-level computation, and σ(x) = (−1)^{f(x_{t−1−i}, a^(2^(t−i)·⌊x/2^(t−i)⌋))} for
x < 2^t. Then the recorded integer has the exact distribution

    P(y) = Σ_z | 2^(−t) Σ_{x < 2^t : a^x ≡ z} σ(x) e^{−2πi·x·y/2^t} |².

In particular a Z fault on a qubit that is |0⟩ in every branch at that moment (σ ≡ 1) has no effect in
any round.

**(d) Dephasing faults are fatal.** Suppose a fault in round i leaves the register so that, in every
later round j < t − ν and for every outcome history, the control-0 and control-1 outputs of the oracle
are orthogonal (e.g. disjoint supports because a dirty ancilla takes different values in the two
halves). Then bits i+1 … t−ν−1 are i.i.d. fair coins independent of the earlier bits, and for every
recorded prefix

    P(ok | faults) ≤ 1/r + r_odd · 2^(i+1+ν−t).

*Proof of (a).* ok(y) depends only on y mod 2^(t−ν): for y' = y + k·2^(t−ν), y'r/2^t = yr/2^t + k·r_odd.
Bits 0 … t−ν−1 are recorded before any faulty operation acts, and an operation cannot change the joint
distribution of outcomes that precede it (causality: the recorded prefix and its probability are fixed
by the fault-free unitary and measurements up to round t−ν−1). So the distribution of y mod 2^(t−ν) is
the fault-free one. Sharpness: flipping bit t−ν−1 shifts yr/2^t by r_odd/2 ∈ ½ + ℤ, so if y is good,
dist(y'r/2^t, ℤ) ≥ ½ − 1/(2r) ≥ 1/(2r) (r ≥ 2): good maps to bad, so P(ok) ≤ P(fault-free y bad) = 1 − S₀. ∎

*Proof of (b).* The bits y_{<i} have the fault-free distribution (causality); bit i is arbitrary. After
round i the work state φ lies in span{|x⟩ : x < N}, where U^(2^k) acts as a permutation whose cycles
(orbits of x ↦ ax) have lengths dividing r. So φ = Σ_θ γ_θ|u_θ⟩ with eigenvectors U|u_θ⟩ = e^{2πiθ}|u_θ⟩,
θ ∈ (1/r)ℤ, and the remaining rounds (fault-free) see orthogonal eigenvectors, so the outcome
distribution is the mixture Σ|γ_θ|²·P_θ. For one eigenvector, with p = y_{≤i} (i+1 recorded bits) and
t' = t − i − 1, the phase of round j = i+1+j' is 2π·2^(t−1−j)θ − π·y_{<j}/2^j, and writing
y_{<j} = p + 2^(i+1)Y_{<j'} and Y* = (2^tθ − p)/2^(i+1) this equals
2π·2^(t'−1−j')·(Y*/2^(t')) − 2π·Y_{<j'}/2^(j'+1): the remaining rounds are exactly a t'-bit
semiclassical phase estimation of ω = Y*/2^(t'). The recorded y = p + 2^(i+1)Y satisfies
y − 2^tθ = 2^(i+1)(Y − Y*) (mod 2^t), so ok(y) holds (with s = rθ) whenever |Y − Y*| < 2^t/(2r²·2^(i+1))
= Δ_i. The t'-bit estimate satisfies P(|Y − b| > e) ≤ 1/(2(e − 1)) for the nearest-below integer b
(Nielsen–Chuang eq. 5.34), and |Y − b| ≤ e ⇒ |Y − Y*| ≤ e + 1; with e = ⌈Δ⌉ − 2 this gives
L = 1 − 1/(2(⌈Δ⌉ − 3)). If Δ ≥ 1, the two integers nearest Y* both qualify and carry probability
≥ 8/π² (Cleve, Ekert, Macchiavello, Mosca 1998). These bounds hold for every θ, hence for the mixture. ∎

*Proof of (c).* Deferred measurement (Griffiths–Niu): the semiclassical circuit has the outcome
distribution of the textbook circuit (all controlled-U on a 2^t counting register, then the inverse QFT,
then measurement). The Z faults are diagonal gates on (control_i, work, ancillas) inside controlled-U_i.
They commute with every inverse-QFT gate on the other counting qubits (H on other qubits; controlled
phases, diagonal) and are placed between the same oracle calls in both orders, so inserting them in the
textbook circuit is equivalent. There, the counting basis state |x⟩ enters round i with work value
a^(2^(t−i)⌊x/2^(t−i)⌋) and control bit x_{t−1−i}; the faults multiply it by σ(x), the round ends with
clean ancillas, so the pre-QFT state is 2^(−t/2) Σ_x σ(x)|x⟩|a^x⟩ and the formula is the inverse QFT. ∎

*Proof of (d).* For orthogonal halves A₀ ⟂ A₁ (‖A₀‖ = ‖A₁‖ = 1, each is the image of the normalised state
under a unitary), P(1) = ‖A₀ − e^{iφ}A₁‖²/4 = ½ whatever φ, so each such bit is a fair coin given the
whole history. Write y mod 2^(t−ν) = p + 2^(i+1)Y with Y uniform on 2^M values, M = t − ν − i − 1 (the
top ν bits are irrelevant by (a)). ok ⇔ |y mod 2^(t−ν) − s·2^(t−ν)/r_odd| < 2^(t−2ν)/(2r_odd²) for some
s: r_odd intervals of length 2^(t−2ν)/r_odd², each containing ≤ 2^(t−2ν)/(r_odd²·2^(i+1)) + 1 values
≡ p (mod 2^(i+1)). Dividing by 2^M gives 1/r + r_odd·2^(i+1+ν−t). ∎

### Checks and comparison with the measured table

* `t3_end_window_is_exactly_harmless` (exact distributions, whole measurement tree, `trajectory_distribution`):
  N = 15 (r = 4), 21 (r = 6), 35 (r = 4), 51 (r = 8): every control-site fault of every Pauli type,
  12 random gate faults per end-window round and 6 random two-fault patterns: **P(ok) = S₀ to 1e-12** in all
  208 patterns; some single fault in round t−ν−1 changes P(ok); its readout flip gives P(ok) ≤ 1 − S₀.
* `t3_start_window_lower_bound_for_phase_faults`: N = 35 (r = 3, 4), 39 (r = 3), 33 (r = 2), every round
  with Δ_i ≥ 1: Prep and readout flips, Z at H1/Phase, X after H2, 10 random Z gate faults per round:
  P(ok) ≥ L(Δ_i) always (smallest margin 0.0014, so the bound is nearly attained); 77 sampled
  X gate faults in the same rounds fall below the bound, so the cleanness hypothesis is necessary.
* `t3_phase_fault_textbook_formula`: 72 random Z gate faults (N = 15, 21): the formula of (c) equals the
  engine's exact distribution to 1e-10 for every y.
* `t3_dephasing_counting_bound`: the counting bound of (d) for every t ≤ 16, r < 2^(t/2), i and prefix.

`research/data/theory-shor/window_bounds.py` evaluates the bounds on the 15 instances of
shor-noise.md (same window definitions, rounds weighted equally):

| window | measured P(ok\|fault) (shor-noise.md) | theorem | verdict |
|---|---|---|---|
| end, X/Y | 0.997 | = S₀; mean S₀ over the instances 0.997 | exact agreement (Thm 3a) |
| end, Z | 0.997 | = S₀ = 0.997 | exact agreement (Thm 3a) |
| start, Z | 0.976 | ≥ 0.741 (pooled mean of L(Δ_i)) | consistent; the bound is weak at the window's edge |
| start, X/Y | 0.032 | ≤ 0.0009 *if* every fault dephased | the measured value is higher, so ≈ 3 % of X/Y faults are not dephasing (e.g. faults that leave a valid, clean state, which (b) then covers) |
| middle, X/Y | 0.055 | ≤ 0.517 for dephasing faults (bound weak near the end window) | consistent |
| middle, Z | 0.517 | no general theorem; exact formula (c) | open, see below |

**What is not proved: Z ≈ ½ in the middle.** By (c) the answer is the spectral mass of y ↦ |Σ σ(x)…|²
that stays in the good set. Two exact limiting cases: σ ≡ 1 (qubit idle at |0⟩, e.g. AND-chain ancillas of
inactive addresses, P(ok | Z) = 0.81 measured) gives S₀; a Z on the control in a middle round shifts the
interference by a half-period and is almost always fatal (0.24 measured). For a sign that depends only on
the work value w, D = diag(σ) acts on the orbit as a circulant and keeps an eigencomponent with amplitude
μ = 1 − 2ρ (ρ = fraction of the orbit where the qubit is 1), which suggests P(ok) ≈ μ² plus the σ ≡ 1 part.
The ≈ ½ average is the location mix of these classes in the windowed oracle (the per-register table in
shor-noise.md, 0.38–0.81). I could not turn this into a statement with a proof for general (N, a), so it
stays empirical.

---------------------------------------------------------------------------------------------------

## Literature: what was known and what is new

* **Semiclassical QFT and the intermediate state.** Griffiths & Niu, PRL 76, 3228 (1996), quant-ph/9511007;
  one recycled control: Mosca & Ekert (1998), Parker & Plenio, PRL 85, 3049 (2000); Beauregard (2003).
  Lemma 1 is the standard intermediate state implied by these constructions. The output distribution of
  order finding for known r is classical knowledge (Shor 1997; Ekerå, arXiv:1905.09084, App. A, simulates
  it classically for known r). Sparse/branch simulation of permutation oracles with cost ∝ number of
  branches is folklore (e.g. "Leveraging State Sparsity for More Efficient Quantum Simulations", ACM TQC
  2022). Matrix-product-state simulation of Shor whose cost depends on the factors rather than r: Dang,
  Hollenberg et al., arXiv:1712.07311.
  **New here, as far as I found:** the exact support count with its explicit cancellation criterion
  (Thm 1b), the 4/r_odd bound on deficient rounds (1c), and the exact closed form of Σ B_i with the
  two-sided error term (1d), i.e. a proved cost law for exact gate-level simulation of semiclassical
  order finding.
* **Stabilizer states and nullity.** Stabilizer states have affine support with quadratic-form phases
  (Dehaene & De Moor 2003; Van den Nest 2010). Stabilizer nullity: Beverland, Campbell, Howard,
  Kliuchnikov, arXiv:1904.01124. NCT circuits generate the alternating group for n ≥ 4: Shende, Prasad,
  Markov, Hayes, quant-ph/0207001. 7-T Toffoli: Nielsen & Chuang Fig. 4.9; Amy, Maslov, Mosca, Roetteler
  2013. Magic of Shor's full order-finding state: arXiv:2605.05347 ("The true cost of factoring"),
  consistent with T2 (the counting-register superposition is where magic is created: many branches,
  large affine dimension).
  **New here, as far as I found:** the exact nullity formula for branch states ν = d − dim A_ψ (2a),
  the refutation of the log₂(#branches) bound (2b), the classification of inputs that stay stabilizer at
  all Toffoli boundaries of all reversible circuits (2c, 2d), and the in-flight bound ν ≤ 2 explaining the
  atlas's 2-qubit recycled register (2e). The upper bound ν ≤ dim aff(supp) is elementary and may well
  be folklore.
* **Fault analysis of Shor.** Devitt, Fowler & Hollenberg (quant-ph/0408081): error location matters.
  Yang, Liang, Yi & Wang (arXiv:2509.00417): fault-tolerant positions in the Beauregard circuit, Z more
  benign than X/Y. QPE tail bounds: Cleve, Ekert, Macchiavello, Mosca (1998); Nielsen & Chuang §5.2.1.
  **New here, as far as I found:** the exact end-window invariance for arbitrary faults with its
  sharpness (3a), the start-window guarantee for every clean fault with an explicit depth-dependent bound
  (3b), the textbook sign formula for Z faults (3c), and the dephasing bound that makes "X/Y faults are
  fatal" a conditional theorem (3d). These are short consequences of known facts (causality, deferred
  measurement, QPE tails); the contribution is the exact statements and their tests against an exact
  gate-level fault simulator.

## Corrections to earlier notes

1. research/shor.md, "|S_i| = B_i … the 12 exceptions are exact destructive interference": correct, and
   now characterised (Thm 1b). Any wording that calls them "measure zero" is wrong: they occur with
   probability up to Θ(1/r_odd) per round (Thm 1c). They are negligible for the record runs
   (r_odd ≥ 10³), not absent.
2. The closed form `Σ B_i ≈ r_odd(2n − log₂ r) + r` overestimates by between 1 and 1 + 0.0861·r_odd
   (exact formula in Thm 1d). The relative error is ≤ (1 + 0.0861·r_odd)/Σ B_i ≈ 0.0861/(2n − log₂ r)
   for large r_odd: 0.25 at worst (r = 3, n = 2), ≈ 0.3 % for the 31-bit record (2n − log₂ r ≈ 34).
3. magic-atlas.md, "the state is always (|0,u⟩ + |1,v⟩)/√2, a stabilizer state": true, and true for every
   reversible circuit. It needs the relative phase of the branches to be a power of i: a controlled-U
   whose control carries a generic phase (for example a phase correction applied *before* the oracle)
   has ν = 1 at every boundary.
4. shor-noise.md, "X/Y faults are fatal except in the last ν₂(r) rounds": the end-window part is an exact
   theorem for all faults. The "fatal" part holds only for faults that dephase the control (Thm 3d); the
   measured 3.2 % survival in the start window is above the dephasing bound, so it comes from X/Y faults
   that do not dephase. "Z harmless in the start window" holds with the depth-dependent bound of Thm 3b,
   not as an exact statement up to ⌊t − 2 log₂ r⌋.

## Reproduce

```
cargo test --release --test theory_shor -- --nocapture       # 12 tests, prints the counts quoted above
python3 research/data/theory-shor/window_bounds.py            # Thm 3 bounds vs the shor-noise table
```
