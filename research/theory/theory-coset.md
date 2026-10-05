# Coset-representation error in the Gidney–Ekerå Shor circuit (topic `theory-coset`, branch `exp/theory-coset`)

Author: qsim-theory-coset agent, 4 Oct 2026, base main = 3919576 (exp/ge-shor merged).
Checks: `tests/theory_coset.rs` (8 tests, all green: `cargo test --release --test theory_coset`,
≈ 15 s on the VPS with 2 threads). Each test fails if its statement is false; one is labelled an
*observation* (exhaustive, not proved). Data and drivers: `examples/theory_coset.rs`,
`research/data/theory-coset/` (README there). All runs on the VPS (nice 15, ≤ 2 threads,
< 200 MB); nothing ran on the Mac.

**Summary.**

| question (brief / research/shor/ge-shor.md §3) | answer |
|---|---|
| (a) why TV ≈ (1.5–7.5)·2^−c, and why the "deviant weight" is ≈ (0.5–1.5)·2^−c per window and barely grows | TV is **linear** in the fraction δ of coset branches that are *misplaced relative to a common reference*, TV ≤ δ_rms + θ_rms + δ̄ + √(δ̄θ̄) ≤ 4δ_rms (Thm A), and measured TV is 0.41–0.62 of that bound in every exhaustive case. δ is the spread of a **zero-mean random walk of the coset index** (step variance ∝ number of lookup-additions per window), so δ·2^c grows like √windows, not like the number of additions. ge-shor's "deviant weight" mostly counts accumulators that are *temporarily* out of range; those are re-absorbed exactly in the next window (Lemma D). |
| (b) why N = 15, 51, 85 give TV = 0 exactly | It is the **order**: if r = ord(a) is a power of two, every work state depends only on E mod r (all paths to a residue apply the *same* multiplier sequence), the output lives on the exact support {k·2^t/r}, and TV = 0 **iff** the r class supports are disjoint (Thm B). Not N mod 2^k, not the base 2, not M ≡ power of the base. Disjointness is generic but **not** automatic: it fails for 8 of 350 (w_e = 2) and 3 of 350 (w_e = 1) power-of-two (N, a) pairs with N ≤ 129, c ≤ 3 (e.g. N = 53, a = 30, w_e = 1, c = 1: TV = 1/32 exactly). Every one of 980 non-power-of-two (N, a, w_e) configurations tested has TV > 0. |
| (c) rigorous bound; additive or not? | Rigorous worst case: **TV ≤ 4·A·2^−c**, A = number of coset additions (Thm C). That is additive, like Gidney's deviation calculus, but *linear* in 2^−c where Gidney's trace-distance theorem gives 2√(A·2^−c). The actual behaviour is **not additive**: the reversible index drift is a random walk (variance 3.2–4.4 per window at 31 bits, linear in windows); δ·2^c ≈ 3·√k + 0.1·k after k windows; the irreversible ("unfaithful") part is only ≈ 0.4·2^−c per window, i.e. ≈ 0.015·2^−c per addition — 70× below the per-addition 2^−c of the additive model. TV ∝ √K at fixed windows (K = additions per multiply). |
| (d) padding for ≤ 1 % at 31 bits | Thm A evaluated by Monte Carlo on the exact 31-bit EH circuit (N = 1 537 596 787, w_e = 2, w_m = 3, 24 windows): **bound·2^c = 34–43** for c = 8…16 (two bases). **c = 12: TV ≤ 0.88–0.92 %** (±0.01 %, 2 000 × 2 000 samples; expected ≈ 0.4–0.6 %); c = 13: TV ≤ 0.46–0.48 %. **Recommend c = 12** (c = 13 if the bound itself must hold with statistical margin). Toffolis: 54 501 (c = 12) / 55 697 (c = 13), + coset preparation ≈ 2c(n+c) = 1 032 / 1 144 → **55.5 k / 56.8 k vs 61 087 for exact EH: still −9 % / −7 %**. Coset loses at c ≥ 15. |

Notation: N odd, n = bit length of N − 1 (so N ≤ 2^n), c padding qubits, M = 2^(n+c); a base of
order r; t exponent bits processed in W windows of w_e bits, highest powers first (as
`shor_ge::run`/`distribution`; for EH the m-bit register with base y^−1 first, then the 2m-bit
register with base g). In window k the multiplier is h = g_k^e (digit e). K = ⌈(n+c)/w_m⌉ lookup
additions per multiply–add; A = 2KW coset additions per run (forward and backward multiply–add).
E ∈ [0, 2^t) is the exponent, u(E) = a^E mod N its **class** (for EH: g^a y^−b).

---------------------------------------------------------------------------------------------------

## 0. The circuit as a permutation (and the model used for every check)

`src/shor/ge.rs` with `GeOpts::coset = c`: registers x, b of n + c qubits start in
Σ_{j,j'<2^c} |1 + jN⟩|j'N⟩ / 2^c. A window with multiplier h is

    x' = (b + A_h(x)) mod M,        b' = (x − A_{h^−1}(x')) mod M,
    A_h(x) = Σ_k T_k[chunk_k(x)],   T_k[v] = h·v·2^{k w_m} mod N ∈ [0, N)

(multiply–add, swap, inverse multiply–add with h^−1; lookups and their measurement-based
uncomputation leave no phase). Each window is therefore a **permutation Π_h of Z_M²** and the work
state for exponent E is the flat state Φ_E = 2^−c Σ_{s∈S_E} |s⟩, S_E = Π_E(S_0), |S_E| = 4^c.
Phase estimation (deferred measurement of the semiclassical windowed QFT) only sees the Gram
matrix K(E,E') = ⟨Φ_{E'}|Φ_E⟩ = |S_E ∩ S_{E'}|/4^c:

    p(y) = 4^−t Σ_{E,E'} ω^{y(E−E')} K(E,E'),     ω = e^{2πi/2^t}.

`tests/theory_coset_model` implements exactly this (≈ 300 lines, independent of the gate-level
engine). **Check** `model_matches_gate_level_engine`: the sorted output laws and the TV agree with
`shor_ge::distribution` (gate-level, MBU, every branch) to 1e−9 for five (N, a, w_e, c);
`model_reproduces_logged_tv` reproduces `research/data/ge-shor/coset_exact.log`.

Exact arithmetic (c = 0) has K_ex(E,E') = [u(E) = u(E')].

## A. TV is linear in the misplaced fraction (Gram lemma)

**Theorem A.** Let {G_u} be *any* family of pairwise disjoint sets of 4^c basis states, one per class
u, and χ_u the flat state on G_u. For each E let

    δ_E = 1 − |S_E ∩ G_{u(E)}| / 4^c          (misplaced fraction),
    θ_E = |S_E ∩ ∪_{u ≠ u(E)} G_u| / 4^c ≤ δ_E   (fraction sitting in another class's reference).

Then, with averages over E uniform,

    TV(p_coset, p_exact) ≤ ½‖ρ_c − ρ_ex‖₁ ≤ δ_rms + θ_rms + δ̄ + √(δ̄ θ̄) ≤ 4 δ_rms.

*Proof.* ρ = 2^−t K is the reduced state of the exponent register; any measurement contracts trace
distance, so TV ≤ ½‖ρ_c − ρ_ex‖₁. Write Φ_E = χ_{u(E)} − m_E + d_E with m_E the flat vector on
G_{u(E)} \ S_E and d_E the flat vector on S_E \ G_{u(E)}; ‖m_E‖² = ‖d_E‖² = δ_E (permutations
preserve 4^c). For vector families v, w let [v,w] be the matrix (E,E') ↦ ⟨w_{E'}|v_E⟩; then
‖[v,w]‖₁ ≤ (Σ‖v_E‖²)^½ (Σ‖w_E‖²)^½ (Hölder, Schatten 2·2). Expanding,

    K_c − K_ex = −[m,χ] − [χ,m] + [d,χ] + [χ,d] + [m,m] + [d,d] − [m,d] − [d,m].

* \[m,χ\](../E,E') = ⟨χ_{u(E')}|m_E⟩ = δ_E·[u(E)=u(E')] because m_E ⊆ G_{u(E)} has flat amplitudes:
  one rank-one block δ|_C 1ᵀ per class C, so ‖[m,χ]‖₁ ≤ Σ_C |C|^½ ‖δ|_C‖ ≤ 2^t δ_rms
  (Cauchy–Schwarz over classes). Same for [χ,m].
* \[d,χ\](../E,E') = θ(E, u(E')) with θ(E,u) = |G_u ∩ S_E \ G_{u(E)}|/4^c: columns constant on classes,
  Σ_u θ(·,u) 1_{C_u}ᵀ, trace norm ≤ Σ_u |C_u|^½ ‖θ(·,u)‖ ≤ 2^{t/2}(Σ_E Σ_u θ(E,u)²)^½ ≤ 2^t θ_rms
  (Σ_u θ(E,u)² ≤ θ_E²). Same for [χ,d].
* [m,m], [d,d] are PSD Gram matrices: trace norm = trace = 2^t δ̄ each.
* [m,d]: m_E ⊆ ∪G, so only the part of d_{E'} inside ∪G (squared norm θ_{E'}) contributes:
  ‖[m,d]‖₁ ≤ 2^t (δ̄ θ̄)^½. Same for [d,m].

Summing and multiplying by 2^−t/2 gives the bound. ∎

The **flat amplitudes** are what make it linear: the cross terms between the ideal state and the
error are rank one per class, so they cost δ, not √δ. For *pure* states the trace distance is
genuinely Θ(√δ) — that is Gidney's Theorem 2.7 (trace distance ≤ 2√ε) — but phase estimation
only sees ρ on the exponent register. The reference family is free; the bound is best when G_u is
where most paths of class u actually end, so **only disagreement between paths of the same class
costs**: a drift common to all E, or garbage common to a class, is free.

**Check** `theorem_a_gram_lemma_bounds_tv`: ten (N, a, w_e, w_m, c), every square reference
G_u = {(u + JN, J'N) : J ∈ [o_x, o_x + 2^c), J' ∈ [o_b, o_b + 2^c)} with |o| ≤ K + 1 and the
"majority" reference (most frequent 4^c keys of each class, when disjoint): TV ≤ bound always,
and **TV / best bound = 0.49–0.62** in every non-trivial case
(`research/data/theory-coset/tv_small.log`; the square references give 0.41–0.59).

## D. Faithful branches, and why the "deviant weight" does not accumulate

Write x = u + J_x N, b = J_b N (mod M) with signed coset indices. A_h(x) ≡ h·x̂ (mod N) for the
*integer* x̂ ∈ [0, M) held in the register (the lookups read all n + c bits), so:

**Lemma D.** (i) A lookup on x returns residues summing to h·u iff J_x ∈ [0, L_u), L_u = #{J ≥ 0 :
u + JN < M} ≥ 2^c. (ii) b is never looked up; it only enters as an addend of x' = b + A_h(x), which is
computed mod M, so an out-of-range J_b (b "wrapped", b mod N ≠ 0 as an integer) is harmless:
x' = u' + (J_b + q)N mod M with q = ⌊A_h(x)/N⌋ ∈ [0, K−1] regardless. (iii) Hence a branch is
*faithful* (congruent to the exact state at the end, x ≡ a^E, b ≡ 0 modulo the signed index) iff
every **looked-up** register — x before the window, x' inside it — had its index in range; and the
indices then follow the walk

    (J_x, J_b) ↦ (J_b + q, J_x − q'),    q, q' ∈ [0, K−1] (carry counts of the K table values),

so J_x after two windows moved by q_2 − q'_1: a walk with increments in [−(K−1), K−1] and mean
≈ 0. ∎

**Check** `lemma_d_faithful_branches_are_congruent_and_b_wraps_reabsorb` (exhaustive, four cases):
faithful ⇒ exactly the predicted (x, b); and 11–36 % of all branches are faithful yet had b out of
range at some window.

This explains §3.2 of ge-shor: its "deviant weight" (b mod N ≠ 0 or x mod N wrong, measured after
each window) is dominated by accumulators that are below index 0 *right now* — a snapshot of
≈ E[q']·2^−c, re-created and re-absorbed every window, hence (0.5–1.5)·2^−c "after the first
window and growing only slowly". The irreversible part is small: the first unfaithful lookup
happens in ≈ 0–0.6·2^−c of the branches per window at n = 5–7 (`tv_small.log`) and
≈ 0.4·2^−c per window at 31 bits (≈ 26–30 additions per window there, i.e. ≈ 0.015·2^−c per
addition).

## B. Power-of-two order (N = 15, 51, 85)

**Theorem B.** Let r = ord(a) = 2^s.
1. Φ_E depends only on E mod r.
2. The coset output law is supported on {k·2^t/r} (the exact support) and
   p(k·2^t/r) = f_k† κ f_k / r², f_k = (e^{2πika/r})_a, κ(a,a') = |S_a ∩ S_{a'}|/4^c.
3. TV = 0 ⇔ κ = I ⇔ the r class supports S_0 … S_{r−1} are pairwise disjoint; in general
   TV = ½Σ_k |f_k†κf_k/r² − 1/r| ≤ (Σ_{a≠a'} κ(a,a'))/(2r).
4. A key shared by two classes is unfaithful in at least one of them, so only garbage
   collisions can make TV > 0.

*Proof.* 1: windows run highest powers first; a window whose digit weight is 2^σ with σ ≥ s has
multiplier a^{e2^σ} = 1 for every digit, so it applies the same permutation to every branch, and
these windows form a prefix of the schedule. Every other window's multiplier depends on e·2^σ
mod 2^s, i.e. on bits of E below s only. So Φ_E = L_{E mod r} V ψ₀. (The same holds if the trivial
windows came last: a common unitary does not change the Gram matrix.) 2: Σ_{E ≡ a} ω^{yE} =
ω^{ya}(2^t/r)[2^t/r | y]. 3: f_k†κf_k/r² = 1/r + r^−2 Σ_{a≠a'} κ(a,a') cos(2πk(a−a')/r); at k = 0
every term is ≥ 0, so equality with the exact law (1/r at every k) forces Σ_{a≠a'}κ = 0; the bound
follows from |cos| ≤ 1. 4: a key faithful for class i has x ≡ a^i (mod N), which holds for one class
only. ∎

Why the order and not anything else: when r is a power of two the class of E determines the whole
multiplier sequence (Theorem B.1), so **all paths that end in the same residue drift their coset
indices identically** and the within-class error of Theorem A is exactly zero. When r is not a power
of two, exponents of one class reach it through different multiplier sequences, whose index drifts
differ.

**Checks.** `theorem_b_power_of_two_order`: all N ≤ 45, all bases of power-of-two order,
w_e ∈ {1,2}, c ∈ {1,2} (232 configurations): 1, 2, the κ formula = full distribution, TV = 0 ⇔
off-diagonal κ = 0, and TV ≤ off/(2r). All 232 have TV = 0. `theorem_b_collision_example_exists`:
N = 53, a = 30 (r = 4), w_e = 1, c = 1 has a cross-class collision and TV = 1/32 exactly — so
"power-of-two order ⇒ TV = 0" is **false** as a general statement. Wider sweeps
(`pow2_we2.log`, `pow2_we1.log`, N ≤ 129, c ≤ 3): TV = 0 at every c ≤ 3 for 342/350 (w_e = 2) and 347/350 (w_e = 1)
(N, a) pairs; the failures have small r-weighted collisions (TV ≤ 0.010) and vanish by c = 3
in this range, but N = 257, a = 3 (r = 256), w_e = 1, c = 1 has 2·10^6 colliding key pairs.
**Observation** (not a theorem) `observation_non_power_of_two_order_always_deviates`, and
`zero_we2.log` / `zero_we1.log`: every base of non-power-of-two order (660 configurations with
N ≤ 63 at w_e = 2, 320 with N ≤ 45 at w_e = 1, c ≤ 3) has TV > 0.

## C. Rigorous worst case, and the actual (random-walk) law

**Theorem C (additive worst case).** For the unshifted square reference (o = 0), δ_E ≤ A·2^−c for
every E, hence TV ≤ 4·A·2^−c.

*Proof.* Encode a register holding residue s as s + JN with J ∈ [0, 2^c) (Gidney's coset encoding,
applied to both registers, intermediate partial sums included). Adding a table value T < N into
the accumulator raises its index by 0 or 1; subtracting lowers it by 0 or 1; so each of the A
additions sends at most a 2^−c fraction of the 4^c encodings of its input out of the encoding set
(Gidney's deviation ≤ 2^−c), lookups only read registers that are still encoded (Lemma D), and
deviation is subadditive under composition (Gidney Thm 2.11). Then Theorem A with θ ≤ δ. ∎

**Check** `theorem_c_additive_worst_case`: for every E and every prefix of the schedule,
δ_E ≤ (additions so far)·2^−c; the worst ratio observed is 0.22–0.38 of the bound.

Comparison of what each bound demands at 31 bits (EH, w_e = 2, w_m = 3, A = 48·K ≈ 720–912) for
output deviation ≤ 1 %:

| bound | c needed |
|---|---|
| Gidney 2019 Thm 2.7 + 2.11: trace distance ≤ 2√(A·2^−c) | 26 |
| GE19 heuristic c_pad = 2 lg n + lg(1/ε) | ≈ 17 |
| Theorem C: TV ≤ 4A·2^−c | 19 |
| **Theorem A evaluated on the 31-bit circuit (Monte Carlo, below)** | **12** (13 with margin) |
| expected actual TV (bound × 0.41–0.62, the exhaustive ratio) | 11–12 |

**The actual law is not additive.** `scan_windows.log` (exhaustive, N = 35, 55, 65, c = 4, w_e = 1,
t = 1…14 windows): TV = 0 while 2^t ≤ r (every exponent is its own class; Thm A with
G_u = S_E gives δ = 0), then TV·2^c grows **sub-linearly** with windows: N = 55: 1.56 (6 windows) →
2.40 (8) → 3.58 (10) → 3.89 (12); N = 65: 0.27 → 0.78 between 6 and 14. At fixed windows
(N = 55, 6 windows) TV·2^c = 1.48, 1.70, 1.98, 2.60, 3.33 for K = 2, 3, 4, 5, 10: **∝ √K**
(ratio 2.25 for K × 5; √5 = 2.24). Both are the signature of the index random walk of Lemma D:
the misplaced fraction is the walk's spread across the boundary of the reference square,
≈ 2·E|J − J₀|·2^−c with variance ∝ K per window.

At 31 bits (Monte Carlo below, 24 windows): the index variance grows **linearly**, 3.2–4.4 per window
(86 after 24 windows at c = 8); δ̄·2^c after k windows fits 3.1√k + 0.06k (c = 8, rms 0.24) — a
pure √k fit is better than a linear one at every c (`fit_growth.py`); 2·√(2/π)·√86 = 14.8 vs the
measured δ̄·2^c = 17.1. The irreversible unfaithful part is ≈ 0.36–0.42·2^−c per window.

So the measured constant TV·2^c ≈ 1.5–7.5 of ge-shor (≈ 1.1–2.7 for c ≥ 4 at n = 5–7) is
≈ ½ × (Theorem A bound) and the bound is ≈ 2 × the walk's boundary spread; with ~5–6 windows and
K ≈ 5 that spread is ≈ 0.6–1.2 index units per coordinate (`tv_small.log`, mean |ΔJ|).

## D'. Padding at 31 bits (replaces ge-shor §3.3's extrapolation)

`theory_coset mc`: exact 31-bit EH schedule (N = 1 537 596 787, m = 16, registers (16, y^−1),
(32, g), w_e = 2, w_m = 3, 24 windows, K = ⌈(31+c)/3⌉), the per-branch walk of the *exact*
permutation (same tables as the circuit), E uniform. For every prefix length a square offset is
chosen on an independent pilot sample; the main sample (400 exponents × 1 000 branches, or
2 000 × 2 000) estimates δ_E per exponent, hence δ̄, δ_rms (bias-corrected for binomial noise) and
θ. Theorem A then bounds TV (up to Monte-Carlo error, standard errors quoted).

| c | bound (a = 3) | bound (a = 5) | bound·2^c | Toffolis (w_e=2, w_m=3) | + coset prep 2c(n+c) |
|---|---|---|---|---|---|
| 8 | 0.134 | 0.131 | 34 | 45 926 | 46 550 |
| 10 | 0.041 | 0.036 | 37–42 | 49 864 | 50 684 |
| 11 | 0.019 | 0.017 | 34–39 | 51 482 | 52 406 |
| **12** | **0.0097 (±0.0003)** | **0.0092 (±0.0003)** | 38–40 | **54 501** | **55 533** |
| 13 | 0.0046 | 0.0048 | 38–39 | 55 697 | 56 841 |
| 14 | 0.0025 | 0.0021 | 35–41 | 57 304 | 58 564 |
| 16 | 0.0006 | 0.0007 | 40–44 | 61 785 | 63 289 |
| exact EH (w_e = 4, w_m = 3) | 0 | 0 | | 61 087 | |

θ = 0 in every 31-bit sample (no branch of one class landed in another class's square).
Larger samples at c = 12 (2 000 exponents × 2 000 branches, `mc31_eh_a*_c12_big.log`):
**bound 0.92 % (a = 3) and 0.88 % (a = 5)**, standard error 0.012 %, upper 2σ 0.94 % / 0.91 %.
The 31-bit *Shor* schedule (62 exponent bits, 31 windows, same c) gives 1.03 %
(`mc31_shor_a3_c12.log`): 1.12× the EH bound for 1.29× the windows — √(31/24) = 1.14, the
random-walk law again, not the additive one.

**Recommendation.** c = 12: the Theorem-A bound is 0.88–0.92 % (2σ upper ≤ 0.94 %),
and the actual TV is expected at 0.4–0.6 % (exhaustive ratio TV/bound = 0.41–0.62). If the
guarantee itself must hold with margin, c = 13 (bound ≤ 0.48 %). Any success criterion
(order recovered, lattice post-processing succeeds) changes by at most TV.

**Does coset still win?** Yes. Exact counts from `ge_shor counts` (`counts31.txt`; w_e = 2,
w_m = 3 is the cheapest coset configuration at every c = 8…16 among w_e ∈ {2,3,4},
w_m ∈ {2,3,4}): c = 12 is 54 501 Toffolis, 55 533 with the coset-state preparation that ge-shor did
not count, vs 61 087 for the best exact EH oracle: **−9.1 %** (c = 13: −7.0 %; c = 14: −4.1 %).
Break-even is c ≈ 15 (60 551 + 1 380 > 61 087). Qubits: 176 vs 168. ge-shor's headline 45 926
(c = 8) has TV ≤ 13 % by Theorem A (≈ 5–8 % expected), so it should be quoted at c = 12 for a 1 %
target. ge-shor's extrapolation TV ≈ (9–35)·2^−c happens to land at the right c (12) for the wrong
reason: it scaled with the number of additions, which overestimates growth with windows and
underestimates the constant.

## What is new vs known

* Known: Zalka 2006 (coset representation, error ~2^−c per operation); Gidney 2019 (1905.08488:
  deviation ≤ 2^−m per coset addition, subadditive, **trace distance ≤ 2√(total deviation)**);
  GE19 (c_pad = 2 lg n + lg(1/ε), additive budget).
* New here: (1) for phase-estimation outputs of flat (permutation-of-uniform) encodings the output
  error is **linear** in the misplaced fraction (Thm A) — the square root in Gidney's bound is an
  artefact of bounding the full pure state; this alone moves the 31-bit requirement from c ≈ 26 to
  c = 19 rigorously (Thm C). (2) The reference can be chosen per class, so only **path-dependent**
  drift matters; with the index random-walk picture (Lemma D: accumulator wraps re-absorb) the
  actual misplaced fraction grows like √windows·√K, and evaluating Thm A on the real 31-bit circuit
  gives c = 12. (3) The exact characterisation of the TV = 0 cases (Thm B): power-of-two order
  removes all within-class error; residual error is cross-class garbage collisions, which exist
  (explicit counterexamples). (4) ge-shor's per-window "deviant weight" is mostly a snapshot of
  re-absorbable accumulator wraps, not accumulated error.
* Caveats: the 31-bit bound is a Monte-Carlo estimate of a rigorous quantity (the estimator is
  unbiased for δ̄; δ_rms uses the binomial bias correction); the "expected TV" uses the exhaustive
  small-n ratio TV/bound, which is an empirical constant, not a theorem. Theorem A and C are about
  the measured output law (exponent register); they do not bound the error of the work register
  itself, for which Gidney's √ bound is the right one. Coset-state preparation cost is estimated
  (2c(n+c)), not counted from a circuit.
