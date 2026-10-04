# Gidney–Ekerå techniques in the exact gate-level Shor simulation (exp/ge-shor)

Branch `exp/ge-shor`, based on `exp/mbu-shor` (`f1258f6`, = main `d44563c` +
MBU). Code: `src/shor_ge.rs` (oracle, windowed engine, Ekerå–Håstad,
coset arithmetic), small changes to `src/shor_mbu.rs` (uncontrolled
lookups, a global-sign op) and `src/shor/sliced.rs` (global sign in the
sign check). Tests: `src/shor_ge.rs` (3 unit tests), `tests/ge_shor.rs`
(7). Driver: `examples/ge_shor.rs` (`run`, `counts`, `coset`, `cosetmc`,
`ehmc`). Data: `research/data/ge-shor/`.

**Machines.** Builds, tests, counts and the Monte-Carlo / coset
campaigns ran on the VPS (nice 15, `-j 2`, 2 threads). Timings ran on the
Mac (M1 Pro, 8 cores, 16 GB) under `/tmp/qsim-mac-bench.lock`, one
31-bit run per lock (≤ 150 s), ≥ 65 s gaps, interleaved with the
`exp/mbu-shor` oracle built from the same checkout. The Mac's 1-min load
from other agents was 8–13 during the timings.

## TL;DR

* **The "≈ 13× gap to Gidney–Ekerå" of research/mbu-shor.md was mostly an
  artefact of evaluating an asymptotic formula at n = 31.** GE19's headline
  `0.3 n³ + 0.0005 n³ lg n` (9.0 k Toffolis at n = 31) is their cost model
  at RSA sizes. Their own construction, costed the same way at n = 31
  (best windows `c_exp = c_mul = 3`), needs **≈ 44 k** Toffolis
  (`research/data/ge-shor/cost_model.py`; the reconstruction reproduces
  their headline to 2 % at n = 2048). Against that, the MBU oracle's
  119 096 is 2.7×, not 13×.
* **All three GE19 techniques are now in the gate-level simulator**, with
  every gate applied to every branch:
  1. **Exponent windowing** (`w_e` exponent qubits live at once; one
     multiply-add per window with lookups addressed by `w_e` exponent bits
     and `w_m` multiplicand bits; uncontrolled lookups, plain swap). Exact:
     the distribution of the measured integer equals the textbook one to
     `1e-12` for every window size tried, and a gate-by-gate quantum
     reference (sparse state vector, real H / Phase / projective
     measurement of each exponent qubit) agrees to `1e-10`.
  2. **Ekerå–Håstad short exponent** (factoring as the short discrete log
     `d = (p+q−2)/2` of `y = g^{(N−1)/2}`): `3⌈n/2⌉` exponent bits instead
     of `2n` (48 vs 62 at 31 bits), lattice post-processing (2-D Lagrange
     reduction + enumeration, every candidate verified by `g^d = y` and
     `p·q = N`). Exact: the gate-level distribution of `(j, k)` equals the
     textbook EH distribution to `1e-12`. **Success probability of one
     run**: 0.94 / 0.93 / 0.92 exactly (N = 35, 77, 143); 0.88 ± 0.03
     (16-bit, 400 runs) and 0.93 ± 0.04 (20-bit, 200 runs) by Monte Carlo.
  3. **Coset representation** (Zalka; GE19 §2.4): `n + c`-qubit registers
     holding `Σ_j |x + jN⟩`, plain `(n + c)`-bit Gidney additions, no
     modular reduction. **Approximate**, and the simulator measures the
     approximation exactly: total variation distance to the exact
     output distribution halves per padding qubit, TV ≈ (1.5–7.5)·2^{−c}
     at n = 5–9 bits, growing with the number of additions (0.16 / 0.08 /
     0.04 at c = 4 / 5 / 6 for N = 55, exact).
* **Toffolis per factoring run at 31 bits** (N = 1 537 596 787; the
  exp/mbu-shor record circuit was **119 096**):

  | technique (cumulative) | Toffolis | vs MBU |
  |---|---|---|
  | MBU oracle (exp/mbu-shor), 62 controlled multiplications | 119 096 | — |
  | + exponent windowing (w_e = 3, w_m = 4), exact | **75 950** | −36 % |
  | + Ekerå–Håstad (48 exponent bits), exact | **61 087** | −49 % |
  | + coset arithmetic, c = 8 (approximate) | **45 926** | −61 % |
  | + coset arithmetic, c = 12 (approximate) | 54 501 | −54 % |
  | GE19 construction at n = 31 (their cost model) | ≈ 44 350 | |

  At c = 8 we are at GE19's own construction cost for n = 31; at the c the
  coset data suggest for a ≤ 10 % output deviation (c ≈ 8–9) the count is
  46–49 k.
* **Simulated end to end at 31 bits** (exact windowed-exponent circuit,
  `w_e = 2`, `w_m = 3`, MBU lookups): measured the same integer as every
  earlier run (2 059 039 373 337 077 151), found r = 256 252 500 and the
  factor 52 501. **TBD s** (min of 3) vs **TBD s** for the exp/mbu-shor
  record oracle in the same session, 164 443 vs 218 421 Toffolis
  (−25 %), peak RSS 4.19 GB vs 4.28 GB. The windowed circuit is
  **slower to simulate** although it has fewer Toffolis: the simulator
  must evaluate the block on every exponent value, `2^{w_e}` branches per
  stored `x`, and each evaluation is a whole windowed multiplication with
  bigger lookups (§2.4).
* **Ekerå–Håstad is cheaper to simulate than Shor** when the base has odd
  order: `g = h^{2^n}` (no factorisation needed; any base works for EH)
  keeps the support at `r_odd` for the whole run (24-bit: 0.21 s vs
  0.57 s for Shor; peak support 79 335 vs 2 538 720/4). With a random base
  the EH support reaches the full order `r` (the 2-D phase estimation
  fills `⟨g⟩`), up to 4× Shor's peak.

## 1. Engine: semiclassical windows

### 1.1 Circuit

Shor's (or EH's) exponent register is processed in windows of `w = w_e`
consecutive rounds `i0 … i0 + w − 1` (round `i` applies `U^{2^{t−1−i}}`,
records bit `y_i`, LSB first, as before). The window's block maps

    |e⟩_w |x⟩ |0…⟩  →  |e⟩_w |g^e x mod N⟩ |0…⟩,    g = a^{2^{t − i0 − w}},

for all `e < 2^w`, as the multiply-add `b += g^e x`, swap, and the exact
inverse of `b += g^{−e} x` (`window_ops`). Each multiply-add is a sum over
multiplicand windows of `w_m` bits of lookups `T[e, v] = g^e·v·2^{k w_m}
mod N` addressed by `w_e + w_m` qubits, followed by the modular adder of
exp/mbu-shor (Gidney adders + measured flag, or Cuccaro + measured flag)
and the measurement-based unlookup. Because `e` is in the address, the
lookup is **uncontrolled**: the unary iteration's root is the constant 1
(the right child's flag is the top address bit itself, the left child's
is its complement), `2^{w_e+w_m} − 2` Toffolis; and the swap needs no
control (`e = 0` multiplies by 1). An uncontrolled phase table can have a
constant term, a *global* phase −1; it is emitted as `MbuOp::GlobalNeg`,
and the engines check "every branch has the program's global sign"
instead of "+1".

The exponent qubits are then measured one at a time, **highest power
first**, each after the Griffiths–Niu phase `−π·y_{<i}/2^i` that includes
the bits of the same window measured before it, and `H`.

### 1.2 Why the distribution is unchanged, and the support law per window

**Proposition W.** Let `ψ = ψ_{i0}(y)` be the state of the work register
after `i0` measured bits (Lemma 1 of research/theory-shor.md). After the
window block the state is `2^{−w/2} Σ_{e<2^w} |e⟩ V^e ψ`, `V = U^{2^{t−i0−w}}`.
After the top `j` exponent qubits are measured with outcomes `y_{i0}, …,
y_{i0+j−1}`, the state is

    2^{−(w−j)/2} Σ_{e'' < 2^{w−j}} |e''⟩ V^{e''} ψ_{i0+j}(y'),

with `ψ_{i0+j}(y')` exactly the unwindowed circuit's state after `i0 + j`
rounds and the outcome probabilities equal to the unwindowed ones.

*Proof.* The block equals `Π_h C_h(V^{2^h})` (controlled powers on
distinct controls; they commute). The measurement of exponent qubit `h`
(phase, H, projection) acts on that qubit only, so every `C_{h'}`,
`h' < h`, commutes with it and can be applied afterwards. Hence measuring
the top qubit is round `i0` of the unwindowed circuit applied to `ψ`,
followed by the remaining controlled powers on `|+⟩^{w−1}`; induction on
`j`. ∎

**Consequences** (support law T1 for windowed rounds). (a) After `j`
measurements there are exactly `2^{w−j} |S_{i0+j}(y')|` branches, `|S|`
given by Theorem 1(b), at most `2^{w−j} B_{i0+j}`; (b) the peak inside a
window is `2^w |S_{i0}|`; (c) the work counter is
`W = Σ_windows 2^w |S_{i0}| G_window`; (d) the distribution of the
measured integer is the textbook one. For Shor the last window's
`2^w |S_{t−w}| = r` when `w ≤ ν₂(r)` (31-bit: `4 · r_odd = r =
256 252 500`, which is the measured peak); the stored peak between
windows is lower than the unwindowed `max(r_odd, r/2)` because the final
state is never materialised.

**Checks.** `windowed_support_law_on_trees` walks measurement trees of
the real gate-level windowed engine (N = 15, 21, 143, 65, 91; w_e = 2, 3,
4; 17 210 nodes, including inside-window nodes): at every node there are
`2^{w−j}` arrays, **each of size exactly the closed form of Theorem 1(b)**
(deficient nodes included), all with the same multiset of amplitudes
(`V^{e''}` permutes), and the work counter equals (c).
`windowed_distribution_matches_textbook`: N = 15, 21, 35, all coprime bases
tested, `w_e = 1…4`, three MBU option sets: the whole-tree distribution
equals the 3n-qubit full-QFT distribution to `1e-12`.
`windowed_gate_by_gate_matches`: the independent quantum reference
(`distribution_sparse`: exponent qubits as real qubits of a sparse state
vector, H, every gate, every MBU X-measurement as H + projection with
`P = 1/2` asserted, then Phase / H / projective measurement per exponent
qubit) agrees to `1e-10`. `window_block_exhaustive_small`: every `e`,
every `x < N`, N ∈ {15, 21, 35, 55, 77}, `w_e, w_m ∈ 1…3`, three option
sets, three outcome streams: `g^e x mod N`, ancillas clean, uniform sign.
`windowed_run_reproduces_semiclassical_bits`: with the same RNG stream the
windowed engine measures the same integer as the permutation oracle
(up to 20 bits, `w_e = 1, 2, 3`); at 24, 28 and 31 bits it reproduced the
integers of research/mbu-shor.md.

### 1.3 The windowed engine (`GeState`)

`GeState::window` evaluates the resolved block on all `(e, x)` branches
with the bit-sliced evaluator of exp/mbu-shor, `e` set on the exponent
qubits' slice words, and checks the exponent qubits unchanged, every
ancilla 0, and the uniform sign. The `e ≥ 1` outputs are written into
copies of `ψ` appended to `ψ`'s own buffer, keys packed as
`(V^e x) << w | e` and sorted; `e = 0` is evaluated in place last, and
when it returns every key unchanged (exact arithmetic) segment 0 is still
`ψ`, already sorted, so only the `e ≥ 1` run is sorted. The `w`
measurements are then **lazy**: one pass computes the joint distribution
of the window's `w` bits by a tabulated `2^w × 2^w` linear map per key (the
Griffiths–Niu phases of later bits depend on earlier outcomes, so each
outcome prefix has its own phase), the bits are drawn one conditional
draw per round (the same RNG consumption as `shor::run_semiclassical`),
and one more pass writes the new state **in place over the window's own
array**, combining the `2^w` values per key level by level exactly as
`(C_lo ± e^{iφ}C_hi)/√(2p)` would on materialised arrays (the same
floating-point operations, rounded to the amplitude type after every
level), so exact cancellations are still exact zeros. Memory: the window
array `2^w |S|` entries (16 B with f32), nothing else.

### 1.4 Cost of simulating a windowed circuit

The simulator's work per exponent bit is `(2^{w_e}/w_e) · |S| ·
steps(window block)`; the old oracle's is `2 · |S| · steps(round block)`.
A window block is a whole multiply–swap–unmultiply with lookups over
`w_e + w_m` address bits, so for `w_e = 2` it has more slice steps than
one old round (31-bit: 23.5 k vs 14.4 k), and `2^{w_e}/w_e = 2` equals
the old 2 evaluations per bit: **1.63× the old gate work**
(`gate_branch_ops` 1.07·10¹⁴ vs 6.85·10¹³), for −25 % Toffolis (−36 %
with Gidney adders). `w_e = 3` would be `8/3` evaluations per bit. Fewer
Toffolis are not cheaper to simulate: the exponent register is quantum,
and the simulator pays for all `2^{w_e}` of its values.

## 2. Ekerå–Håstad (`eh_regs`, `eh_postprocess`, `eh_run`)

For `N = pq` with `p, q < 2^{⌈n/2⌉}` (balanced), `y = g^{(N−1)/2} = g^d`
with `d = (p + q − 2)/2 < 2^m`, `m = ⌈n/2⌉`, because `λ(N) | φ(N)/2 =
(N−1)/2 − d`. The quantum part is two-register phase estimation of
`g^a y^{−b}` with `a < 2^{2m}`, `b < 2^m` (EH17 with `s = 1`, `ℓ = m`):
`3m` controlled multiplications (48 at 31 bits instead of 62), each
register with its own semiclassical QFT; both are just exponent
registers for the windowed engine (`ExpReg`). The `m`-bit register runs
first (it does not change the distribution of `(j, k)`; it keeps the
support small, §2.2).

**Post-processing** (EH17 §4–5; Ekerå 2020, "On post-processing in the
quantum algorithm for computing short discrete logarithms", DCC 88): a good
pair has `{d j + 2^m k}_{2^{2m}}` small, i.e. the lattice `L` spanned by
`(j, 1)` and `(2^{2m}, 0)` has a vector `(d j mod 2^{2m}, d)` close to
`(−2^m k mod 2^{2m}, 0)`. We Lagrange-reduce the basis exactly (i128) and
enumerate **every** lattice vector in the box `|v_1 − u_1| ≤ 2^{m+1}`,
`0 < v_2 < 2^m` (a handful of candidates), accept `d' = v_2` if
`g^{d'} = y` and `z² − (2d'+2) z + N` has integer roots `p, q` with
`pq = N`. Everything after the measurement is classical, deterministic and
verified; no factorisation is used.

**Exactness.** `eh_distribution_matches_textbook`: the gate-level
distribution of `(j, k)` (windowed engine, `w_e = 1` and `2`, two MBU
option sets) equals the textbook
`P(j,k) = Σ_z |2^{−3m} Σ_{g^a y^{−b} = z} e^{−2πi(aj/2^{2m} + bk/2^m)}|²`
(brute force over all `(a, b)`) to `1e-12` for N = 35 (g = 2, 3),
77, 143.

### 2.1 Success probability per run

| N (bits) | method | EH, random base | EH, odd-order base `h^{2^n}` | Shor, same bases |
|---|---|---|---|---|
| 35 (6) | exact (whole distribution) | 0.9388 (g = 2 and 3) | | |
| 77 (7) | exact | 0.9346 | | |
| 143 (8) | exact | 0.9207 | | |
| 60 491 (16) | gate-level Monte Carlo, 800 / 400 runs | 0.884 ± 0.022 | 0.915 ± 0.027 | 0.921 ± 0.019 |
| 1 005 973 (20) | 500 / 300 runs | 0.912 ± 0.025 | 0.883 ± 0.036 | 0.918 ± 0.024 |
| 10 161 323 (24) | 60 runs | 0.90 ± 0.08 | 0.97 ± 0.05 | 0.97 ± 0.05 |

(±: 95 % normal intervals; `research/data/ge-shor/eh_mc.log`; every run is
a full gate-level simulation with `w_e = 2`, `w_m = 3`, MBU lookups.)
"Shor" is order finding on the same random base with the repo's
post-processing (convergents, up to 256 multiples, small-factor
stripping, then `gcd(a^{r/2} ± 1, N)`); it found the order in 98–100 % of
runs and a factor in ≈ 92 %. One EH run succeeds about as often as one
Shor run here, with 25 % fewer multiplications. These sizes are far
below EH's asymptotic regime (`r` is only ≈ `2^{2m}`, not ≫ `2^{ℓ+m}`), so
the numbers are measurements, not EH's bounds.

### 2.2 Simulator support for EH

The work register holds `g^a y^{−b}`. If the `2m`-bit register runs first,
its last rounds (`g^2`, `g`) fill the whole group `⟨g⟩` (support `r`), and
all `m` rounds of the second register run at support `r`. Running the
`m`-bit register first keeps the support inside `Y·⟨g^{2^{2m−i}}⟩`,
`Y = {y^{−b}}`, i.e. `r_odd` times the number of cosets of the odd-order
subgroup that `Y` meets: `r_odd` if `d ≡ 0 (mod 2^{ν₂(r)})` (the 31-bit
N: d = 40 892 ≡ 0 mod 4 = 2^{ν₂(r)}), up to `r` otherwise. Choosing an
**odd-order base** `g = h^{2^n}` (no knowledge of the factors needed; EH
works for any base of large enough order) makes `ν₂(r) = 0` and the
support `≤ r_odd` throughout, *smaller* than Shor's `max(r_odd, r/2)`
(Shor needs even order). 24-bit: support 79 335 and 0.21 s vs 2.5 M and
5.5 s with a random base (2 threads, VPS).

## 3. Coset representation (`GeOpts::coset`)

Registers `x` and `b` have `n + c` qubits; `x` starts as
`Σ_{j<2^c} |1 + jN⟩`, `b` as `Σ_{j<2^c} |jN⟩` (2^{2c} branches per value),
and every lookup-addition is a plain `(n + c)`-bit Gidney addition of the
looked-up residue (`n + c − 1` Toffolis instead of the ≈ 3.5 n modular
adder); the lookups read all `n + c` bits of the coset value, which works
because `Σ_k T[x_k] ≡ g^e · (x + jN) ≡ g^e x (mod N)`. Nothing is reduced,
so a branch whose coset index leaves `[0, 2^{n+c}/N)` wraps modulo
`2^{n+c}` and is no longer congruent: **the deviation**. The simulator
keeps every coset branch (the support is `≈ 2^{2c}·|S|`), so the effect on
the output distribution is computed, not modelled. (Preparing the coset
states costs `≈ 2c(n + c)` Toffolis once; not included in the counts.)

`coset_block_is_congruent` checks that most branches stay congruent.

### 3.1 Exact deviation at small N (whole distributions)

`ge_shor coset N a w_e w_m c_max` (Mac, exact tree walk): total variation
distance to the exact distribution and the *strict* success probability
(the true order is a convergent of `y/2^t`, no multiples tried; the repo's
generous post-processing recovers the order from almost any outcome at
these sizes, so it does not discriminate). `w_m = 2`.

| N (bits) | r | w_e | c = 1 | 2 | 3 | 4 | 5 | 6 | P_strict exact → c = 6 |
|---|---|---|---|---|---|---|---|---|---|
| 21 (5) | 6 | 2 | 0.245 | 0.167 | 0.134 | 0.071 | | | |
| 35 (6) | 12 | 2 | 0.477 | 0.309 | 0.185 | 0.086 | | | |
| 55 (6) | 20 | 1 | 0.920 | 0.872 | 0.517 | 0.243 | 0.116 | 0.061 | 0.393 → 0.369 |
| 55 (6) | 20 | 2 | 0.754 | 0.663 | 0.343 | 0.163 | 0.081 | 0.044 | 0.393 → 0.377 |
| 57 (6) | 18 | 2 | 0.884 | 0.703 | 0.429 | 0.216 | 0.111 | 0.052 | 0.329 → 0.313 |
| 65 (7) | 12 | 2 | 0.589 | 0.383 | 0.155 | 0.078 | 0.038 | 0.022 | 0.333 → 0.328 |
| 77 (7) | 30 | 2 | 0.881 | 0.656 | 0.389 | 0.196 | | | 0.265 |
| 15, 33, 51, 85 | 4, 10, 8, 8 | | 0 | 0 | 0 | 0 | 0 | 0 | |

TV **halves per padding qubit** once `c ≥ 3`. Orders that are powers of
two (and N = 33) show no deviation at all: their output peaks sit on
multiples of `2^t/r` exactly, and the deviant branches only change
amplitudes that are zero anyway. N = 65 (just above 2^6) deviates least:
its register has `2^{n+c}/N ≈ 2^c · 1.97` coset slots, nearly twice the
`2^c` that the initial superposition uses, so most index drift fits
without wrapping.

### 3.2 Larger N by exact path likelihoods

`ge_shor cosetmc` samples outcome paths from the coset circuit and runs
the exact circuit in lockstep on the same outcomes, so `Q(y)`, `P(y)` and
the deviant weight after every window are exact per path; TV =
`E_Q[(1 − P/Q)_+]` is an unbiased path average (standard error ≤
0.5/√paths: ±0.035 at 200 paths, ±0.065 at 60, ±0.09 at 30). VPS,
`w_e = w_m = 2`, `research/data/ge-shor/coset_paths.log`:

| N (bits) | c = 2 | c = 4 | c = 6 | c = 8 | deviant weight per window, c = 4 / 6 / 8 | peak coset support (c = 6) |
|---|---|---|---|---|---|---|
| 143 (8) | 0.58 | 0.21 | 0.08 | ≈ 0 (30 paths) | 0.04–0.08 / 0.01–0.02 / ≤ 0.007 | 0.86 M |
| 221 (8) | 0.57 | 0.15 | 0.07 | 0.03 | 0.06–0.08 / 0.03 / 0.01 | 0.17 M |
| 391 (9) | 0.82 | 0.27 | 0.12 | 0.03 | 0.14–0.19 / 0.04–0.07 / 0.01–0.02 | 4.5 M |
| 899 (10) | 0.96 | 0.39 | 0.04 (60 paths) | | 0.15–0.24 / 0.01–0.04 / | 59 M |

The **deviant weight** (probability on branches that are no longer
congruent to the exact state, i.e. that wrapped) is ≈ (0.5–1.5)·2^{−c}
after the first window and **grows only slowly** over the run — by a
factor 0.05–2 from the first to the last of 8–10 windows, not by the
number of windows: a wrapped branch is a garbage residue that no longer
interferes with the good ones, and each window's measurement renormalises
the state (for small orders the good part dominates the outcomes and the
deviant weight even shrinks, N = 143). The output TV is 3–8× the
per-window deviant weight. So the deviation does not accumulate like
`(number of additions)·2^{−c}` along one path; the fit of §3.3 in A is an
upper-bound style extrapolation.

### 3.3 Padding needed at 31 bits

Fitting `TV ≈ κ · A · 2^{−c}` (A = number of lookup-additions in the
run, `2 ⌈(n+c)/w_m⌉` per window) to §3.1–3.2 at c = 6 gives κ between
0.014 (N = 65, lots of headroom `2^n/N`) and 0.052 (N = 391), with no
clear trend in n beyond A. The 31-bit EH circuit (`w_e = 2`, `w_m = 3`,
24 windows) has A ≈ 670, so TV ≈ (9–35)·2^{−c}: c ≈ 9 for TV ≤ 0.1,
c ≈ 12 for TV ≤ 0.01. GE19 use `c_pad = 2 lg n + lg(1/ε)` (≈ 13–17 here).
The Toffoli counts for c = 8 / 12 / 16 at 31 bits are 45 926 / 54 501 /
61 785. This extrapolation from n ≤ 9 is the weakest number in this note.
The strict success probability degrades less than TV: at c = 6 it is
within 0.025 of the exact circuit's for every N of §3.1.

## 4. Counts

Whole runs, every op of every resolved block (outcome streams as in the
engine), `research/data/ge-shor/counts_all.txt` (`ge_shor counts`; 544
configurations: N of 20/24/28/31 bits, Shor and EH, `w_e = 1…4`,
`w_m = 2…5`, MBU `lookups` / `all`, coset `c = 4…16`). "Gates" counts
every op including X-measurements and Z/CZ fix-ups; "steps" are
bit-sliced engine steps. Baseline rows are exp/mbu-shor's
(`research/data/mbu-shor/counts.txt`, `w = 4`).

**Fewest Toffolis per family** (all with Gidney adders and every MBU
construction; `w_e, w_m` chosen per row):

| n | MBU (62 → 2n mults) | + exp. windowing | + Ekerå–Håstad | + coset c = 8 | c = 12 | GE19 model at n | Toffolis/n³: MBU → c = 8 |
|---|---|---|---|---|---|---|---|
| 20 | 31 399 | 24 715 (3, 3) | 18 679 (2, 4) | 17 911 (2, 3) | 21 380 | 14 400 | 3.92 → 2.24 |
| 24 | 53 969 | 37 115 (3, 3) | 28 098 (3, 3) | 25 631 (3, 2) | 30 509 | 21 500 | 3.90 → 1.85 |
| 28 | 85 673 | 59 402 (3, 4) | 46 100 (2, 4) | 35 590 (2, 3) | 42 661 | 33 600 | 3.90 → 1.62 |
| 31 | 119 096 | **75 950** (3, 4) | **61 087** (4, 3) | **45 926** (2, 3) | 54 501 | 44 350 | 4.00 → 1.54 |

Per technique at 31 bits: exponent windowing −36 % (`w_e = 3`: one
multiplication per 3 exponent bits, but 2^7-entry lookups); EH −20 % on
top (48 instead of 62 exponent bits; slightly less than 3/4 because the
best windows shift); coset arithmetic −25 % on top at c = 8 (a plain
38-bit addition instead of a 3.5n modular adder, but `n + c` bits to look
up and swap), −11 % at c = 12. The coset gain grows with n: at n = 20 it is
only −4 %, because the lookups dominate.

**The configurations the simulator runs** (fewer slice steps, exact):

| n | oracle | qubits | Toffolis | gates | X-meas | slice steps | exponent bits |
|---|---|---|---|---|---|---|---|
| 31 | exp/mbu-shor `windowed-mbu-lookup` (w = 4) | 132 | 218 421 | 872 071 | 45 379 | 894 601 | 62 |
| 31 | exp/mbu-shor `windowed-mbu` (w = 4) | 162 | 119 096 | 1 180 796 | 143 400 | 1 252 554 | 62 |
| 31 | GE Shor, w_e = 2, w_m = 3, lookups | 134 | 164 443 | 708 798 | 41 838 | 729 835 | 62 |
| 31 | GE Shor, w_e = 2, w_m = 4, all | 165 | 81 438 | 837 830 | 94 551 | 885 340 | 62 |
| 31 | GE EH, w_e = 2, w_m = 3, lookups | 134 | 127 485 | 549 694 | 32 415 | 565 882 | 48 |
| 28 | GE Shor, w_e = 2, w_m = 3, lookups | 122 | 122 865 | 513 693 | 32 453 | 529 929 | 56 |
| 28 | GE Shor, w_e = 2, w_m = 4, all | 150 | 61 003 | 611 100 | 70 187 | 646 162 | 56 |
| 24 | GE Shor, w_e = 2, w_m = 3, lookups | 106 | 72 996 | 316 429 | 21 612 | 327 212 | 48 |
| 24 | GE Shor, w_e = 2, w_m = 4, all | 130 | 40 183 | 387 329 | 45 799 | 410 241 | 48 |

Qubits: `4n + 3 + 2w_e + w_m` (+ `n − 1` carries with Gidney adders): the
`w_e` exponent qubits (the first is the old recycled control) and
`w_e + w_m` AND ancillas for the wider lookups. Coset (Gidney adders):
`4(n + c) − 3 + 2w_e + w_m` (160 at n = 31, c = 8, `w_e = 2`, `w_m = 3`).
EH does not change the qubit count.

## 5. Timings (Mac)

TBD

## 6. Against Gidney–Ekerå 2019

`research/data/ge-shor/cost_model.py` (output in `cost_model.out`)
reconstructs GE19's count as `2·⌈n_e/c_e⌉·⌈n/c_m⌉·(2n + 2^{c_e+c_m})`
Toffolis (`n_e = 1.5n`: one lookup-addition per `c_m` multiplicand bits,
two multiply-adds per exponent window, `2n + 2^{c}` per lookup-addition).
With the best integer windows this gives 2.58·10⁹ at n = 2048, within 2 %
of their headline `0.3n³ + 0.0005 n³ lg n` = 2.62·10⁹, so the
reconstruction is faithful where it can be checked. At small n the
optimal windows are small (`c_e = c_m = 3` at n = 31) and the lookups
(`2^6`) are not negligible against `2n`, so the same model gives **44 350
at n = 31**, 4.9× the headline formula's 9 011. The "≈ 13×" of
research/mbu-shor.md §6 compared our 31-bit circuit with the formula,
not with the construction.

| n | GE19 headline | GE19 model (best c_e, c_m) | ours, measured (all three, c = 8) | ours, model with c = 2 lg n + 4 |
|---|---|---|---|---|
| 20 | 2 417 | 14 400 (3, 2) | 17 911 | 22 770 |
| 31 | 9 011 | 44 350 (3, 3) | 45 926 | 58 320 |
| 64 | 7.9·10⁴ | 2.6·10⁵ | | 2.7·10⁵ |
| 256 | 5.1·10⁶ | 9.4·10⁶ | | 7.4·10⁶ |
| 2048 | 2.62·10⁹ | 2.58·10⁹ (6, 5) | | 1.61·10⁹ |

Our model column uses the per-component costs measured in this code
(uncontrolled lookup `2^w − 2`, measurement-based unlookup
`min_k (2^{w−k} − 1) + (2^k − k − 1)`, Gidney addition `n + c − 1`);
it is below GE19 at large n only because their accounting charges `2n` per
addition (and we have no carry runways, which they need for depth, not
Toffolis). At n = 31 the measured count with c = 8 is within 4 % of the
GE19 model; with the c our deviation data suggest for a ≤ 10 % output
deviation (c ≈ 8–9, §3.3) it is within ≈ 10 %.

So, of the old "13×": ≈ 4.9× was the formula/construction mismatch at
n = 31, and the remaining 2.7× is now closed by the three techniques
(119 096 → 45 926: windowing 1.57×, EH 1.24×, coset 1.33×).

## 7. Limits / not done

* **Coset arithmetic is not simulated at 24–31 bits.** It multiplies the
  support by `2^{2c}` (every coset index is a branch), so it is exactly
  simulable only at n ≤ 9 here (c ≤ 8). Its counts at 31 bits are exact
  circuit counts; its deviation at 31 bits is an extrapolation (§3.3).
* **Oblivious carry runways** (GE19 §2.6) are not implemented: they cut
  depth, not Toffolis, and our simulator has no notion of depth.
* **Coset state preparation** (`≈ 2c(n + c)` Toffolis per run) is not in
  the counts; neither are the classical post-processing costs.
* **EH success probabilities** are measured far below EH's asymptotic
  regime (`r` comparable to `2^{2m}`), with `s = 1` only; EH's
  tradeoff `s > 1` (shorter registers, several runs) is not implemented.
* **The windowed engine needs `2^{w_e}` times the branches** of a stored
  state inside a window, so `w_e ≥ 3` is impractical at 31 bits on 16 GB
  (`w_e = 2` peaks at `r` branches, like the unwindowed final round).
  `w_e = 3, 4` circuits are counted at all sizes and simulated up to 24
  bits.
* **Noise.** As in exp/mbu-shor, the noisy trajectory engine does not run
  these oracles.
* **Gidney 2025's 2.5n modular adder and approximate residue arithmetic**
  (arXiv:2505.15917) are not combined with this; they trade Toffolis for
  qubits in a different direction (≈ 1400 logical qubits at RSA-2048).
