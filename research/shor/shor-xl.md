# The supercomputer's Shor instance on one VM, gate by gate, and the generic-N frontier (exp/shor-xl)

Branch `exp/shor-xl`, based on main `6b21728`, 5 October 2026. Code: `src/shor/sliced.rs` (AVX-512
tier of the bit-sliced evaluator, aligned slice buffers), `src/shor/ge.rs` (in-place window finish,
`factor_from_power_order`, `outcome_seed`), `examples/ge_shor.rs` (`info`, `dump`, `slicebench`,
`shor-odd`, repeated runs). Tests: `src/shor/sliced.rs` (2 new), `src/shor/ge.rs` (2 new),
`tests/shor/ge_shor.rs` (1 new). Data, scripts and every log: [`research/data/shor-xl/`](../data/shor-xl/).
The instance rules were committed before the runs: [`PREREG.md`](../data/shor-xl/PREREG.md).

**Machine.** Intel Xeon Gold 6548Y+ (Emerald Rapids) Hyper-V VM, 16 vCPU = 8 cores × 2 hyper-threads,
31 GB RAM, AVX-512, Linux 5.15, Rust 1.93.1 release build (portable target, runtime dispatch). The
machine is shared with other users and agents; every timed run held the swarm-wide bench lock, and the
1-minute load average before and after each run is in the logs and in the tables. Threads: rayon
default (16) unless stated.

## Headline

* **N_W = 549 755 813 701 = 712 321 × 771 781**, the largest N that Willsch et al. (2023) factored by
  simulating Shor's algorithm (a 40-qubit state vector on up to 2048 A100 GPUs of the JUWELS
  Booster), is factored here on one 16-vCPU VM by **exact simulation of every gate of a 165-qubit
  Ekerå–Håstad circuit** (60 exponent bits; 1.48 M operations, of which 352 k Toffolis, 82 k
  X-basis measurements) on every branch, with a base fixed in advance by a seeded rule that uses no
  knowledge of the factors: one run measures `(j, k) = (1 046 455 110 769, 358 190)`, and the
  lattice post-processing returns 712 321 × 771 781 — in **99.6 s on 16 threads at a 1-min load of
  21–27 (loaded machine), with 2.26 GB peak RSS**. The cost is set by the support,
  `ord(g) = 71 582 595` branches, small because the odd part of `λ(N_W)` is small
  (`p − 1 = 2^7·3·5·7·53`); Willsch et al. paid for the full 2^40-amplitude state whatever the
  order, so the two costs are not comparable and nothing here is faster than their simulation in
  any like-for-like sense. A second circuit (`w_e = 2`, 166 qubits, 235 k Toffolis) measures the
  same `(j, k)`, and Shor's order finding on the same base (78 exponent bits) plus Miller's
  reduction factors N_W too.
* **Generic N.** Every first balanced semiprime of 22–33 bits from the repo's seeded generator
  factors with the seed-1 odd-order base (≤ 3.6 GB, ≤ 75 s). Above 33 bits the seed-1 supports need
  14 GB to 10^10 GB, except at 43 bits: **N = 4 911 456 443 897 = 1 456 057 × 3 373 121 is factored
  by exact gate-level EH simulation (181 qubits, 66 exponent bits, 470 k Toffolis; 190 s at load
  ≈ 30, 3.6 GB)** — as far as we know the largest N not constructed to be easy (it comes from a
  seeded random-semiprime generator) that has been factored by gate-level simulation of a Shor-type
  circuit, **with its cost set by the support `ord(g) = 115 574 445`, which is small because this
  seed's base misses the factor 83 of `λ_odd = 9.59·10^9`** (a uniformly random base is this lucky
  with probability ≈ 5 %; one of full odd order would need ≈ 300 GB here). This repo's own 52-bit
  gate-level runs ([shor.md](shor.md)) used N = p(2p − 1), constructed so that `λ ≈ √(2N)`. The
  43-bit instance was picked, after computing the supports of all 22–63-bit generator instances from
  their factors, as the only one above 33 bits that fits; its `p − 1` and `q − 1` are 243- and
  127-smooth, so it is classically trivial (Pollard's `p − 1`).
* **AVX-512.** A `VPTERNLOGQ` kernel makes the bit-sliced gate evaluation 1.5× faster per thread
  than AVX2 and 1.18–1.32× faster inside whole runs; end to end the 28- and 31-bit record runs gain
  only 4–9 % (load 22–38), because sorting and merging the branches now take most of the time. The
  31-bit record run (`qsim run shor … windowed-mbu-lookup --f32`) takes 76.4 s here at a load of
  23–29 (89.3 s on the M1 Pro of [mbu-shor.md](mbu-shor.md)).
* **Memory.** Peak RSS is 16.0–16.5 B per peak branch (f32: 8-byte key + 8-byte amplitude) after an
  in-place rewrite of the window finish; the old finish needed 32.8 B per branch whenever the support
  grew inside a window (N_W: 4.59 → 2.26 GB).


## 1. Why this instance is cheap here, and not for a state-vector simulator

**The cost law.** The simulator stores the branches (basis states with non-zero amplitude) of the
exact state and applies every gate to every branch. Between exponent windows only the work register
and an amplitude are stored; inside a window the block is evaluated on all `2^{w_e}` exponent values
of every stored branch ([ge-shor.md](ge-shor.md) §1). For Ekerå–Håstad (EH) with an **odd-order
base** `g = h^(2^n)` the stored support never exceeds `ord(g)` ([ge-shor.md](ge-shor.md) §2.2), so

    peak branches = 2^{w_e} · ord(g),     work ≈ Σ_windows 2^{w_e} · |S_k| · (steps of window k).

`ord(g)` divides the odd part `λ_odd` of the Carmichael function `λ(N)`; it does not depend on `N` as
such.

**N_W.** `N_W = 549 755 813 701 = 712 321 × 771 781` (39 bits) is the largest N that Willsch et al.
factored by simulating Shor's algorithm ("Large-Scale Simulation of Shor's Quantum Factoring
Algorithm", Mathematics 11, 4222 (2023), arXiv:2308.05047). Their simulator `shorgpu` runs the
iterative (one recycled control qubit) Shor algorithm on a **40-qubit state vector**: 2^40 complex
double-precision amplitudes (16 TiB), two state buffers plus index buffers, "slightly larger than
40 TiB" in total, on up to 2048 A100 GPUs of the JUWELS Booster; the controlled modular
multiplication is applied as a permutation of the amplitudes among the GPUs, not compiled to gates.
Their cost is set by 2^40 whatever the order of the base.

Here (verified in [`orders_seed1.txt`](../data/shor-xl/orders_seed1.txt), factors used only for
this write-up and for the RAM estimate):

* `p − 1 = 712 320 = 2^7·3·5·7·53`, `q − 1 = 771 780 = 2^2·3·5·19·677`;
  `λ(N_W) = lcm = 2^7·3·5·7·19·53·677 = 9 162 572 160`, `λ_odd = 71 582 595 ≈ 7.2·10^7`;
* so **every** odd-order base `g = h^(2^39)` has `ord(g) | 71 582 595`, and the peak support is at
  most 7.2·10^7 branches, smaller than this repo's 31-bit record (`r = 2.56·10^8`, peak support
  `r_odd = 6.4·10^7` with EH). The base of the pre-registered seed 1 has the full
  `ord(g) = 71 582 595`.
* A typical 39-bit semiprime is very different: the generator's own 39-bit N
  (`478 046 366 261`) has `λ_odd = 1.49·10^10`, and its seed-1 base `ord(g) = 4.98·10^9`, which
  would need ≈ 160 GB here (§5).

So N_W is reachable on one VM *because* `λ(N_W)` has a small odd part (`p − 1` has the factor 2^7
and both `p − 1` and `q − 1` are 677-smooth), while a state-vector simulator pays for all 2^40
amplitudes regardless. The same property makes N_W classically weak: Pollard's `p − 1` method with
smoothness bound 53 returns `712 321` immediately (checked; and any 39-bit N falls to trial division
in under a second). **Nothing here is a classical factoring speed-up, and no timing below is
comparable to Willsch et al.'s: the two simulations do different amounts of work for different
reasons.**

## 2. Method

### 2.1 Circuit and base rule (fixed in advance)

* **Circuit.** Ekerå–Håstad short discrete logarithm (`y = g^{(N−1)/2} = g^d`, `d = (p + q − 2)/2`),
  `m = ⌈39/2⌉ = 20`, registers of `m` and `2m` bits (60 exponent bits instead of Shor's 78), each
  with its own semiclassical QFT; the oracle is the Gidney–Ekerå windowed multiplier of
  [ge-shor.md](ge-shor.md) built from X / CNOT / Toffoli gates plus measurement-based uncomputation of
  the lookups (`lookups`: temporary-AND unary iteration, X-basis measurement of the lookup register
  with Z / CZ phase fix-ups), exact modular arithmetic (no coset approximation). Two window
  configurations, both run on the same base:
  * **A** = `(w_e, w_m) = (2, 3)`: the configuration of the 31-bit EH run of [ge-shor.md](ge-shor.md)
    (166 qubits; fewest Toffolis of the two);
  * **B** = `(1, 4)`: one exponent qubit per window (165 qubits): 2 instead of 4 evaluated branches
    per stored value, so half the memory, and 31 % fewer gate·branch operations per exponent bit
    (`2·25.3 k` vs `2·36.6 k` slice steps per bit at 39 bits), at 50 % more Toffolis.
* **Base rule** (the rule of the 31-bit EH run): `h` = first draw of
  `StdRng::seed_from_u64(seed).random_range(2..N−1)`, `g = h^(2^n) mod N` with `n` = bits of `N − 1`;
  the measurement outcomes come from the same RNG stream. Nothing in the simulation reads `p`, `q`,
  `λ(N)` or `ord(g)`. Seeds in order 1, 2, … until a run factors N, all runs reported; for N_W seed 1
  was the first and only seed run. Seed 1: `h = 535 596 708 274`, `g = 345 241 646 758`.
* **Classical post-processing.** EH's 2-D lattice reduction and enumeration (`eh_postprocess`); every
  candidate is verified by `g^{d'} = y` and `p·q = N`.
* **Shor's order finding on the same g** (`shor-odd`, 78 exponent bits, config B): finds `r = ord(g)`,
  odd, so `a^{r/2}` is not available; but `2^n·r` is a multiple of `ord(h)`, and Miller's reduction
  (square `h^r` until it reaches 1; a square root of 1 other than ±1 gives `gcd(·−1, N)`)
  splits N unless `ord_p(h)` and `ord_q(h)` have the same 2-adic valuation
  (`shor::ge::factor_from_power_order`, unit-tested on every base of two small N).

### 2.2 Engine changes on this branch

1. **AVX-512 tier** of the bit-sliced evaluator (`SliceIsa`, `src/shor/sliced.rs`). A compiled block
   is a list of steps `w[t] ^= w[a] & w[b]` on slice words of `64·L` branches (X and CNOT address an
   all-ones word, the measured-uncompute reset is `[q, q, q]`, phase fix-ups address a sign word).
   The new kernel does each step per 512-bit word with one `VPTERNLOGQ` (truth table `0x78` =
   `t ^ (a & b)`); for a CNOT the third operand is the all-ones word, so the instruction computes
   `t ^ a` (a dedicated `VPXORQ` path that would skip loading the all-ones word was not
   implemented). Runtime
   dispatch: AVX-512F and `L % 8 == 0` → AVX-512, else AVX2, else portable; `QSIM_NO_AVX512` /
   `QSIM_NO_AVX2` force a tier off. The per-thread slice buffers are now 64-byte aligned
   (`SliceBuf`), so a 512-bit load never straddles two cache lines. `QSIM_SLICE_LANES` also accepts 64.
2. **In-place window finish.** After a window's measurements the new state was written in place over
   the `e = 0` run when it fit, but as soon as the support grew inside a window every later output of
   that chunk went to an overflow buffer and the whole state was then copied into a fresh vector: a
   transient of two extra copies of the new state. Now outputs that would overtake the read pointer
   wait in a per-chunk FIFO and are written back as soon as more of the run has been read; the chunks
   are then moved together inside the window's own array (right-moving chunks last-first, left-moving
   first-last, so no move overwrites an unmoved part). Peak memory is the window array plus the growth
   of the support in that window.

### 2.3 Checks

* **Full-size oracle check, independent interpreter** ([`oracle_check.py`](../data/shor-xl/oracle_check.py),
  numpy bit operations on 3-limb basis states, written from scratch): `ge_shor dump` prints a resolved
  window block exactly as the run uses it (same measurement-outcome stream); the script applies every
  X / CNOT / Toffoli / Z / CZ / X-measurement to ≈ 2 000 random valid inputs `|e⟩|x⟩|0…⟩` plus the edge
  cases `x ∈ {0, 1, N − 1}` and checks `e` unchanged, `x → g^e x mod N`, every other qubit 0 and one
  common sign on every branch (the measurement phases cancel).
* **Small-N distributions** (`tests/shor/ge_shor.rs::odd_order_bases_match_textbook`): for
  N = 35, 77, 143 and up to four distinct odd-order bases `g = h^(2^n) ≠ 1` each, both
  configurations: the gate-level EH distribution of `(j, k)` equals the textbook EH distribution
  (brute force over all exponent pairs) to 1e-12; Shor's order finding on `g` equals the 3n-qubit
  full-QFT distribution to 1e-12 (N = 35, 77); and for N = 35 the EH distribution also equals a
  gate-by-gate quantum reference (sparse state vector, real H / Phase / projective measurements,
  every X-measurement as H + projection with P = 1/2 asserted) to 1e-10.
* **Engine changes**: differential tests of every tier on random programs (all step kinds, aliased
  indices, sign word, L = 4…32, aligned and misaligned buffers) and on real oracle blocks (MBU
  controlled-U at 20 and 31 bits, both N_W window blocks; outputs also checked to be the right
  products) (`src/shor/sliced.rs`); the in-place finish against the out-of-place `materialize` on every
  window of sampled Shor and EH runs at 20 and 24 bits, including ≥ 20 windows where the support grows
  (`src/shor/ge.rs::finish_in_place_equals_materialize`); and every run itself asserts after every
  window that the exponent qubits are unchanged, every ancilla is 0 on every branch, every branch has
  the program's global sign, and no two branches collide.

## 3. N_W, gate by gate

### 3.1 Runs (seed 1, the only seed run)

Logs: [`nw_B_seed1.log`](../data/shor-xl/nw_B_seed1.log), [`nw_A_seed1.log`](../data/shor-xl/nw_A_seed1.log),
[`nw_shorodd_B_seed1.log`](../data/shor-xl/nw_shorodd_B_seed1.log) (first build: AVX2 tier, old window
finish) and [`nw_B_seed1_v2.log`](../data/shor-xl/nw_B_seed1_v2.log) (final build). "Ops" counts
every gate, X-basis measurement and phase fix-up once; "steps" are bit-sliced engine steps.

| circuit | qubits | exponent bits / windows | ops | Toffolis | X-meas. | steps | peak support | peak branches | gate·branch ops | time (s) | peak RSS | 1-min load before → after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EH, config B (`w_e = 1, w_m = 4`), final build | 165 | 60 / 60 | 1 480 218 | 351 697 | 82 033 | 1 520 967 | 71 582 595 | 143 165 190 | 1.209·10^14 | **99.6** | **2.26 GB** | 21 → 27 |
| same circuit, first build | 165 | 60 / 60 | 1 480 218 | 351 697 | 82 033 | 1 520 967 | 71 582 595 | 143 165 190 | 1.209·10^14 | 143.9 | 4.59 GB | 15 → 38 |
| EH, config A (`w_e = 2, w_m = 3`), first build | 166 | 60 / 30 | 1 069 881 | 235 280 | 56 258 | 1 098 077 | 71 582 595 | 286 330 380 | 1.716·10^14 | 178.3 | 4.73 GB | 19 → 48 |
| Shor on g (`shor-odd`, config B), first build | 165 | 78 / 78 | 1 929 347 | 458 420 | 106 562 | 1 982 122 | 71 582 595 | 143 165 190 | 1.869·10^14 | 211.1 | 4.50 GB | 47 → 56 |

All times are wall-clock on 16 threads with the machine loaded by other users (the house rule
"no timing claims above a load of 16" is not met by any of them; they are reported as loaded).
Breakdown of the final-build run: gate evaluation 33.9 s, sort 49.1 s, window probabilities 5.9 s,
new state 10.2 s.

**Outcomes.** Both EH circuits measured the same pair `j = 1 046 455 110 769` (40 bits),
`k = 358 190` (20 bits) from the same RNG stream; `eh_postprocess` found `d = 742 050`, verified
`g^d = y`, and `p + q = 2d + 2 = 1 484 102`, `pq = N` gave **712 321 × 771 781**. `shor-odd`
measured `y = 56 421 831 118 435 002 971 950` (78 bits), the convergents gave
`ord(g) = 71 582 595`, and Miller's reduction from `h^{ord(g)}` gave the factor **771 781**.

**Support trajectory** (config B, `QSIM_GE_PROFILE`): the 20 rounds of the `m`-bit register double
the support to 2^20 (`ord(y) = λ_odd/15 = 4.77·10^6 > 2^20`, so no wrap-around); the first rounds
of the `2m`-bit register take it to 2^21, 2^22, 2^23, then 1.61·10^7, 1.88·10^7, 3.76·10^7,
7.09·10^7 and `ord(g) = 71 582 595` after 28 windows; the remaining 32 windows each evaluate
1.43·10^8 branches. The final window's state is not materialised (the measured integers are
complete).

### 3.2 Plain Shor with the random base h: not run

With the seed-1 base itself, `ord(h) = 9 162 572 160 = λ(N_W)` (`ν₂ = 7`). Shor's support grows to
`ord(h)/2 = 4.58·10^9` in the last rounds, and the last window evaluates `ord(h) = 9.16·10^9`
branches: 147 GB at 16 B per branch (f32), 12× the 12 GB cap on this machine. This is not a
property of seed 1: `p − 1` has 2-adic valuation 7, so a uniformly random `h` has
`ν₂(ord(h)) = 7` with probability 1/2 and `≥ 4` with probability 15/16, i.e. a peak of
`≥ 2^4·r_odd(h)` branches, ≥ 18 GB at the full `r_odd`. The odd-order base is what makes N_W
reachable; it does not need the factors.

### 3.3 Checks at full size and at small N

* **Independent oracle check** ([`oracle_check_nw.out`](../data/shor-xl/oracle_check_nw.out)): 11
  resolved window blocks of the three runs (config B windows 0, 19, 20, 40, 59; config A windows 0,
  9, 10, 29; `shor-odd` windows 0, 77; 23 815–36 056 operations each, every operation kind present),
  2 006–2 012 inputs each: **all correct** (`x → g^e x mod N`, `e` unchanged, every other qubit 0,
  one common sign).
* **Run-time assertions**: in every window of every run, on every branch, the exponent qubits came
  back unchanged, every ancilla was 0 and every branch carried the program's global sign; no two
  outputs collided (`run`/`window` assertions in `src/shor/ge.rs`).
* **Small N** (`odd_order_bases_match_textbook`, [`tests_small_n.log`](../data/shor-xl/tests_small_n.log)):
  SMALLN_RESULT. The exact probability that Shor-on-`g` plus Miller's reduction factors N is 1 or 0
  per base at these sizes (0 exactly when `ν₂(ord_p h) = ν₂(ord_q h)`, as the reduction predicts).
* This test first **failed** on the sparse reference: an X-measurement had `P(m) = 0.4999999986`
  instead of 1/2. The cause was in `SparseState::collapse` (not in the engine under test): it took
  `P(0) = 1 − P(1)` and rescaled by it, so a norm error doubled at every outcome-0 collapse of a
  `P = 1/2` qubit, and long measurement-based-uncomputation circuits drifted. `collapse` now rescales
  by the kept weight itself; regression test `sparse_collapse_keeps_the_state_normalised`
  (`tests/shor/shor_scale.rs`) fails on the old code, whose error doubles every round.

## 4. AVX-512 for the bit-sliced evaluator

**Kernel** (`ge_shor slicebench`, one N_W window block, random valid inputs, one thread pinned with
`taskset`, min of 3 × 20 evaluations; [`slicebench.log`](../data/shor-xl/slicebench.log)):

| L (64·L branches per batch) | portable | AVX2 | AVX-512 (`VPTERNLOGQ`) | AVX-512 / AVX2 |
|---|---|---|---|---|
| 8 | 0.383 ns | 0.267 ns | 0.203 ns | 1.32× |
| 16 (default) | 0.338 ns | 0.212 ns | 0.138 ns | 1.54× |
| 32 | 0.309 ns | 0.175 ns | 0.119 ns | 1.47× |
| 64 | 0.356 ns | 0.194 ns | 0.106 ns | 1.83× |

(ns per step per 64-branch word, config B window 40, 24 675 steps; config A window 25 gives
0.144 / 0.211 ns at L = 16. All tiers use the new 64-byte-aligned buffers.) At L = 16 one AVX-512
thread does 4.6·10^11 gate·branch operations per second.

**Whole runs** (interleaved A/B under the bench lock, 3 repetitions each, 16 threads; A =
`QSIM_NO_AVX512=1`, B = default; same binary; [`ab_28.log`](../data/shor-xl/ab_28.log),
[`ab_31_qsim.log`](../data/shor-xl/ab_31_qsim.log), [`ab_31_eh.log`](../data/shor-xl/ab_31_eh.log)):

| run | AVX2 tier: wall (s) | AVX-512 tier: wall (s) | min/min | gate evaluation, min (s) | 1-min load before runs |
|---|---|---|---|---|---|
| eh-odd-28bit-A | 7.67 / 7.62 / 7.96 | 7.29 / 7.49 / 7.37 | 1.045 | 2.35 → 1.96 (1.20×) | 32–34 |
| qsim-28bit-mbu-lookup-f64 | 13.41 / 13.61 / 13.61 | 12.61 / 13.36 / 13.04 | 1.063 | 4.49 → 3.79 (1.18×) | 25–32 |
| qsim-31bit-record-mbu-lookup-f32 | 80.68 / 83.71 / 83.71 | 76.42 / 77.84 / 77.23 | 1.056 | 30.52 → 25.95 (1.18×) | 23–29 |
| eh-odd-31bit-A | 75.03 / 76.58 / 76.93 | 68.78 / 69.15 / 71.37 | 1.091 | 25.54 → 19.37 (1.32×) | 22–38 |


The 31-bit qsim row is the repo's record command
(`qsim run shor --modulus 1537596787 --semiclassical --sliced --window 4 --oracle windowed-mbu-lookup --f32 --seed 2 --tries 1`):
**re-timed here at 76.4 s with AVX-512 (80.7 s with AVX2)**, same measured integer
2 059 039 373 337 077 151, `r = 256 252 500`, factor 52 501, 4.3–4.8 GB, at a load of 23–29
(the Mac's 89.3 s in [mbu-shor.md](mbu-shor.md) was at a load of 10–17). Every run of every row
measured the same integer / pair as on the Mac. The machine was loaded (22–38) throughout, so the
ratios carry an uncertainty of a few percent; the gate-evaluation ratio is the cleaner number.

**Why the end-to-end gain is small.** The kernel is 1.5× faster but gate evaluation is now only
28–40 % of a run on this machine: sorting the evaluated branches (`par_sort_unstable` of 16-byte
entries) takes 29–55 % and the merge/collapse most of the rest (31-bit qsim: eval 26 s, sort 23 s,
collapse 24 s). With 16 threads on 8 cores the two hyper-threads of a core share its vector units,
which also caps the kernel gain inside runs (1.18–1.32× vs 1.5× single-thread).

## 5. Generic N: the frontier on this machine

### 5.1 Memory model

Between windows a branch is an 8-byte key (the work register; `n ≤ 62` bits) and an amplitude (8 B
in f32, 16 B in f64); the ancilla words exist only per batch of 64·L branches in the evaluator. A
window appends `2^{w_e} − 1` evaluated copies of the support to the state's own buffer and sorts
them in place; the finish now writes the new state into the same buffer. So

    peak RSS ≈ 16 B × 2^{w_e} × max_k |S_k|  (f32; 24 B in f64),  |S_k| ≤ ord(g) for EH-odd.

Measured (final build, f32): N_W 16.2 B, 43-bit 16.0 B, 30-bit 16.5 B, 33-bit 16.1 B per peak branch
(config B), 31-bit config A 16.0 B. Before the in-place finish, windows in which the support grows
kept the window array, an overflow buffer and a fresh copy of the new state at once: 32.8 B per
branch for N_W config B (4.59 GB). Choosing `w_e = 1` halves the branches per stored value (2
instead of 4), so N_W needs 2.3 GB instead of the 4.7 GB of config A, and the 43-bit run 3.7 GB.

### 5.2 Instances

Generator: [`gen_generic.py`](../data/shor-xl/gen_generic.py) (the round-4 generator, loop extended
to 63 bits); orders: [`orders_seed1.txt`](../data/shor-xl/orders_seed1.txt). Runs: EH-odd, seed 1,
config B, f32, final build, under the bench lock ([`frontier_22_33.log`](../data/shor-xl/frontier_22_33.log),
[`g43_B_seed1.log`](../data/shor-xl/g43_B_seed1.log)).

| bits | N | ord(g) | λ_odd | qubits | Toffolis | peak branches | gate·branch ops | time (s) | peak RSS (GB) | load before/after | factored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 2306833 | 143983 | 143983 | 97 | 64760 | 287966 | 3.62e+10 | 0.3 | 0.01 | 19.35/19.35 | run 2 |
| 23 | 4297567 | 268321 | 268321 | 101 | 73571 | 536642 | 8.28e+10 | 0.3 | 0.01 | 19.35/19.35 | run 1 |
| 24 | 10161323 | 79335 | 79335 | 105 | 78170 | 158670 | 2.80e+10 | 0.1 | 0.01 | 19.35/19.35 | run 1 |
| 25 | 18942389 | 9861 | 147915 | 109 | 102452 | 19722 | 5.33e+09 | 0.1 | 0.01 | 19.35/19.35 | run 1 |
| 26 | 43584217 | 10892749 | 10892749 | 113 | 106422 | 21785498 | 3.54e+12 | 7.0 | 0.36 | 19.35/20.36 | run 1 |
| 27 | 82337219 | 21437 | 321555 | 117 | 118568 | 42874 | 1.35e+10 | 0.1 | 0.01 | 20.36/20.36 | run 1 |
| 28 | 221643407 | 6925425 | 6925425 | 121 | 123909 | 13850850 | 3.28e+12 | 6.4 | 0.24 | 20.36/19.99 | run 1 |
| 29 | 282304153 | 23522415 | 23522415 | 125 | 156021 | 47044830 | 1.37e+13 | 20.2 | 0.76 | 19.99/23.39 | run 1 |
| 30 | 869985671 | 108740799 | 108740799 | 129 | 161835 | 217481598 | 6.06e+13 | 75.1 | 3.59 | 23.39/26.47 | run 1 |
| 31 | 1537596787 | 21354375 | 64063125 | 133 | 178913 | 42708750 | 1.59e+13 | 21.2 | 0.69 | 26.47/27.74 | run 1 |
| 32 | 3631204201 | 3504907 | 907770913 | 137 | 185763 | 7009814 | 3.00e+12 | 4.9 | 0.12 | 27.74/28.41 | run 1 |
| 33 | 5135420869 | 53492289 | 53492289 | 141 | 228178 | 106984578 | 4.78e+13 | 54.3 | 1.72 | 28.41/24.31 | run 1 |
| 43 | 4911456443897 | 115574445 | 9592678935 | 181 | 470393 | 231148890 | 2.65e+14 | 190.1 | 3.71 | 32.30/29.24 | run 1 |

All 13 instances that fit were run and all factored (22 bits on the second run of the same base,
the rest on the first). Instances not run (window memory alone above the 10 GB the rule allows):

| bits | N | λ_odd | ord(g), seed-1 base | window memory needed (2·ord(g)·16 B) |
|---|---|---|---|---|
| 34 | 13311882539 | 1.664e+09 | 1.664e+09 | 53 GB |
| 35 | 21941113489 | 1.371e+09 | 1.371e+09 | 44 GB |
| 36 | 42725174083 | 4.450e+08 | 4.450e+08 | 14 GB |
| 37 | 97395687533 | 2.435e+10 | 2.435e+10 | 779 GB |
| 38 | 206049643663 | 2.576e+10 | 2.576e+10 | 824 GB |
| 39 | 478046366261 | 1.494e+10 | 4.980e+09 | 159 GB |
| 40 | 706008776849 | 1.765e+11 | 1.765e+11 | 5,648 GB |
| 41 | 1758781661737 | 1.466e+11 | 4.885e+10 | 1,563 GB |
| 42 | 2252715200279 | 2.816e+11 | 2.816e+11 | 9,011 GB |
| 44 | 15519334259309 | 3.880e+12 | 3.880e+12 | 124,155 GB |
| 45 | 20814987273719 | 2.098e+10 | 6.994e+09 | 224 GB |
| 46 | 63448748484379 | 3.966e+12 | 7.931e+11 | 25,379 GB |
| 47 | 76698860661583 | 1.598e+12 | 5.326e+11 | 17,044 GB |
| 48 | 181391910836177 | 4.535e+13 | 4.319e+11 | 13,820 GB |
| 49 | 299435908660573 | 2.495e+13 | 8.318e+12 | 266,165 GB |
| 50 | 565374444486823 | 2.356e+13 | 2.356e+13 | 753,833 GB |
| 51 | 1255966957231829 | 3.140e+14 | 3.140e+14 | 10,047,735 GB |
| 52 | 3857653709797969 | 9.644e+14 | 9.644e+14 | 30,861,229 GB |
| 53 | 5660217289348339 | 2.358e+14 | 4.717e+13 | 1,509,391 GB |
| 54 | 9792182088976349 | 3.060e+14 | 3.060e+14 | 9,792,182 GB |
| 55 | 22146511668858779 | 2.768e+15 | 9.228e+14 | 29,528,682 GB |
| 56 | 54913808377912427 | 3.432e+15 | 1.040e+14 | 3,328,110 GB |
| 57 | 107759707670444509 | 1.122e+15 | 1.122e+15 | 35,919,902 GB |
| 58 | 147832693759633801 | 3.080e+15 | 3.080e+15 | 98,555,129 GB |
| 59 | 346860792455719253 | 1.355e+15 | 1.355e+15 | 43,357,599 GB |
| 60 | 659878970474292769 | 1.790e+13 | 8.524e+11 | 27,277 GB |
| 61 | 1191368052034775239 | 2.482e+16 | 2.482e+16 | 794,245,366 GB |
| 62 | 2376476597879466283 | 3.301e+16 | 3.301e+16 | 1,056,211,820 GB |
| 63 | 6513859281536866367 | 8.142e+17 | 8.142e+17 | 26,055,437,104 GB |

**Support-size scaling.** Time follows the work counter, not N: the 32-bit N (`ord(g) = 3.5·10^6`)
takes 4.9 s, the 30-bit N (`ord(g) = 1.09·10^8`) 75 s; across the runs above 1 s, time / (gate·branch
operations) is 0.7–2.0 ns per 10^3 operations at the loads shown (the 43-bit run, with the most
windows at full support, 0.72). `ord(g)` itself is `λ_odd` divided by whatever factors the base
misses: 1 for most instances, but 1/3 (31 bits), 1/15 (25, 27), 1/83 (43), 1/259 (32). For a
generic semiprime `λ_odd ≈ N/2^{ν}/small`, so the support, and with it time and memory, grows
exponentially in the bit length, exactly as in [shor.md](shor.md) ("What is exponential in what"):
the **contiguous frontier** is 33 bits (every generator N up to 33 bits fits with its seed-1 base,
the 34-bit one needs 53 GB), and the first instance that would fit with any base of full odd order
above 33 bits is none up to 63 bits.

### 5.3 The 43-bit run

`N = 4 911 456 443 897 = 1 456 057 × 3 373 121` (the generator's first balanced 43-bit
semiprime); seed 1: `h = 4 784 960 592 739`, `g = h^(2^43) = 3 089 596 228 147`; `m = 22`, 66
exponent bits, config B: 181 qubits, 66 windows, 1 894 354 operations (470 393 Toffolis,
105 583 X-measurements, 32 702 phase fix-ups), peak support `ord(g) = 115 574 445`, peak branches
231 148 890, 2.646·10^14 gate·branch operations; one run measured `j = 691 091 360 860`,
`k = 1 406 766`, and the post-processing returned 1 456 057 × 3 373 121. **190.0 s on 16 threads at
load 29–32 (loaded), peak RSS 3.62 GB** (gate evaluation 69.7 s, sort 91.3 s, probabilities 10.2 s,
new state 18.3 s). A first attempt was killed by the kernel's OOM killer after 53 s when another
job on the machine took the free memory to 0.4 GB (`g43_B_seed1_oomkilled.log`); the rerun is the
one reported.

Why this instance and not another: `λ = 2^6·3^5·5·7·83·107·127`, `λ_odd = 9 592 678 935`; the
seed-1 base `h` has order `λ/(2^3·83)`, so `g = h^(2^43)` has order `λ_odd/83`. For a uniformly random
base the odd order is at most `λ_odd/83` with probability 5.1 % and equals `λ_odd` with probability
44 % (exact, from the factorisations of `p − 1` and `q − 1`); a base of full odd order would need
2·9.59·10^9·16 B ≈ 307 GB here.
Both `p − 1 = 2^3·3^5·7·107` and `q − 1 = 2^6·5·83·127` are 243-smooth, so this N, like N_W, is
classically weak to Pollard's `p − 1`; the selection rule (largest instance whose support fits)
favours such N, because a small support needs a small `λ_odd`.

## 6. Against Willsch et al. 2023, honestly

| | Willsch et al. 2023 (`shorgpu`) | this notebook |
|---|---|---|
| largest N | 549 755 813 701 (39 bits) | the same N; and 4 911 456 443 897 (43 bits) |
| algorithm | iterative Shor (one recycled control), Shor's and Ekerå's post-processing | Ekerå–Håstad (also Shor's order finding on an odd-order base + Miller) |
| oracle | controlled modular multiplication applied as a permutation of the amplitudes | X / CNOT / Toffoli circuit with measurement-based uncomputation, every gate on every branch |
| state | full 40-qubit state vector: 2^40 complex doubles (16 TiB; > 40 TiB with buffers) | the branches of the exact 165–181-qubit state: ≤ 2^{w_e}·ord(g) (1.4·10^8 / 2.3·10^8) |
| cost set by | 2^40, whatever the base's order | the support `ord(g)`, a divisor of `λ_odd(N)` |
| hardware | up to 2048 A100 GPUs (JUWELS Booster) | one 16-vCPU VM, 2.3 / 3.6 GB |
| base | random `a` coprime to N | `h` from a seeded RNG, `g = h^(2^n)` (odd order); no factors used in the run |

Their simulation would cost the same for any 39-bit N and any base; ours would not run at all for a
typical 39-bit N (§5.2: 159 GB for the generator's 39-bit N) or for N_W with a typical *Shor* base
(§3.2: 147 GB). The comparison shows that this N happens to have a small `λ_odd`, not that one
simulator is faster than the other. It is also no statement about factoring: both N are classically
easy (trial division, and Pollard's `p − 1` in particular).

## 7. Caveats and negative results

* **Selection.** N_W was chosen because Willsch et al. factored it; its smooth `p − 1` (odd part of
  `λ` only 7.2·10^7) is what makes it reachable here. The 43-bit N was selected, after computing the
  supports of all 22–63-bit generator instances from their factors, as the only instance above 33
  bits whose seed-1 run fits; the seed-1 base happens to miss the factor 83 of `λ_odd` (probability
  ≈ 5 % for a random base). These orders were computed before the rule in `PREREG.md` was written
  (stated there); the circuit runs themselves never read the factors or the orders.
* **Loaded machine.** Every timing in this notebook was taken at a 1-min load of 15–56 from other
  users and agents (16 vCPUs), so absolute times are pessimistic and ratios carry a few percent of
  noise. The house rule asks for re-timing on an idle machine; no idle window occurred.
* **Memory incident.** The first 43-bit attempt was killed by the kernel's OOM killer when another
  job drove the machine's free memory to 0.4 GB (our process had 3.6 GB); the rerun is reported.
* **AVX-512 gains less than the kernel suggests**: 1.5× per thread on the kernel, 1.18–1.32× on gate
  evaluation inside 16-thread runs, 4–9 % end to end. Sorting the evaluated branches is now the
  largest cost (29–55 %); an in-place parallel radix sort is the obvious next lever and was not
  attempted. Larger slices help the AVX-512 kernel (L = 64: 0.106 ns vs 0.138 ns at the default 16)
  in the kernel benchmark; in whole 31-bit EH runs at a load of 56–60 ([`lanes_31_eh.log`](../data/shor-xl/lanes_31_eh.log))
  the gate evaluation took 17.9–23.1 s at L = 64, 21.7–30.1 s at L = 32 and 28.7 s at L = 16 — too
  noisy to justify changing the default (16); `QSIM_SLICE_LANES=64` is available.
* **No coset arithmetic** at these sizes (as in [ge-shor.md](ge-shor.md) §7: it multiplies the support
  by 2^{2c}); the circuits are exact.
* **f32 amplitudes** (precision checked in [shor.md](shor.md): TV ≤ 1.5·10^−7 at small N); the
  measured pairs are those of the f64 distribution only up to this rounding.
* **Frontier is memory, not time**: with 10 GB for the window, `2·ord(g)·16 B` caps `ord(g)` at
  ≈ 3·10^8; the next generator instance (36 bits, `ord(g) = 4.45·10^8`) needs 14 GB. A 12-byte
  branch would not change that conclusion.
* **Bug found on the way** (fixed, with a regression test): `SparseState::collapse` amplified norm
  errors (§3.3); it only affected the gate-by-gate reference simulator, not the sliced engine.

## 8. Reproduction

Scripts and every log are in [`research/data/shor-xl/`](../data/shor-xl/) (its README lists them).

```sh
cargo build --release --example ge_shor --bin qsim
G=target/release/examples/ge_shor
$G run 549755813701 1 1 4 lookups eh-odd f32 3      # N_W (EH, config B): 2.3 GB
$G run 549755813701 1 2 3 lookups eh-odd f32 3      # N_W (EH, config A): 4.7 GB
$G run 549755813701 1 1 4 lookups shor-odd f32 3    # N_W (Shor on g, Miller)
$G run 4911456443897 1 1 4 lookups eh-odd f32 3     # 43-bit generator N: 3.7 GB
research/data/shor-xl/frontier.sh $G 22 23 24 25 26 27 28 29 30 31 32 33
python3 research/data/shor-xl/oracle_check.py $G 549755813701 1 1 4 lookups eh-odd 40
$G slicebench 549755813701 1 1 4 lookups eh-odd 40 20   # kernel per tier and lane count
QSIM_NO_AVX512=1 $G run 1537596787 2 2 3 lookups eh-odd f32   # AVX2 tier (A/B)
cargo test --release --lib shor::sliced shor::ge
cargo test --release --test ge_shor --test shor_scale
python3 research/data/shor-xl/orders.py $G 1 22 63     # supports (uses the factors)
python3 research/data/shor-xl/tables.py                # tables of this notebook
```

