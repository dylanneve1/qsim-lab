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

TBD-HEADLINE

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
   `t ^ (a & b)`), so a CNOT is effectively a `VPXORQ` with the all-ones word folded in. Runtime
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
  N = 35, 77, 143 and every distinct odd-order base `g = h^(2^n) ≠ 1` from the first 40 bases, both
  configurations: the gate-level EH distribution of `(j, k)` equals the textbook EH distribution
  (brute force over all exponent pairs) to 1e-12, Shor's order finding on `g` equals the 3n-qubit
  full-QFT distribution to 1e-12, and for N = 35, 77 the EH distribution also equals a gate-by-gate
  quantum reference (sparse state vector, real H / Phase / projective measurements, every
  X-measurement as H + projection with P = 1/2 asserted) to 1e-10.
* **Engine changes**: differential tests of every tier on random programs (all step kinds, aliased
  indices, sign word, L = 4…32, aligned and misaligned buffers) and on real oracle blocks (MBU
  controlled-U at 20 and 31 bits, both N_W window blocks; outputs also checked to be the right
  products) (`src/shor/sliced.rs`); the in-place finish against the out-of-place `materialize` on every
  window of sampled Shor and EH runs at 20 and 24 bits, including ≥ 20 windows where the support grows
  (`src/shor/ge.rs::finish_in_place_equals_materialize`); and every run itself asserts after every
  window that the exponent qubits are unchanged, every ancilla is 0 on every branch, every branch has
  the program's global sign, and no two branches collide.

TBD-SECTIONS
