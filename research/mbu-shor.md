# Measurement-based uncomputation in the exact gate-level Shor simulation (exp/mbu-shor)

Branch `exp/mbu-shor`, based on `d44563c` (main + exp/superopt). Code:
`src/shor_mbu.rs` (the measurement-based blocks, `MbuOpts`, logical-op IR,
outcome resolution), `src/shor/sliced.rs` (sign word, `compile_ops`,
`oracle_ops`, the sign assertion), `src/shor.rs` (`Oracle::WindowedMbu`,
`Oracle::WindowedMbuLookup`, the gate-by-gate path with real X-basis
measurements), CLI `--oracle windowed-mbu | windowed-mbu-lookup`,
`examples/mbu_counts.rs`, `examples/shor_seed_orders.rs`, tests in
`src/shor_mbu.rs`, `src/shor/sliced.rs`, `tests/mbu_shor.rs`,
`tests/theory_shor_mbu.rs`. Raw data: `research/data/mbu-shor/`.

**Machines.** Builds, tests, counts and timings ran on the Mac (M1 Pro,
8 cores, 16 GB). Timings were taken under `/tmp/qsim-mac-bench.lock`, one
lock per 31-bit run and one per 24-bit block or 28-bit triple, with 65 s
gaps, interleaving windowed-opt / mbu-lookup / mbu. The 1-min load average
was 10–17 during the timings, from other agents' unlocked work. The VPS
was used for git, `cargo fmt` and `cargo clippy --all-targets -D warnings`
(clean).

## TL;DR

* The sliced engine now runs **X-basis measurements with classically
  controlled Z / CZ fix-ups exactly**. A measured ancilla that is a
  deterministic function of the other qubits on every branch gives a
  uniformly random outcome `m`. For `m = 1` every branch picks up a sign
  `(−1)^{f(x)}`. The engine keeps one extra slice word for the per-branch
  sign (`sign ^= w[q]` for Z or a measurement, `sign ^= w[a] & w[b]` for
  CZ, `w[q] = 0` for the reset). At the end of every block it **asserts**
  that every branch has sign +1. This sits next to the existing assertions
  (ancillas clean, control unchanged, block injective on the support). The
  injectivity check is what fails if a measured qubit was *not* a function
  of the rest, because two branches would then collide.
* Three measurement-based constructions are built on this, all exact:
  1. **temporary logical-AND** (Gidney 2017) in every lookup's unary
     iteration: `2^w − 1` Toffolis per lookup instead of `2(2^w − 1)`;
  2. **measurement-based unlookup** (Berry et al. 2019 / Gidney 2019): the
     lookup register is X-measured, then a *phase lookup* of `g(v) = m·T[v]`
     runs. It uses unary iteration over the high address bits and the
     algebraic normal form over the low bits, with the split chosen per
     table: 4 Toffolis for `w = 4` instead of 15–28;
  3. **measured modular-adder flag** (the flag trick of Gidney 2025, Sec. 2):
     the flag `t = [b_new < L]` is X-measured instead of
     comparator-uncomputed. The fix-up is a *phase* comparator, needed only
     on outcome 1 (half the shots).

  With all three plus **Gidney temporary-AND adders** in the modular adder
  (`Oracle::WindowedMbu`, `n − 1` extra qubits), the 31-bit record circuit
  goes from **261 454 to 119 096 Toffolis (−54.4 %)**. Its gate count
  (measurements and fix-ups included) rises from 1 039 303 to 1 180 796
  (+13.6 %). Without the Gidney adders (`Oracle::WindowedMbuLookup`, same
  132 qubits) Toffolis fall 16.5 % and gates 16.1 %.
* **Simulator cost is set by slice steps, not Toffolis.** One CCX is one
  step. A measured uncompute is 1 step (outcome 0) or 3 steps (outcome 1,
  sign + reset + CZ). The Gidney adder has about 1.5× the CNOTs of
  Cuccaro's. So the full-MBU circuit is **+20.5 % slice steps** at 31 bits
  (+4.7 % at `w = 5`), while the lookup-only variant is **−13.9 %**.
  End to end (Mac, bench lock, interleaved, min of 3), at the 31-bit
  record the lookup-only oracle runs **98.0 s → 89.3 s (−8.9 %)** and the
  full-MBU oracle 112.3 s (+14.6 %). At 28 bits: 12.18 → 11.23 s (−7.8 %) /
  13.48 s (+10.7 %). Every run measured the same integer and factored N.
  The gate-evaluation time tracks the slice-step count to within 2 points:
  −11.9 % / +20.1 % measured at 31 bits vs −13.9 % / +20.5 % predicted.
* **32 bits.** The first seeded 32-bit generic N, 3 631 204 201 = 58 907 ×
  61 643, is factored at gate level: 136 qubits, 0.95 M gates, exact
  peak support 1.30·10⁸, 6.3 GB, 498 s on 2 background threads. The base
  was chosen by seed so its order fits in RAM, and that choice used λ(N);
  see the caveat in §5b.
* **Against the literature.** Halving Toffolis with MBU is the known
  Gidney/Babbush/Berry result, now reproduced exactly at gate level. That
  part is not new. Our remaining 4.0 n³ Toffolis per run at n = 31
  (119 096 / 31³) compare with Gidney–Ekerå 2019's 0.3 n³ + 0.0005 n³ lg n.
  The gap of about 13× is accounted for by exponent windowing, the
  Ekerå–Håstad short exponent and coset (reduction-free) addition (§6).
  None of these is an uncomputation trick, and none is in this exact,
  non-approximate oracle.

## 1. Engine extension (exact)

**Representation.** A round's block is now a list of resolved ops
`MbuOp::{G(gate), MeasX(q, m)}` with gates from {X, CNOT, CCX, SWAP, Z, CZ}.
`SlicedProgram::compile_ops` maps them to the engine's `w[t] ^= w[a] & w[b]`
steps using one extra word `sign = nq + 1`:

| op | slice steps |
|---|---|
| X / CNOT / CCX | `w[t] ^= w[one]·…` (unchanged) |
| Z(q) | `sign ^= w[q] & one` |
| CZ(a, b) | `sign ^= w[a] & w[b]` |
| MeasX(q, m) | if `m`: `sign ^= w[q] & one`; then `w[q] ^= w[q] & w[q]` (= 0) |

The evaluation loop is untouched; reversible programs compile exactly as
before (`signed = false`, no sign check).

**Why this is exact.** Let `q` be a deterministic function `f` of the other
qubits on the support: `ψ = Σ_x α_x |x⟩|f(x)⟩_q`. `H_q` then a measurement
with outcome `m` gives `P(m) = Σ_x |α_x|²/2 = 1/2` and the post-measurement
state `Σ_x α_x (−1)^{m f(x)} |x⟩|m⟩`. After the reset, the branch set is
unchanged and every branch carries a sign. If `f` were not a function, two
branches would differ only in `q`, and the reset would merge them. The
sliced engine would then keep two identical keys. The round's existing
injectivity check (`oracle block is not injective on the support`) and the
control-0 identity check both catch that. The fix-up is a diagonal ±1
phase, again a sign. **The engine asserts that the sign word is zero on
every valid lane of every batch, for both control halves**
("measurement-based uncomputation left a relative sign"). This is stricter
than "a global sign": a −1 shared by all branches would be harmless, but
every fix-up here cancels exactly, and the constructions avoid the
global-sign case (the flag is flipped by an `X` before it is measured, so
its value is `[b < L]`, not `¬[b < L]`).

**Outcomes.** The outcomes are drawn once per block from a SplitMix64
stream seeded by `(N, mult)` (`QSIM_MBU_SEED` salts it;
`QSIM_MBU_OUTCOMES=zero|one` forces all-0 or all-1). Both control halves and
every branch see the same outcomes, as one physical shot would. Because
the post-fix-up state does not depend on `m`, the sampled outcome stream
only changes the circuit's length (which fix-ups run), not its result.
`mbu_block_is_outcome_independent` checks all-0, all-1 and three random
streams.

**Logical-op IR, so the inverse multiplier is exact.** The controlled-U is
`cmult(a)`, then a controlled swap, then `cmult(a⁻¹)⁻¹`. A circuit with
measurements has no gate-reversal inverse. So blocks are built as logical
ops: `And`/`UnAnd` (compute / measured uncompute of a temporary AND),
`Lookup`/`Unlookup` (unlookup = measured), and `FlagCompute`/`FlagUncompute`
(a flag with a reversible `compute` and a phase `fix`). `inverse()` reverses
the list and swaps each pair. This is valid because an uncompute is only
ever reversed into a compute on a clean target, and vice versa. Outcomes
are resolved after inversion.

**Gate-by-gate reference.** For gate-by-gate states (`Backend::Sparse`, dense),
`Instance::round` applies a real `H(q)` for each `MeasX(q, m)`, **asserts
`P(m) = 1/2`**, projects, renormalises and applies `X` if `m = 1`. No
determinism is assumed on this path, so it is an independent quantum
reference.

## 2. The measurement-based blocks (`src/shor_mbu.rs`)

| block | construction | Toffolis (n = 31, w = 4) | proof / test |
|---|---|---|---|
| lookup | unary iteration MSB-first with optimal tree fan-out (exp/superopt). Every internal node is `And(p, bit, f)`, …, `UnAnd(p, bit, f)`; the `UnAnd` is valid because the sibling switch `CNOT(p, f)` is applied twice, so `f = p∧bit` again at the end | `2^w − 1` = 15 (was 30) | `mbu_lookup_unlookup_exhaustive` (w ≤ 5, every address and control, all outcome modes) |
| unlookup | X-measure the n output qubits (outcomes `m`). Phase `(−1)^{ctrl·g(v)}`, `g(v) = popcount(m & T[v]) mod 2`. Fix-up: split the address into `w − k` high bits (unary iteration over them, temporary ANDs, optimal fan-out of the per-leaf ANF masks) and `k` low bits (all ANF monomials of degree ≥ 2 precomputed into the freed output qubits). At a node flag `p`, monomial `s` is `Z(p)` (s = ∅), `CZ(p, x_i)` or `CZ(p, mon_s)`. `k` is chosen per table: min Toffolis, then min ops | `(2^{w−k} − 1) + (2^k − k − 1)` = **4** for w = 4 (k = 2); 0 when `m·T ≡ 0` | `phase_table_all_splits` (every k ≤ w ≤ 5, random g), the unlookup test above |
| temporary-AND adder `b += a` (n + 1-bit b) | Gidney 2017: `CNOT(c,a) CNOT(c,b) And(a,b → c′) CNOT(c,c′)` up, the top carry as a full CCX into `b[n]`, the mirror down with `UnAnd`. Subtraction is the IR inverse | n | `gidney_adder_and_comparator_exhaustive` (n ≤ 5, every a, b, outcome mode) |
| comparator `t ^= [b < a]` | carry-out of `a + ¬b` with the same chain | n | same |
| phase comparator `(−1)^{[b < a]}` | same chain; the top carry kicked back as `CZ(a′, b′)·Z(c)` because `MAJ(a,b,c) = (a⊕c)(b⊕c)⊕c` | n − 1 | same |
| modular adder, Gidney adders | superopt's structure (add L, sub K = N, `t = b[n]`, K-flip, add `t·N`), then `X(t)` and **measured flag**: `t = [b_new < L]`, compute = comparator, fix = phase comparator | 3n + ½(n − 1) expected = **108.5** (superopt: 8n − 1 = 247) | `controlled_ua_exhaustive_small` |
| modular adder, Cuccaro (lookup variant) | superopt's reversible adder up to `X(t)`, with the peephole + SAT-rule passes as in exp/superopt; measured flag with Cuccaro comparator / phase comparator (`Z(borrow)` for `CNOT(borrow, t)`) | 6n − SAT + ½·2n ≈ 212 | same |

**Per-round Toffoli budget at 31 bits** (`counts.txt`, ÷ 62 rounds):
windowed-opt 4217. mbu-lookup 3523: lookups 15 × 15, unlookups ≈ 15 × 4 ×
P(g ≢ 0), modular adders 14 × ≈ 212, controlled swap 31. mbu 1921:
modular adders 14 × ≈ 108.

### Verification (all green on the Mac)

* `cargo test --release --lib shor_mbu` (5 tests). Exhaustive adders,
  comparators and phase comparators (n ≤ 5). Lookups and unlookups (w ≤ 5).
  Every phase-table split. `controlled_ua_exhaustive_small`: N ∈ {15, 21,
  35, 55, 63, 77}, w = 1…4, five option sets (ALL, LOOKUPS, NONE and both
  without the flag trick), three outcome modes, every x < N and both
  controls, checked on the resolved block **including the sign**.
  `controlled_ua_on_sparse_state_with_real_measurements`: random complex
  superpositions over all x and both controls on `SparseState`, real `H` +
  projection, `P(m) = 1/2` asserted, amplitudes equal to 1e−12.
* `src/shor/sliced.rs`: `missing_mbu_fixup_is_caught` (should_panic: a
  measurement phase without its CZ trips the sign assertion) and
  `mbu_fixup_cancels_the_measurement_phase`.
* `tests/mbu_shor.rs` (6 tests). (i) The exact outcome distributions of the
  whole measurement tree match the permutation oracle to < 1e−12, both
  through the sliced engine and gate by gate on the sparse state (real
  measurements), for N = 15, 21, 35, w = 1…4 and both MBU oracles. (ii) The
  measured integers equal those of the permutation and windowed-opt paths
  up to N = 1 005 973, with fewer Toffolis and measurements > 0. (iii)
  Beyond 64 qubits (24-bit N, w = 4, 5) results match the permutation
  oracle. (iv) The block is independent of the outcome stream. (v) T2(c)
  (§4). (vi) The Toffoli count is at most 55 % of windowed-opt at 20 bits.
* `tests/theory_shor_mbu.rs`: **all T1/T2 checks of `tests/theory_shor.rs`
  re-run on `Oracle::WindowedMbu(4)`**. These include the support law on
  whole gate-level measurement trees (20 instances, exact amplitudes) and
  the work-counter identity `W = Σ 2 B_i G_i` (G = ops of the resolved
  block). The three T3 noise checks return early, because the noisy
  trajectory engine only runs reversible oracles. `theory_shor_opt`,
  `superopt` and `shor_scale` are unchanged and green.

## 3. Counts (whole runs, 2n controlled-U rounds; `research/data/mbu-shor/counts.txt`)

"gates" counts every op: measurements and fix-up Z/CZ count once each.
"steps" are bit-sliced engine steps; an outcome-1 measurement is 2 steps.

| N (bits) | oracle | qubits | gates | Toffolis | X-meas | fix-ups | slice steps |
|---|---|---|---|---|---|---|---|
| 1 005 973 (20) | windowed-opt | 88 | 271 380 | 70 720 | 0 | 0 | 271 380 |
| | mbu-lookup | 88 | 224 622 (−17.2 %) | 55 169 (−22.0 %) | 14 089 | 5 469 | 231 677 (−14.6 %) |
| | mbu | 107 | 294 892 (+8.7 %) | **31 399 (−55.6 %)** | 36 839 | 16 999 | 313 347 (+15.5 %) |
| 10 161 323 (24) | windowed-opt | 104 | 463 922 | 120 672 | | | 463 922 |
| | mbu-lookup | 104 | 388 070 (−16.4 %) | 97 996 (−18.8 %) | 22 780 | 8 035 | 399 458 (−13.9 %) |
| | mbu | 127 | 515 073 (+11.0 %) | **53 969 (−55.3 %)** | 64 049 | 28 706 | 546 963 (+17.9 %) |
| 221 643 407 (28) | windowed-opt | 120 | 736 962 | 189 728 | | | 736 962 |
| | mbu-lookup | 120 | 608 834 (−17.4 %) | 155 095 (−18.3 %) | 34 359 | 11 175 | 626 144 (−15.0 %) |
| | mbu | 147 | 832 514 (+13.0 %) | **85 673 (−54.8 %)** | 102 473 | 45 503 | 883 992 (+20.0 %) |
| 1 537 596 787 (31) | windowed-opt | 132 | 1 039 303 | 261 454 | | | 1 039 303 |
| | mbu-lookup | 132 | 872 071 (−16.1 %) | 218 421 (−16.5 %) | 45 379 | 13 126 | 894 601 (−13.9 %) |
| | mbu | 162 | 1 180 796 (+13.6 %) | **119 096 (−54.4 %)** | 143 400 | 62 495 | 1 252 554 (+20.5 %) |
| same, w = 5 | windowed-opt | 133 | 1 182 602 | 264 306 | | | 1 182 602 |
| | mbu-lookup | 133 | 907 122 | 198 758 | 51 260 | 20 780 | 932 581 |
| | mbu | 163 | 1 171 150 | **114 138 (−56.3 % vs w=4 opt)** | 134 970 | 62 908 | 1 238 701 |

**Ablations.** Without the measured flag (`flag: false`), at 31 bits:
mbu-lookup is 911 367 gates / 232 459 Toffolis and mbu is 1 245 710 /
125 695. So the flag trick is worth −4 % gates / −6 % Toffolis in the
lookup variant, and −5 % / −5 % in the full one. The window sweep (flag
off) moves the gate-optimal window up once lookups are cheap. For
mbu-lookup, Toffolis at w = 3/4/5/6 are 318 k / 232 k / 212 k / 196 k, and
slice steps are lowest at w = 4 (934 k vs 968 k at w = 5). The full MBU
oracle has its fewest steps at w = 5 (1.30 M vs 1.32 M at w = 4).
Measurements cost about one X-measurement per Toffoli saved, as expected:
every temporary AND is uncomputed by one.

## 4. Theory consistency (T1, T2 of `research/theory-shor.md`)

* **T1 (support law).** MBU changes the circuit but not the map. After
  every block the stored support and amplitudes are identical to the
  reversible oracle's, so the support law and the peak-support formula
  are unchanged. `t1_support_law_on_gate_level_tree` passes on
  `WindowedMbu(4)`. So do the closed forms and the work-counter identity,
  with G counted as resolved ops. Memory is unchanged; time scales with
  slice steps instead of gates.
* **T2(c) (two-branch states stay stabilizer only for phases in {±1, ±i}).**
  An X-basis measurement of a deterministic ancilla multiplies each branch
  by `(−1)^{m f(x)}`. So with a two-branch input (`|+⟩` control, x a basis
  state) the relative phase only ever changes by ±1. Weights stay equal,
  because `P(m) = 1/2` scales both branches by the same `√2`. The class
  "equal weight, relative phase in {±1, ±i}" is therefore **closed under
  measured uncomputation and its Z/CZ fix-ups**, and by Thm 2(c) the state
  stays a stabilizer state (ν = 0) at every op boundary, including the
  mid-block points where the relative phase is −1 before the fix-up.
  `t2_mbu_two_branch_boundaries_stay_stabilizer` checks the theorem's
  premises at every boundary: two distinct keys, no measurement of the
  only differing bit, equal weights, sign ±1. It covers N = 15, 21, 143
  and 1003 (up to 64 x per N, every op of the full MBU block). The −1
  relative phase does occur mid-block. A generic phase (Thm 2(c), ν = 1)
  could only enter through a non-Clifford fix-up; there are none. The
  fix-ups are Z and CZ, i.e. diagonal Clifford. This matches T2(c): MBU
  never takes a two-branch state outside the stabilizer set.

## 5. End to end on the Mac

`qsim run shor --semiclassical --sliced --window 4 --oracle {windowed-opt |
windowed-mbu-lookup | windowed-mbu} --modulus N --seed S --tries 1`; the
31-bit run adds `--f32 --seed 2`, the 24- and 28-bit runs use `--seed 1`.
These are the record bases of research/shor.md. Logs:
`research/data/mbu-shor/bench_24_28.log`, `bench_31.log`. "Eval" is the
gate-evaluation time (control-1 + control-0 halves) from `QSIM_SLICE_PROFILE`.

| N (bits) | oracle | qubits | Toffolis / run | X-meas / run | time, 3 runs (s) | min | Δ | eval (min) | Δ eval | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 161 323 (24) | windowed-opt | 104 | 120 672 | 0 | 0.291 / 0.290 / 0.290 | 0.290 | | 0.113 | | 105 MB |
| | mbu-lookup | 104 | 97 996 | 22 780 | 0.259 / 0.250 / 0.250 | 0.250 | −13.8 % | 0.099 | −12.4 % | 105 MB |
| | mbu | 127 | 53 969 | 64 049 | 0.257 / 0.259 / 0.261 | 0.257 | −11.4 % | 0.129 | +14.2 % | 105 MB |
| 221 643 407 (28) | windowed-opt | 120 | 189 728 | 0 | 12.18 / 12.29 / 12.31 | 12.18 | | 8.00 | | 2.83 GB |
| | mbu-lookup | 120 | 155 095 | 34 359 | 11.23 / 11.25 / 11.29 | **11.23** | **−7.8 %** | 7.05 | −11.9 % | 2.83 GB |
| | mbu | 147 | 85 673 | 102 473 | 13.55 / 13.51 / 13.48 | 13.48 | +10.7 % | 9.37 | +17.0 % | 2.83 GB |
| 1 537 596 787 (31) | windowed-opt | 132 | 261 454 | 0 | 98.04 / 98.07 / 98.17 | 98.04 | | 72.31 | | 4.28 GB |
| | mbu-lookup | 132 | 218 421 | 45 379 | 89.25 / 89.79 / 89.34 | **89.25** | **−9.0 %** | 63.69 | −11.9 % | 4.28 GB |
| | mbu | 162 | 119 096 | 143 400 | 112.90 / 112.58 / 112.33 | 112.33 | +14.6 % | 86.81 | +20.1 % | 4.28 GB |

Measured integers, identical across all 9 runs of each N:
150 071 647 041 326, 19 301 499 017 721 646 and
2 059 039 373 337 077 151. All runs found the true order and a factor
(2753, 15601, 52501).

* **Time follows slice steps, not Toffolis.** The eval change matches the
  step-count change of §3 to 2 points at every size. Sort, P(1) merge and
  collapse depend on the support, not the circuit, so they are unchanged
  (31-bit: sort 17.1–17.2 s, collapse 6.7–6.8 s in every run). They cap the
  end-to-end gain below the eval gain.
* At 24 bits the full-MBU oracle is still faster end to end. Its build
  (0.022 s) skips the per-multiplier peephole/SAT passes on the Cuccaro
  block (0.046 s), and at that size build is a visible share of the run.
* **Peak RSS is unchanged.** Memory is set by the support (T1). The full
  oracle's 30 extra qubits only widen the per-batch slice buffers (kB).
* The 31-bit record run is now **89.3 s** with the lookup-only MBU oracle
  (was 97.5 s with windowed-opt in exp/superopt and 133.4 s in round 4;
  same base, same measured integer). Command:
  `qsim run shor --modulus 1537596787 --semiclassical --sliced --window 4 --oracle windowed-mbu-lookup --f32 --seed 2 --tries 1`.

## 5b. 32 bits: N = 3 631 204 201 factored at gate level (caveated)

**Instance.** N = 3 631 204 201 = 58 907 × 61 643 is the 32-bit row of
`research/data/shor_r4/gen_instances.py`: the first balanced 32-bit
semiprime from `random.seed(1)`, not selected. λ(N) = 1 815 541 826 =
2·7²·17·37·29 453.

**RAM first.** By the support law the peak support is
`max(r_odd, r/2)`. For r = λ that is 9.08·10⁸, ≈ 30 GB: impossible on the
16 GB Mac, as research/shor.md already predicted. The base is the first
draw of `StdRng(seed)` (`--tries 1`). `examples/shor_seed_orders.rs` lists
the base, its order and the predicted peak for seeds 1–40
(`research/data/mbu-shor/seed_orders_32bit.txt`):

* seeds 2, 3, 5, 6, 8, 9, 10 need ≈ 30 GB;
* seeds 1, 4 and 7 have small orders, but the classical post-processing
  fails (r odd, or a^(r/2) ≡ −1);
* **seed 11 is the first seed that both fits and can succeed.** Its base is
  a = 1 002 069 679 with r = 259 363 118 = λ/7, ν₂ = 1, and predicted peak
  support r_odd = 129 681 559.

**Caveat.** This selection computed r from λ(N), i.e. it used the
factorisation. It selects which random base is *feasible*, the property
the cost law says governs everything. The quantum simulation itself is
exact and blind: every gate of the 136-qubit circuit on every branch.
This is the same status as choosing a seed whose run fits; it is not a
factoring of an unknown N, and the simulation is no classical speed-up
(research/shor.md §"What is exponential in what").

**Run.** Mac, `--oracle windowed-mbu-lookup`, f32. It ran as a background
(non-timing) job, because a ~3 min run exceeds the 150 s lock chunk:
`RAYON_NUM_THREADS=2`, SIGSTOP-on-lock watcher (never triggered), load
15–18 from other work. Free + inactive memory was checked (≥ 7 GB) before
launch.

```
RAYON_NUM_THREADS=2 qsim run shor --modulus 3631204201 --semiclassical --sliced --window 4 \
    --oracle windowed-mbu-lookup --f32 --seed 11 --tries 1
a=1002069679  qubits=136  measured=3249162659058944524  order=Some(259363118)  factor=Some(61643)
peak_amplitudes=129681559  total_gates=945159  toffoli_gates=236165  mbu_measurements=49093  gate_branch_ops=1.454e14
3631204201 = 58907 x 61643
time 498.353 s (2 threads; eval 379 s, sort 89 s, collapse 23 s)   max RSS 6.28 GB
```

The measured peak support is exactly the support-law prediction
(129 681 559). The order found is the true order. N is factored on the
first base. This is the largest N factored here by simulating every gate
of a Shor circuit; the previous was the 31-bit 1 537 596 787. The 2-thread
wall time is not a benchmark. Scaling the 8-thread 31-bit run by the cost
law (Σ B_i · G ratio ≈ 2.1×) suggests ≈ 3 min with 8 threads.

**Memory constant, corrected.** RSS was 6.28 GB, not the 4.3 GB that
"33 B per peak element" predicts. The 33 B rule (research/shor.md) was
fitted at 31 bits, where ν₂(r) = 2: there the full peak support r/2 is only
reached by the last *materialised* collapse, from a support of r/4. With
ν₂(r) = 1 the support sits at r_odd for 36 rounds, and every collapse runs
at full size. During a collapse the old keys/amplitudes (16 B), the
`(Ux, ψ)` join array (16 B) and the new per-chunk parts (16 B) coexist
(`SlicedState::collapse` drops the old buffers only after the merge). That
gives 48 B × 1.297·10⁸ = 6.2 GB, which matches. Peak RSS is therefore
≈ 48 B · r_odd when ν₂(r) = 1, and ≈ 32 B · r/2 when ν₂(r) ≥ 2. Streaming
the merge into the old buffers would bring the first case down to 32 B.

## 6. Literature: how this compares

| item | literature | here |
|---|---|---|
| temporary AND, 0-Toffoli uncompute | Gidney 2017 (arXiv:1709.06648) | implemented exactly; adder n Toffolis (Gidney: n − 1 with a carry-in-free first bit, the same as ours up to the top carry, which we do as a full CCX into `b[n]` because `b[n]` is not clean) |
| unary-iteration lookup, L − 1 Toffolis | Babbush et al. 2018 (arXiv:1805.03662) | implemented exactly (15 for w = 4) |
| measurement-based unlookup, O(√L) Toffolis | Berry et al. 2019 (arXiv:1902.02134), Gidney 2019 windowed arithmetic (arXiv:1905.07682) | implemented; our phase table with the ANF split costs `(2^{w−k} − 1) + (2^k − k − 1)`, 4 for L = 16. This is the same √L scaling, written as an ANF / unary hybrid. We did not find this exact formulation in those papers, but it is an elementary variant |
| measured modular-adder flag, phase fix on 50 % of shots | Gidney 2025 (arXiv:2505.15917), Sec. 2: 2.5n Toffolis per modular addition (3.5n in Berry et al. 2024) | our version is 3n + ½(n − 1) ≈ **3.5n**. Gidney's 2.5n also merges the `−N` into the looked-up addend (one fewer adder). In the windowed multiplier, that needs the phase fix-up to compare `b` with `L − 2^n + N`, a register-plus-constant comparison. Not done |
| total Toffolis per factoring run | Gidney–Ekerå 2019 (arXiv:1905.09749): 0.3 n³ + 0.0005 n³ lg n Toffolis, 3n + 0.002 n lg n logical qubits (≈ 3·10⁹ at n = 2048; the formula evaluates to 9.0 k at n = 31, but its constants are fitted to RSA-size parameters). Gidney 2025 (arXiv:2505.15917, Table 5): 6.5·10⁹ expected Toffolis per 2048-bit factoring (≈ 9 shots, approximate residue arithmetic) with 1399 logical qubits. That is more Toffolis than GE2019, traded for ≈ 4× fewer logical qubits | **119 096 = 4.0 n³ at n = 31** (windowed-opt: 8.8 n³). The ≈ 13× gap to the GE2019 formula, as a rough large-n decomposition: (1) exponent windowing: GE does one multiply per c_e ≈ 4–5 exponent bits, with lookups over exponent and multiplicand bits together, which gives ×≈ c_e fewer multiplications; (2) the Ekerå–Håstad short exponent: 1.5n instead of 2n exponent bits, ×4/3; (3) coset representation: one n + O(lg n)-bit addition per window and no modular reduction, where ours needs ≈ 3.5n. These are of the right order together (≈ 4.5 · 1.33 · 3 ≈ 18). All three change *what* is computed (approximate coset arithmetic, exponent windows), not how temporaries are uncomputed. At n = 31 GE's optimal window sizes would differ, so the 13× is indicative only |

**Bottom line.** MBU is now part of the exact gate-level simulation, at
no cost to exactness. The engine asserts the phase bookkeeping and has an
independent real-measurement reference path. With it, the oracle's
Toffoli count falls to 46 % of windowed-opt (the "roughly halve" of the
brief). The simulator's own cost does not follow Toffolis: it follows
slice steps, and the halved-Toffoli circuit has more of them. The
lookup-only variant cuts both.

## 7. Limits / not done

* **Gidney's 2.5n modular adder** (merge `−N` into the lookup), see §6.
* **Exponent windowing / coset arithmetic.** These are the remaining ≈ 13×
  to Gidney–Ekerå. Coset representation is approximate (deviant
  amplitudes of size `2^{−c_pad}`). It would be exactly simulable, but it
  changes the measured distribution; that would be a different study.
* **Noise.** The noisy trajectory engine (`src/shor/noisy.rs`) still
  accepts only reversible oracles. Faults on measured qubits and on the
  classical feed-forward would need new fault locations, so the T3 noise
  checks are skipped for the MBU oracle.
* **Engine cost of a measured uncompute** is up to 3 steps: sign, reset and
  CZ. A dedicated fused step (`sign ^= w[t] ^ (w[a]&w[b])`) would make it 2,
  but needs a second step type in the hot loop. Not attempted.
* The peephole / SAT-rule passes of exp/superopt run on the reversible part
  of the Cuccaro modular adder only. They are not run on blocks containing
  `And`/`UnAnd`, because moving a temporary AND would break the
  clean-target / `t = a∧b` invariants the measurement relies on.
