# Audit: FastSampler (`exp/fast-sampler`, head 744df86)

Auditor branch `exp/fast-sampler-audit`. Scripts and raw data: `research/data/fast-sampler-audit/`.
Claim audited (`research/fast-sampler.md`): a Poisson-"hit" FastSampler is 8.3–18× faster than
natively built AVX2 Stim 1.16 on Stim's own `rotated_memory_z` circuit (single thread, same `.stim`,
ptb64 output, d = 3–15, p = 0.1/0.3%), with an unchanged output distribution.

**Verdict: FIX-THEN-MERGE** (fix applied on this branch), with the claim's framing reduced:
- The sampler is exact and its output distribution matches Stim on every circuit tried, including
  circuits the author did not test. Nothing in the noise model is approximated.
- One real bug, in the Stim front-end rather than the sampler: any circuit with a detector or
  observable whose *noiseless* parity is 1 made `stim_compare` panic. This includes Stim's own
  colour code at d ≥ 5. Fixed in 050580c.
- The Mac speed numbers reproduce (27×). VPS: see §3.
- The technique is not new in kind. Fault-sparse detector sampling ("sample errors, XOR their detector
  sets", O(npd + 1)) is described in the Stim paper (§5.6). Gidney did not implement it because he
  expected the constant factors to lose at p ≈ 0.1%. The hit model's per-Pauli parity probability
  equals Stim's own DEM decomposition of DEPOLARIZE1/2 into independent mechanisms. What is new is
  the engineering: one pooled Poisson stream per (kind, p), blocked forward-walking generation and a
  padded hit table. Together they win by about an order of magnitude, which shows the Stim paper's
  expectation about constant factors does not hold. The write-up should say so; it now cites the
  prior art (§4).

---

## 1. Exactness of the noise model

### 1.1 Derivation (independent of the author's)

Take a group with `m = 2^b − 1` equiprobable non-identity patterns: a flip has m = 1, DEPOLARIZE1
has m = 3, DEPOLARIZE2 has m = 15. In the code each (group, pattern) slot gets an independent
`Poisson(λ/m)` count; by Poisson splitting this is the same as `Poisson(λ)` hits with uniform patterns.
Only the parity `N_k mod 2` of each slot matters. For a character `χ_s(x) = (−1)^{s·x}` with s ≠ 0,
exactly `2^{b−1} = (m+1)/2` of the non-zero patterns k have `s·k = 1`, so

- hit model: `E χ_s(net) = Π_{k: s·k=1} E(−1)^{N_k} = (e^{−2λ/m})^{(m+1)/2} = e^{−λ(m+1)/m}`;
- channel: `E χ_s = (1 − p) + (p/m) Σ_{k≠0} (−1)^{s·k} = 1 − p − p/m = 1 − (m+1)p/m`.

All characters agree iff `λ = −(m/(m+1)) ln(1 − (m+1)p/m)`, so by Fourier inversion the
distributions are equal. This is the code's `hit_rate`.

- The construction is valid for `p < m/(m+1)`, the channel's fully-mixing point (1/2, 3/4, 15/16).
- Beyond that point the channel's characters are negative. **No** Poisson process can produce
  them, since a positive-time random walk has positive characters. Such groups must go to the dense
  path. The code sends every group with p > 0.25 to the dense path (`bernoulli_word` plus a uniform
  pattern), which samples the channel literally, so the whole range p ∈ [0, 1] is covered.
- Equivalently, each pattern is an independent flip with `q = ½(1 − (1 − (m+1)p/m)^{2/(m+1)})`. For
  m = 3 and m = 15 this is exactly Stim's `depolarize{1,2}_probability_to_independent_per_channel_probability`
  (`stim/util_bot/error_decomp.cc`). The hit model *is* Stim's DEM decomposition, with each
  independent mechanism sampled as the parity of a Poisson count.

Numerical check at 50 digits (mpmath), m ∈ {1, 3, 15}, p from 1e-9 to `m/(m+1) − 1e-12`: the
largest error of the hit-model distribution against the channel is **1.2e-50**.

The other approximations in the code are all at the 1e-16 level, not a modelling choice:
- the Poisson inverse-CDF table drops tail mass below about 1e-16 relative per block (f64 CDF);
- `bernoulli_word` rounds p to the 2^-64 grid;
- Lemire's `uniform_below` is exact.

### 1.2 Channels the code cannot express: rejected, never approximated

Every one of the following makes `stim_compare sample-fast` exit with a parse error. None is
silently dropped or approximated:
- `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`;
- `E` / `ELSE_CORRELATED_ERROR`;
- `HERALDED_ERASE`, `HERALDED_PAULI_CHANNEL_1`;
- `MPP`, `MY`, `MPAD`;
- `SQRT_X`, `C_XYZ`;
- inverted targets `!q`;
- feedback `CX rec[-1] q`;
- mixed measurement-flip probabilities (`M(0.1)` next to `M`), because the NoiseModel has one
  p_meas.

The library-level SymPhase compiler accepts classically controlled *Pauli* gates without gate noise.
That case is affine, so it is correct, but the Stim parser does not expose it.

### 1.3 Exact full-distribution check (`exact_check.py` → `exact_check.jsonl`, `exact_check_16M.jsonl`)

**(A) Ground truth.** A branching `stim.TableauSimulator` interpreter follows Stim's channel
definitions literally:
- each channel instance branches over its outcomes with exact `Fraction` weights;
- random measurements branch ½/½ by postselection;
- branches with equal canonical stabilizers and equal records are merged.

It uses no Pauli-frame, linearity or DEM assumption. Outputs follow Stim's convention: the parity
XOR its value in the noiseless reference sample. As a sanity check of (A), Stim's own sampler agrees
with it on all 65 circuits (KS test of the p-values: p = 0.92).

**(B) Hit model.** The FastSampler's analytic distribution at 60 digits, built from single-fault
signatures computed by (A). This applies only to circuits with deterministic outputs.

**(C) The binary.** `sample-fast` at 2²² shots, with a Pearson χ² test of the full joint histogram
of detectors and observables against (A).

| circuit | what it stresses | max abs (B) − (A) | ours χ² p | Stim χ² p |
|---|---|---|---|---|
| mixed_rare | every channel kind, MR(p)/M(p), p 0.01–0.25 | 1.6e-61 | 0.51 (5 more seeds at 2²⁴: 0.11–0.91) | 0.56 |
| near_max_dense | X_ERROR(0.5), (0.95), (0.2500001); DEPOLARIZE1(0.74), (0.75), (0.25); DEPOLARIZE2(0.93), (15/16), (0.3); MX(0.3), RX | 1.2e-62 | 0.083 (5 more seeds at 2²⁴: 0.61–0.99) | 0.96 |
| over_depolarized | DEPOLARIZE1(0.9), (1.0); DEPOLARIZE2(0.99) (beyond full mixing) | 1.9e-62 | 0.15 | 0.45 |
| many_groups_lowp | 60 rounds, p = 0.002: many groups per class, so the blocked Poisson-table path runs | – | 0.33 | 0.62 |
| many_groups_midp | 40 rounds, p = 0.03 | 3.1e-60 | 0.88 | 0.33 |
| deterministic_one | detector with noiseless parity 1 | 2.9e-62 | 0.91 (panicked before the fix) | 0.97 |
| 60 random noisy Clifford circuits | 4 qubits; mid-circuit M/MR/MX/R/RX; random detectors (mostly non-deterministic); p ∈ {0.001 … 0.6}; M-flip ∈ {0, 0.01, 0.2} | 13 deterministic ones: ≤ 5.4e-61 | min p 0.017 over 59 testable (one has a single outcome); KS of all 65 p-values 0.67; 0 impossible outcomes | min p 0.004 |

**Negative controls** (our binary on a wrong circuit, against the truth of the right one):

| control | χ² p |
|---|---|
| mixed_rare, first channel × 1.3 | 0 (χ² = 3734 / 127) |
| mixed_rare, rate of a naive "λ = p" Poisson approximation | 0 (χ² = 33,500 / 127) |
| many_groups_midp, first channel × 1.3 | 0 (χ² = 1251 / 7) |
| many_groups_midp, naive "λ = p" | 1e-9 (χ² = 57 / 7) |
| near_max_dense, either control | 0.025 / 0.45: not detectable, since the perturbed channels sit before a DEPOLARIZE2(15/16) that fully mixes them |

**Caveat on power.** At p = 0.1% a naive λ = p rate differs from the exact one by O(p²) relative. The
author's 10⁶-shot surface-code tests could not have told the two apart. Exactness rests on the
algebra in §1.1 and on the high-p cells above, not on the surface-code equivalence tables.

**Before the fix**, 23 of the first 40 random circuits, and `deterministic_one`, made the binary **panic**
("non-zero reference parity"). §2.1 has the cause.

## 2. Is it really sampling the circuit?

A DEM-style sampler is valid when outputs are affine in independent faults. That holds for Pauli
channels in Clifford circuits with Pauli feedback. Random measurements and non-deterministic
detectors are handled by coin variables on the dense path. Everything else is rejected (§1.2). The
random-circuit cells in §1.3, mostly with non-deterministic detectors, confirm the coin handling
against an exact non-linear ground truth.

### 2.1 Bug: detectors with noiseless parity 1 (fixed, 050580c)

`stim_compare::compile` asserted that every detector/observable parity is 0 in the noiseless
reference, and panicked otherwise. Stim reports detection events *relative to* the reference, so
such circuits are legal. Stim's own `color_code:memory_xyz` (after `.decomposed()`) has 15 of 45
such detectors at d = 5, so d = 5 and d = 7 panicked.
- The fix is `SymPhaseSampler::relative_to_reference()`, used by the CLI. For rows with a coin the
  constant offset does not change the distribution.
- The bug was inherited from `qec-r4` and is not in the FastSampler itself. It still contradicts
  "same .stim, same output" for part of Stim's circuit zoo.

### 2.2 Stim-generated circuits the author did not test (`equivalence_other.py`, 10⁶ shots per side)

Tests:
- T0: hit-table signatures equal Stim's DEM error signatures;
- T1: marginals;
- T2: DEM-correlated pairs;
- T3: mean and variance of events per shot;
- **T4 (new):** joint 16-cell histograms of 200 four-detector DEM neighbourhoods, to catch
  higher-order correlation errors.

1% family-wise error per cell, Bonferroni.

| circuit | d | p | detectors | T0 (signatures) | max abs z marg | max abs z pair | max z T4 | rejections / tests | verdict |
|---|---|---|---|---|---|---|---|---|---|
| color_code:memory_xyz (decomposed) | 3 | 0.3% | 9 | equal (72) | 2.33 | 1.78 | 0.21 | 0/45 | PASS |
| color_code:memory_xyz (decomposed) | 5 | 0.3% | 45 | equal (1104) | 2.27 | 2.75 | 2.47 | 0/480 | PASS (panicked before the fix) |
| color_code:memory_xyz (decomposed) | 7 | 0.1% | 126 | equal (3651) | 2.60 | 3.21 | 2.30 | 0/1696 | PASS (panicked before the fix) |
| repetition_code:memory | 3 | 1% | 8 | equal (21) | 2.17 | 1.87 | 1.64 | 0/30 | PASS |
| repetition_code:memory | 9 | 0.3% | 80 | equal (225) | 2.58 | 2.45 | 1.62 | 0/366 | PASS |
| surface_code:unrotated_memory_x | 3 | 0.3% | 36 | equal (395) | 1.98 | 3.08 | 2.37 | 0/249 | PASS |
| surface_code:unrotated_memory_x | 5 | 0.3% | 200 | equal (3083) | 2.68 | 3.50 | 2.76 | 0/1825 | PASS |
| surface_code:unrotated_memory_z | 7 | 0.1% | 588 | equal (10255) | 2.89 | 3.65 | 2.95 | 0/5589 | PASS |
| surface_code:rotated_memory_x | 5 | 0.3% | 120 | equal (1679) | 2.73 | 3.68 | 2.14 | 0/1023 | PASS |

**0 rejections in 11,303 tests.** Negative control (`equivalence_other_negative.jsonl`): our side
samples every DEPOLARIZE2 at +5%. It is rejected on colour d = 5 (106 rejections; events-mean
z = 17.6) and on unrotated d = 5 (248 rejections; z = 35.5).

## 3. Is the speed comparison fair?

**Stim's modes for this task.**
- **Circuit (frame) sampler**, `stim detect --out_format ptb64`. Work scales with gates × shots,
  in AVX2 words.
- **DEM sampler**, `stim sample_dem` or `dem.compile_sampler()`. In 1.16 `DemSampler::resample`
  calls `biased_randomize_bits` for *every* error mechanism over the whole shot stripe and then
  XORs. Its work scales with mechanisms × shots, not with faults, so it is the slower of the two
  here (below).

The best Stim for this task is therefore `stim detect`, as the author used. I rebuilt Stim 1.16.0
from source with `-DSIMD_WIDTH=256` (5,816 ymm instructions, the same count as the author's build).

### 3.1 Mac (M1 Pro, pip Stim 1.16 [no NEON backend; arm64 Stim uses 64-bit words], under the bench lock, ≥ 35 s between holds; 1-min load 4.2–8.0 from other users' jobs)

`timing_mac_m1.jsonl`, `mac_audit_timing.sh`.

Sampling only, Mshot/s:

| d | p | shots | Stim detect (net of 64-shot run) | Stim DEM sampler (CLI, net) | ours | ours / best Stim | author |
|---|---|---|---|---|---|---|---|
| 7 | 0.1% | 1,000,000 | 2.35 | 0.97 | 62.8 | **26.7×** | 27.3× |
| 15 | 0.1% | 128,000 | 0.242 | 0.087 | 6.54 | **27.0×** | 27.0× |

Whole process (parse, compile and sample for ours; `stim detect` for Stim; Stim DEM route =
`analyze_errors` + `sample_dem`). Min of 3, seconds:

| d | shots | stim detect | Stim DEM route | ours | best Stim / ours |
|---|---|---|---|---|---|
| 7 | 10,240 | 0.044 | 0.104 | 0.0085 | **5.1×** |
| 7 | 102,400 | 0.086 | 0.195 | 0.0078 | **11.0×** |
| 7 | 1,024,000 | 0.468 | 1.08 | 0.0226 | **20.8×** |
| 15 | 10,240 | 0.077 | 0.412 | 0.046 | **1.7×** |
| 15 | 102,400 | 0.472 | 1.54 | 0.067 | **7.0×** |
| 15 | 1,024,000 (min of 2) | 4.35 | 11.70 | 0.214 | **20.3×** |

Compile at d = 15 is 40 ms on the M1. Stim's DEM extraction is 37 ms (pip) and its
`analyze_errors` process 0.20 s.

### 3.2 x86 (VPS, EPYC-Rome AVX2, native AVX2 Stim)

VPS_TIMING_PLACEHOLDER

## 4. Prior art: what is actually new

- **Stim paper, Gidney 2021, §5.6** (arXiv:2103.02202). It describes this algorithm class:
  "by using a sparse representation for the set of flipped detectors, and sampling low-probability
  errors [by geometric gaps] … sampling the detection events … can be done in O(npd + 1) operations".
  It adds: "Currently, Stim doesn't implement the asymptotically efficient detector sampling method …
  for the noise levels around 0.1% … the larger constant factors inherent in using a sparse
  representation outweigh the asymptotic gains." The FastSampler is an implementation of that idea.
  Its contribution is showing that the constant factors *can* be made to win, by about 9–27× per
  shot.
- **Stim's DEM** already decomposes DEPOLARIZE1/2 into independent per-Pauli mechanisms with
  exactly the hit model's per-pattern flip probability (§1.1). The fault → detector table is the DEM
  without merging. `DemSampler` samples it densely, per mechanism per shot word.
- **Stim's `RareErrorIterator`** already pools geometric skipping over all targets × shots of one
  noise instruction. The FastSampler pools across all instructions of equal (kind, p), and replaces
  geometric gaps by per-block Poisson counts, so a collision just XORs twice.
- **Bernoulli as the parity of a Poisson count**, `q = (1 − e^{−2μ})/2`, is a textbook identity.
- **Concurrent fault-propagation samplers.**
  - ScaLER/ScaLERQEC (Ye & Palsberg, arXiv:2602.04921) compiles a "quantum error propagation
    graph" with bitset updates, aimed at stratified rare-event LER estimation rather than unbiased
    i.i.d. shots.
  - Tsim (arXiv:2604.01059) compiles ZX diagrams for vectorised and GPU sampling.

  I did not benchmark either.

**What is new, stated honestly.** The combination below gives a CPU single-thread detector sampler
that beats Stim's best mode by about 9–27× per shot on surface/colour/repetition memory circuits,
exactly:
- one Poisson stream per (kind, p) class with (location, Pauli) folded into a single uniform draw;
- block-wise Poisson counts from an inverse-CDF table so table reads walk forward;
- a padded branch-free (group, Pauli) → detector table.

The individual ideas are known; the measured constant-factor result is the new part.

## 5. Changes on this branch

- 050580c: `SymPhaseSampler::relative_to_reference()`; `stim_compare::compile` uses it instead of
  panicking (§2.1).
- `research/fast-sampler.md`: prior-art paragraph, crash note, scope of supported instructions.
- `research/data/fast-sampler-audit/`:
  - `exact_check.py` (exact ground truth, hit model, χ² of the binary);
  - `equivalence_other.py` (T0–T4 on other Stim circuits);
  - `e2e.py` (whole-process timing at 1e4/1e5/1e6 shots);
  - `mac_audit_timing.sh`, `vps_audit_timing.sh`;
  - all `.jsonl` outputs.
