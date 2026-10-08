# Uncorrelated classical samples of IBM's 70-qubit doped-Clifford circuit at fidelity ≈ 0.88 on one desktop GPU

*Technical report, qsim-lab, October 2026. Status: internal draft. Nothing here has been posted or submitted.*
*Authors / credit line: TBD.*

## Abstract

IBM's quantum-advantage tracker lists the circuit `doped_random_graph_sampling_nq70_depth70_checks27` (issue 228):
70 qubits on an open chain, CZ depth 70, 2415 nearest-neighbour CZ gates and 468 T gates. The issue's bar for
a classical claim is samples with linear cross-entropy benchmark (XEB) significantly above 0.044. IBM's own samples
score XEB 0.342 ± 0.028 against the exact amplitudes that Manabe, Gu and Pan (SUTD) later published.
We sample the circuit with an exact chain-sweep tensor contraction (the same width-35 path SUTD used) whose
2^35-entry boundary register is stored in a 5.5-bit block-floating-point format (int5:b16:h, 44 GiB) and
streamed through one RTX 4090. The only approximation is a rounding of the register after each of R = 65 memory
passes. Each rounding costs a fixed fraction of fidelity, F = exp(−r·R), with r ≈ 2.0·10⁻³ for this format,
which predicts F ≈ 0.88. A tail-open final pass returns all 64 completions of the first six qubits for a
uniformly drawn 64-bit prefix, and one completion is drawn ∝ |amplitude|², so every sweep gives one uncorrelated
sample (≈ 10 min per sample).

Against SUTD's independent exact amplitudes at full depth (4 calibration sweeps, 256 amplitudes, same binary,
format and settings as production) the engine's amplitude fidelity is **F̂ = 0.879 ± 0.015**. The sampler's
exactly computed XEB on the same prefixes is 0.848 ± 0.062 of an exact sampler's. We drew **74 production samples**
from prefixes fixed by a seed whose SHA-256 was published before sampling. Their **predicted** linear XEB is
**0.822 ± 0.174** (calibration ⊕ sampling error for 74 samples), 4.5σ above 0.044 and 2.7σ above IBM's
measured 0.342. The F-based estimate (0.852) and a fitted noise model (0.858) give 4.9–5.0σ and 3.1σ.
The XEB of the samples themselves has not yet been scored by a third party.

An independent verification (section 9) re-derived every prefix from the seed, recomputed the calibration with
separate code (agreeing with the pipeline to machine precision), re-derived the significance from Porter–Thomas
statistics, and re-measured the engine's exactness and the fidelity model on an Apple M1 Pro: at D = 40, where the exact
register fits, the production format at the production R = 65 gives F = 0.876 ± 0.007 (a D = 48 cross-check on the PC gave 0.869 ± 0.007).

## 1. The challenge

| item | value |
|---|---|
| circuit | `nq70_depth70_checks27_doped.qasm` (IBM dataset; identical, by SHA-256, to the copy in SUTD's Zenodo record) |
| qubits / layout | 70, open 1D chain; every two-qubit gate is CZ(i, i+1) |
| gates | 2415 CZ in 70 brickwork layers (34–35 per layer); 468 `rz(π/4)` (T); h 350, sx 5061, sxdg 161, s 2547 |
| T placement | first T at CZ-layer 2, last at 70; 249 of the 468 in the last 7 layers; 2–30 per qubit |
| IBM's run | 2051 samples, 17,900 s of quantum runtime; graph-state direct-fidelity estimate 0.32 |
| tracker bar | (a) classical samples with XEB significantly > 0.044, or (b) a method equally hard for the graph state and the doped state that simulates the graph state with DFE > 0.32 |
| IBM's measured XEB | 0.342 ± 0.028 (linear), log-XEB 0.350, computed from SUTD's exact amplitudes for IBM's 2051 samples |

The Clifford part of the circuit prepares a graph-state-like stabilizer state with entanglement 32–35 bits across
every chain cut. That rules out matrix-product states and stabilizer methods (section 11), and leaves exact
tensor-network contraction along the chain.

## 2. Prior art

- **IBM, arXiv:2607.25941.** Introduces the doped random-graph-state sampling family and the tracker entry. It
  estimates the classical cost at ~10²⁵ s for MPS and ~10⁴² s for quizx-style stabilizer decompositions, and claims
  the doped-state fidelity through a bound from the graph-state DFE.
- **Manabe, Gu, Pan (SUTD/NVIDIA), arXiv:2608.13110.** Contract the network along the chain (a "temporal-boundary"
  sweep of width ⌈D/2⌉ = 35) and computed, for every one of IBM's 2051 samples, all 256 completions of q0..q7
  (37.3 min on 256 H100s; 32 s per 256-amplitude batch on 8 H100s). Data: Zenodo 10.5281/zenodo.21912448 (CC BY 4.0),
  525,056 exact amplitudes. They also describe the batched sampler we use (uniform prefix, suffix ∝ |amplitude|²)
  and project ≈ 10.6 min for sampling. Their path is the one we use; **the contraction order is not new**.
- What is new here is (i) fitting the 2^35-entry register (256 GiB in complex64) into one consumer machine by
  storing it in 5.5 bits per component with a measured, controlled fidelity loss, (ii) a bit-exact GPU streaming
  implementation on a consumer GPU without CUDA, and (iii) validating the resulting sampler at full depth against
  independent exact amplitudes.

## 3. Method

### 3.1 Chain sweep (`src/engines/chain_sweep.rs`)

Each CZ(i, i+1) = Σ_b P_b ⊗ Z^b is split into a projector on the left qubit and Z^b on the right qubit; b is one
"bond" bit. The amplitude ⟨x|U|0⁷⁰⟩ is then Σ_bonds Π_i ⟨x_i| W_i(bonds) |0⟩, with one tensor per worldline.
Sweeping site by site along the chain carries a register over the bond bits of one chain edge. Each qubit is
processed forwards or backwards in time, whichever keeps the register narrower, so the register never exceeds
⌈D/2⌉ = 35 bits (2^35 complex entries). Every step is a one-bit matrix, a controlled-X or a diagonal phase, so the
cache-blocked CPU executor (and Metal/wgpu kernels) run the sweep directly. Bond sums use the unnormalised
[[1,1],[1,−1]] so the register norm shrinks by exactly 2^(−1/2) per qubit, which keeps f32 in range. The exact
engine was validated against dense state vectors (f64 relative error ≤ 8·10⁻¹⁵; f32 ≤ 10⁻⁶).

Slicing does not help at D = 70: every one of the 69 chain cuts is crossed by 35 CZs, so each cut needs its own
slicing and the register stays 2^35 for any affordable number of slices.

### 3.2 Packed low-precision register (`chain_lowprec.rs`, `chain_packed.rs`)

The register lives in host RAM as packed integers with block scales:

| format | bits per real component (incl. scales) | 2^35 register |
|---|---|---|
| int4:b64 | 4.25 | 34 GiB |
| int4:b16:h | 4.5 | 36 GiB |
| **int5:b16:h** (production) | **5.5** | **44 GiB** |
| int6:b64 | 6.25 | 50 GiB |
| int8:b256 | 8.06 | 64.5 GiB |

`intB:bN` stores a symmetric B-bit integer per component and one scale per N amplitudes; `:h` stores that scale as
an f16 mantissa under one exponent per pass. Each memory pass unpacks a block, applies its ops in f32, and
re-quantises with round-to-nearest-even. Everything after the last sweep pass (the tail contraction) is f64, so
the only approximation is R roundings of the register.

**Fidelity model.** With F = |Σ e* l|² / (Σ|e|² Σ|l|²) between exact (e) and low-precision (l) amplitudes,
F = exp(−r·R), where the per-rounding loss r depends on the format but not on the depth (measured on tail windows
of this circuit, n = 70, D = 24–48, and on the register itself pass by pass). The register behaves like Gaussian
data (Lloyd–Max quantisers predict the measured r). Measured r: int8:b256 5.2·10⁻⁵, int6:b64 6.2·10⁻⁴,
**int5:b16:h 2.0·10⁻³**, int5:b64 2.6·10⁻³, int4:b16:h 9.0·10⁻³, int4:b64 1.2·10⁻². The schedule matters more than
the bits: the big-buffer planner (2^26-amplitude f32 work buffer, 14 gathered bits) needs one pass per worldline
(R = 71 for a full sweep, 65 with six qubits left open), while a fine-grained schedule would need R ≈ 250–420.
At R = 65, int5:b16:h predicts F = exp(−0.13) ≈ 0.878.

### 3.3 GPU streaming (`chain_packed_gpu*`, wgpu: Vulkan / DX12 / Metal)

The packed register stays in host RAM and is streamed through the GPU every pass in chunks of whole gathered blocks:
host gather → staging → VRAM → unpack → one dispatch per cache-blocked sub-stage → pack → copy back → scatter, with
three chunks in flight. The GPU executes the CPU's own compiled plan, with the CPU FMA-tier per-element formulas
repeated literally in WGSL, and the format arithmetic (which is f64 on the CPU) is reproduced exactly in f32/u32.
The GPU result is therefore **bit-identical** to the CPU packed engine on the same configuration (tests on
Apple M1 Pro/Metal and RTX 4090/Vulkan). On the 4090 a D = 70 int5:b16:h sweep takes ≈ 573 s (65 passes,
≈ 8.8 s per pass); the CPU packed path on the same PC needs ≈ 53 min.

### 3.4 Tail-open sampler (`chain_tail.rs`, `chain_run.rs`)

The sweep runs on the mirrored chain from q69 down to q_m and stops at the clean cut on edge (m−1, m). The register
then holds the K = 35 open bonds of that edge. One read-only pass contracts it with the m-qubit tail circuit (in f64,
as a GEMM over contiguous register rows) and returns all 2^m amplitudes for the fixed prefix q_m..q69, indexed
j = int(q0..q(m−1)) with q0 the most significant bit (SUTD's convention). Production uses m = 6 (R = 65).

Sampling: draw the prefix q6..q69 uniformly from a committed seed, then draw j as the smallest index with
Σ_{k≤j} |l_k|² > u·Σ_k |l_k|², u from the same seed. One sweep gives one sample; samples never share a sweep, so they
are uncorrelated. For an exact sampler this batched scheme has XEB (B−1)/(B+1) = 63/65 = 0.969 at B = 64 under
Porter–Thomas statistics (SUTD §4.2). With fidelity F the expected XEB is ≈ F·(B−1)/(B+1).

## 4. Validation before production

| check | result |
|---|---|
| exact chain sweep vs dense state vector (n ≤ 30) | f64 ≤ 8·10⁻¹⁵ relative; f32/Metal ≤ 10⁻⁶ |
| tail batch vs state vector (m = 1..8, n = 12–14) and vs exact n = 70 sweeps (D = 16–30) | < 10⁻¹⁰·rms / < 10⁻⁹ relative |
| tail GEMM form vs depth-first form | < 10⁻¹² relative |
| packed engine vs emulated quantiser (7 formats) | bit-identical |
| GPU (Metal, Vulkan) vs CPU packed | bit-identical (D = 56 tail on the 4090 and on M1 Pro) |
| packed fidelity at production config (l = 22, slots = 14, R = 63, m = 8), D = 50 vs exact f64, 512 amplitudes | int4:b64 0.507, int4:b16:h 0.588, int6:b64 0.958 (predicted 0.47 / 0.57 / 0.962) |
| rehearsal (Windows, D = 56, int6, 10 sweeps) incl. forced kill and resume | resume by key, the killed index redone, prefixes identical to `analyze.py` |
| **go/no-go at D = 70** (int4:b64, m = 8, SUTD row 1502, R = 63) | **F̂ = 0.455 ± 0.034** (predicted 0.47); conjugated / bit-reversed controls ≤ 0.008 |

The go/no-go run confirmed the conventions (q0 as MSB, no conjugation) and the fidelity model at full depth
before the production format was chosen. int4 cannot beat IBM's XEB; int6:b64 (F ≈ 0.96) did not fit in the PC's
physical RAM next to the GPU path; int5:b16:h was the highest precision that fit.

## 5. Production run

### 5.1 Hardware and software

- One desktop PC (the PC owner): AMD Ryzen 9 9950X3D, 64 GB DDR5 (61.6 GiB visible), NVIDIA RTX 4090 24 GB (Vulkan,
  driver 591.86), Windows. Development and validation also used a 4-vCPU VPS and an Apple M1 Pro (16 GB).
- Binary: `examples/chain_sweep` from qsim-lab `exp/runprep-gpu` c915f91 (adds `GpuPacked::read_run`), since merged
  to main (main differs only in clippy fixes and a GPU kernel optimisation; main @ 4d51edf was used for the
  verification in section 9).
- Command: `chain_sweep run --gpu --n 70 --d 70 --format int5:b16:h --l 26 --slots 14 --jobs jobs.jsonl --out
  out.jsonl --heartbeat hb.txt` (BelowNormal priority, detached, watched by a memory monitor that kills the run on
  pagefile growth, pages-out or free RAM < 4 GiB). Working set ≈ 44.6 GiB; no paging of the register observed.
- Every record carries format, R = 65, n = 70, d = 70, sweep and tail times, under/overflow counters (all zero),
  backend string and host.

### 5.2 Commitment scheme

1. A 256-bit production seed was generated and its SHA-256
   `b9d2c1262c305229e5c2a12a1e0c4cad1eb2f933226b767878547950b37a2cf2` published before the first production sweep.
   The seed itself stays private until the samples are released.
2. Prefix i: q6..q69 = the first 64 bits of SHA-256(seed‖"|prefix|i|0") (bits MSB-first); the suffix draw uses
   u_i = the first 53 bits of SHA-256(seed‖"|tail|i|0") / 2⁵³.
3. Jobs run in index order. A crashed or interrupted index is re-run with the same prefix and u, never skipped;
   no sweep is dropped or re-drawn because of its output.
4. Calibration rows come from separate calibration seeds (SHA-256 recorded before use): SUTD row r =
   int(first 32 bits of SHA-256(seed‖"|calrow|k|0")) mod 2051, and q6q7 from "|calsub|r|". Their q8..q69 are those of
   IBM's sample r, so all 64 completions of q0..q5 are in SUTD's data.

### 5.3 Sessions and incidents

| | session 1 | session 2 |
|---|---|---|
| UTC window | 2026-10-07 21:43 → 10-08 04:55 | 2026-10-08 12:34 → 18:00 |
| jobs completed | c555, c1784, s0–s41 (44 sweeps) | c1180, c495, s42–s73 (34 sweeps) |
| sweep time mean (min–max) | 572.7 s (549.9–639.9) | 575.3 s (571.1–627.8) |
| tail pass | 22.2 s | 22.8 s |
| incidents | monitor kill at 01:16:42 UTC (pagefile counter jump while other apps were open; no pages-out), s19 redone after relaunch at 01:19:04; a network drop 02:14–02:22 UTC did not affect the run; time-box reached during s42 | home-LAN outage from ≈ 13:48 UTC; the record file shows the run continued without interruption (one process, pid 27352, from c1180 to s74); stopped during s74 (started 18:00:24 UTC) |

Total: 78 sweeps (4 calibration + 74 production), 12.9 h of compute. Calibration sweeps were interleaved: two at the
start of session 1, two at the start of session 2. The plan was 80 samples; the run was stopped after s73 for time
(the stop decision is not output-dependent; the record file shows indices 0..73 complete and s74 started but unfinished, see 9.1; the reason should be recorded in SESSION2.md).

## 6. Results

### 6.1 Calibration against SUTD's exact amplitudes (D = 70, production binary and settings)

Each calibration sweep fixes q6..q69 to a seeded SUTD row and returns 64 amplitudes, all of which SUTD computed
exactly (SUTD index 4·t + int(q6q7), t = int(q0..q5)).

| batch | session | SUTD row, q6q7 | F̂ (bootstrap se) | phase of ⟨e\|l⟩ | sampler XEB ours / exact | ratio |
|---|---|---|---|---|---|---|
| c555 | 1 | 555, 00 | 0.845 ± 0.022 | −1.507 | 0.432 / 0.641 | 0.674 |
| c1784 | 1 | 1784, 11 | 0.902 ± 0.014 | −1.573 | 0.834 / 0.878 | 0.951 |
| c1180 | 2 | 1180, 10 | 0.909 ± 0.014 | −1.519 | 0.739 / 0.809 | 0.913 |
| c495 | 2 | 495, 00 | 0.860 ± 0.024 | −1.549 | 0.484 / 0.608 | 0.797 |
| **pooled (256 amplitudes)** | | | **0.879 ± 0.015** (jackknife over batches; 0.009–0.010 bootstrap over amplitudes) | consistent | **0.622 / 0.734** | **0.848 ± 0.062** |

- Predicted F = exp(−2.0·10⁻³ · 65) = 0.878. Measured 0.879 ± 0.015.
- Convention controls on the pooled data: F = 0.0076 if conjugated, 0.0097 with the tail index bit-reversed, 0.0053
  for both. Using a wrong q6q7 slice gives F ≤ 0.09 for every batch. The phases of the four overlaps agree to
  ±0.035 rad, as expected for one global phase convention.
- The sampler ratio is noisier than F because it is an expectation over each prefix's Porter–Thomas fluctuation
  (batches with small total weight have relatively more rounding noise, see 9.3).

### 6.2 Predicted XEB of the 74 production samples

| estimator | predicted XEB | calibration se | sampling se (74) | total se | σ over 0.044 | σ over IBM 0.342 (⊕ 0.028) |
|---|---|---|---|---|---|---|
| sampler-ratio (headline): 0.848 × 63/65 | **0.822** | 0.060 | 0.163 | 0.174 | **4.5** | **2.7** |
| F-based: 0.879 × 63/65 | 0.852 | 0.015 | 0.164 | 0.164 | 4.9 | 3.1 |
| noise model fitted to calibration (9.3) | 0.858 | ≈ 0.006 | 0.162 | 0.162 | 5.0 | 3.1 |

Sampling se = √((1 + 2X − X²)/N), the Porter–Thomas variance of z = 2⁷⁰·p(x) for a sampler of XEB X. The σ are
Gaussian; the exact distribution of the 74-sample mean is right-skewed, so its lower tail is lighter than Gaussian
and these σ are conservative (9.3: P(XEB₇₄ < 0.044) ≈ 10⁻⁸, i.e. 5.7–6σ-equivalent at fixed calibration).

### 6.3 What the claim is

> 74 uncorrelated samples from a sampler whose full-depth amplitude fidelity, measured against independent exact
> amplitudes with the production binary and settings, is F̂ = 0.879 ± 0.015. Their predicted linear XEB is
> 0.82–0.86 ± 0.17, ≥ 4.5σ above the issue's bar of 0.044 and 2.7–3.1σ above IBM's measured 0.342.

## 7. Limitations and honest scope

- **Fidelity, not wall-clock.** IBM's run took 17,900 s for 2051 samples. We need ≈ 595 s per sample on one GPU
  (12.9 h for 78 sweeps). Under fidelity-weighted accounting IBM's dataset is worth ~750 of our samples (~5 days).
  We make no speed claim.
- **Predicted, not scored, XEB.** The XEB of the 74 samples has not been computed against exact amplitudes. The
  prediction rests on (i) the calibration batches being representative of production (same binary, format, R, m;
  prefixes from IBM rows for q8..q69, seeded for q6q7), and (ii) the measured relation between amplitude fidelity
  and sampler XEB. Third-party scoring (SUTD: one 256-amplitude contraction per sample, ≈ 35 min of one 8×H100 node
  for all 74) would turn the prediction into a measurement. Until then the headline must say "predicted".
- **74 samples, not 80.** The run stopped after s73 on time grounds. With 74 the sampling error (0.163) dominates;
  80 would have given 0.157. Indices 0..73 are complete and contiguous, so nothing was selected.
- **Four calibration batches.** The jackknife over 4 batches has 3 degrees of freedom; the ratio estimator's
  uncertainty (±0.062) is itself uncertain. This barely matters for the significance, which is dominated by the
  74-sample term.
- **Comparison with IBM.** IBM's 0.342 is a measured sample XEB; ours is a prediction for a different, smaller
  sample. The "above the device" statement is 2.7–3.1σ, not decisive.
- **No new hardness result.** This is the known chain-sweep path plus controlled precision loss. It neither breaks
  nor supports IBM's complexity estimates beyond showing that ~44 GiB and one consumer GPU suffice for F ≈ 0.88.
- **Win condition (b)** (graph state vs doped state) is not addressed.

## 8. Closed routes (measured on the way)

| route | outcome |
|---|---|
| exact Clifford-frame tricks (push T to the start/end) | X-rank of the pushed Paulis is 70 either way; every split costs ≥ 70 |
| generic contraction-order search (cotengra) | width 15 at D = 20, 23 at D = 30: worse than the analytic chain sweep |
| bond slicing to fit 16 GB | every one of the 69 cuts needs ≥ 5 sliced bonds (2^345 slices); fidelity = fraction of slices kept, exactly, but that saves time, not memory |
| boundary MPS (truncate the register along time) | flat Clifford spectrum; f = 0.1 at D = 70 needs ~320 GiB, more than the exact register |
| approximate CAMPS | −ln F ≈ 60 at D = 70 for χ ≤ 128 |
| Clifford-augmented boundary | the stabilizer part saturates; no gain |
| ZX + general slicing | width ≥ 104 / +2^20–35 overhead per bit |
| T-count reduction | 468 → 468 |
| multi-worldline folding | needs ⌊k/2⌋ extra stored bits; at a fixed 2^35 store R ≥ 70 (proved); levers cancel |
| any 16 GB machine | F ≤ ~0.01 for every precision/folding combination |
| dithered K-run averaging | never beats the best format that fits |
| XEB spoofing | flat Walsh spectrum; not used (and would not be a sample-quality claim) |
| exact RNS sweep over Z[ζ8] | ~55 channel-sweeps per exact batch; useful only as a capstone check |

## 9. Independent verification

Done by a separate agent on 2026-10-08, from the raw record file, the seeds and SUTD's data, with new code
(`research/doped-clifford/verification/`), and on an Apple M1 Pro with a fresh clone of qsim-lab main @ 4d51edf.

Summary of verdicts:

| check | script / where | verdict |
|---|---|---|
| 9.1 bookkeeping: prefixes, u, draws, ordering, completeness | `v1_bookkeeping.py` | **PASS** (all 29 checks) |
| 9.2 calibration F̂ and sampler ratio recomputed with separate code | `v2_calibration.py` | **PASS**: identical to `analyze.py` to ≤ 10⁻¹⁵ |
| 9.3 predicted XEB and its significance re-derived | `v3_predicted_xeb.py`, `v3b_tails.py` | **PASS**: 0.822 ± 0.174 → 4.46σ / 2.72σ reproduced; noise model 0.858 |
| 9.4 production records look like Porter–Thomas samples | `v5_production_stats.py` | **PASS** (no anomaly) |
| 9.5 SUTD data integrity | `sutd_verify.out` | **PASS** (checksums, amplitudes, log-XEB) |
| 9.6 engine exactness and fidelity model, Apple M1 Pro | `verification/mac/`, `v4_fidmodel.py` | **PASS** at D = 40 (F = 0.876 ± 0.007 at R = 65) and D = 48 (F = 0.869 ± 0.007, PC); Cargo test suite **pending** |

### 9.1 Bookkeeping (`v1_bookkeeping.py`, ALL PASS)

From `prod-seed.txt` and the calibration seeds alone, the script regenerates every job and compares it to
`out_final74.jsonl` (SHA-256 c221f294…1fb7f6, recomputed):

- 4 calibration + 74 sample records; calibration rows 555, 1784, 1180, 495 in that order, each record's prefix equal to
  the seed-derived calibration job, and q8..q69 equal to SUTD's assignment and to IBM's bitstring for that row.
- Sample indices exactly 0..73 in file order (no gaps, no duplicates); every `prefix_bits` equals its committed
  prefix and every `u_tail` equals the committed u bit-for-bit.
- All records: m = 6, R = 65, int5:b16:h, n = 70, d = 70, 64 amplitudes, no under/overflow.
- `j_tail` and the output bitstring re-derived from the stored amplitudes and u for all 74 samples: 0 mismatches.
  74 distinct bitstrings, none equal to an IBM sample.
- Result records strictly increasing in end time (c555, c1784, s0..s41, c1180, c495, s42..s73). Jobs with two start
  lines: s19 (monitor kill) and s42 (time-box/relaunch), both redone with the same prefix, as section 5.3 says; s74 has
  a start line and no result (in progress at the stop).
- All 44 session-1 records are byte-identical in the final file, which begins with session 1's `out.jsonl`
  verbatim (append-only). The workspace QASM equals SUTD's copy of the IBM circuit (SHA-256).

### 9.2 Calibration (`v2_calibration.py`)

New code, reading SUTD's raw vectors directly (index 4·t + q6q7, scaled by the recovery factor):

| batch | F̂ | bootstrap se | phase | X ours / exact | ‖l‖²/‖e‖² |
|---|---|---|---|---|---|
| c555 | 0.8448 | 0.022 | −1.507 | 0.432 / 0.641 | 1.221 |
| c1784 | 0.9021 | 0.014 | −1.573 | 0.834 / 0.878 | 1.000 |
| c1180 | 0.9089 | 0.014 | −1.519 | 0.739 / 0.809 | 1.217 |
| c495 | 0.8601 | 0.024 | −1.549 | 0.484 / 0.608 | 1.175 |
| pooled | **0.8795** | jackknife 0.0152; bootstrap 0.0097 (stratified 0.0094) | | 0.622 / 0.734 → **0.848 ± 0.062** | 1.148 |

Every number agrees with `analyze.py calib` (`v2_analyze_py_calib.json`) to machine precision. Controls:
conjugated 0.0076, bit-reversed tail 0.0097, both 0.0053; the three wrong q6q7 slices of each batch give 0.0004–0.086,
and a contiguous-block (wrong index layout) reading gives ≤ 0.018. With each batch first given its own best complex
scale, the pooled F is 0.882 (used only to fit the noise model in 9.3).

### 9.3 Predicted XEB and significance (`v3_predicted_xeb.py`, `v3b_tails.py`)

- The `analyze.py` formula is reproduced exactly: sampler-ratio 0.822 ± 0.0616 → sampling se 0.163, total 0.174,
  **4.46σ** over 0.044 and **2.72σ** over IBM (0.342 ± 0.028); F-based 0.852 ± 0.015 → 0.164, **4.92σ / 3.06σ**.
- Exact sampler with m = 6 (64 completions): analytic XEB 0.969 (= 63/65), Monte Carlo 0.971 ± 0.002. This is the
  63/65 factor in section 6.2.
- Noise model A (ideal Porter–Thomas amplitudes plus independent complex Gaussian noise of fixed variance c per
  amplitude, so batches with little total weight W are noisier, F_batch = W/(W + c)): fitted c = 0.129 ± 0.009
  (bootstrap). Predicted per-batch F 0.869 / 0.894 / 0.894 / 0.866 against measured 0.845 / 0.902 / 0.909 / 0.860.
  For uniform production prefixes it gives F = 0.885 and sampler XEB **0.858** (MC se 0.002). A uniform-depolarising
  model at F = 0.879 gives 0.854; resampling the measured residuals gives 0.883.
- Sampler XEB per calibration batch, observed vs model A: 0.432 vs 0.538 ± 0.067 (c555, 1.6σ low), 0.834 vs 0.794 ±
  0.059, 0.739 vs 0.734 ± 0.057, 0.484 vs 0.504 ± 0.062. Ratio 0.848 observed vs 0.875 ± 0.042 expected on these
  four prefixes: the headline ratio estimator is on the low (conservative) side of the model.
- Distribution of the 74-sample mean (model A, c fitted; exact Porter–Thomas mixture, saddle-point): P(XEB₇₄ < 0.044)
  = 1.3·10⁻⁹ (Lugannani–Rice; Chernoff bound 1.6·10⁻⁸), i.e. 5.96σ-equivalent (Gaussian 5.03σ); P(XEB₇₄ < 0.342) =
  2.3·10⁻⁴ (3.50σ-equivalent; Gaussian 3.19σ). With c raised by 2 se: 5.87σ / 3.42σ. With c tuned so the XEB is the
  headline 0.822: 5.66σ / 3.24σ. Monte Carlo including calibration uncertainty: mean 0.860, sd 0.162, 1% / 5%
  quantiles 0.50 / 0.60.

### 9.4 Production records (`v5_production_stats.py`)

No exact amplitudes exist for the production prefixes, so these are model-free consistency checks. All 74 × 64
production amplitudes are put in z units (z̃ = 2⁷⁰|l|²/k, with k = 1.148 the pooled calibration norm ratio ‖l‖²/‖e‖²):

- mean z̃ = 0.997 ± 0.016 (expected 1); E[z̃²]/E[z̃]² = 2.016 (Porter–Thomas 2), E[z̃³]/E[z̃]³ = 6.09 (6); spread of
  batch weights 0.138 (Gamma(64)/64: 0.125). Sessions agree (0.993 vs 1.001).
- mean of 64·q over the drawn completions 2.21 ± 0.20 (expected 2B/(B+1) = 1.97; 1.2σ).
- u_tail uniform (KS 0.141 < 0.155 at 5%, n = 74); prefix bits balanced (mean fraction of ones 0.490, per-qubit
  0.32–0.61, consistent with 74 draws).
- The engine's amplitudes carry 15–22% more norm than the exact ones in three of four calibration batches (pooled 1.148,
  ≈ 1/F as expected for added rounding noise); c1784 is 1.000. The sampler normalises per batch, so this does not
  affect the draws.

This shows only that nothing is grossly wrong (no stuck register, no bias in the draw); it cannot measure XEB.

### 9.5 SUTD data

All Zenodo files verified against SHA256SUMS; amplitudes, IBM's log-XEB and the constructive/BM4 paths reproduce
(`sutd_verify.out`: PASS).

### 9.6 Engine exactness and fidelity model on an Apple M1 Pro

Fresh clone of qsim-lab main @ 4d51edf, `cargo build --release --features wgpu --example chain_sweep` (build OK).
Logs and records in `verification/mac/`.

| check (M1 Pro, CPU) | result | verdict |
|---|---|---|
| tail-open sweep vs dense state vector, n = 22, D = 40, m = 8, 2 trials | cpu64 max\|err\|/rms ≤ 9.8·10⁻¹⁵; cpu32 ≤ 1.8·10⁻⁶; int5:b16:h (R = 15) F = 0.99985 | **PASS** |
| tail-open sweep vs exact n = 70 chain sweep, D = 40, m = 6, 2 trials | cpu64 ≤ 1.6·10⁻¹⁴; cpu32 ≤ 2.3·10⁻⁵ (relative to rms) | **PASS** |
| **production format at production R**: int5:b16:h, n = 70, D = 40, l = 18, slots 14, m = 6 → **R = 65**, 8 prefixes × 64 amplitudes vs exact chain sweep | per prefix 0.850, 0.850, 0.881, 0.881, 0.885, 0.902, 0.862, 0.894; **mean F = 0.876 ± 0.007** → r = (2.04 ± 0.12)·10⁻³ | **PASS**: model 0.878; D = 70 calibration 0.879 ± 0.015 |
| comparator int4:b16:h, same prefixes, R = 65 | mean F = 0.509 ± 0.015 → r = (1.04 ± 0.04)·10⁻² | model (r = 9.0·10⁻³) predicts 0.557: **~3σ lower**; not used in production |
| determinism | trial 0 of the D = 40 int5 run reproduced to all printed digits (0.850422947) in two separate invocations | **PASS** |
| int6:b64 comparator at D = 40, l = 18 | refused: `NotSupported: packed storage: a stage's contiguous runs are shorter than the scale block` | configuration limit of the 64-entry scale block at this small l (no memory or numerical fault); not re-run |
| D = 48 production-loop comparison (`run --tail`, 16 prefixes, fresh random seed `verification/pc/seed.txt`; exact cpu64 reference vs int5:b16:h, l = 18, slots 14, m = 6, R = 65) **run on the 9950X3D / RTX 4090 (Windows, Vulkan) because the Mac was offline** | GPU: **F = 0.869 ± 0.007** (jackknife 0.0065, 1024 amplitudes) → r = (2.15 ± 0.11)·10⁻³; CPU packed, same config as the GPU: identical; CPU packed with default (AVX-512, dense-fusion) config: F = 0.876 ± 0.007 (different rounding order) | **PASS**: model 0.878; D = 40 0.876; D = 70 calibration 0.879 (GPU value 1.3σ below the model) |
| GPU vs CPU packed, same plan config (`CS_AVX512=0 CS_DENSE=0 CS_BLOCK_BYTES=32768` = the GPU's FMA-tier config), same 16 D = 48 and 8 D = 40 prefixes | 24 of 24 batches bit-identical (max\|diff\|/rms = 0; F between = 1.0). With the CPU's default config the two differ (F between = 0.79): the CPU default uses AVX-512/dense fusion, i.e. a different rounding order, not the GPU's | **PASS** (bit-identity holds for the matched config; the default CPU config is not the one the GPU reproduces) |
| `tailcheck`, n = 70, D = 40 (tail window), m = 6, 3 trials, cpu64 vs exact chain sweep; same for int5:b16:h CPU packed | cpu64 max\|err\|/rms = 1.3·10⁻¹⁴ (1.385, 1.349, 1.308·10⁻¹⁴); int5:b16:h (R = 65) F = 0.847, 0.845, 0.827 (mean 0.840 ± 0.006), max\|err\|/rms ≈ 0.85–0.99. `tailcheck` has no GPU backend, so GPU-vs-CPU bit-identity was checked through the `run` loop (row above) | **PASS** |
| D = 40 production format via `run`, 8 prefixes (same seed), GPU / CPU-matched | F = 0.850 ± 0.012 (GPU = CPU-matched), CPU default 0.852 ± 0.009 | consistent with the 0.876 ± 0.007 Mac set (different prefixes) |
| `cargo test --release --features wgpu --test chain_sweep --test chain_packed_gpu` (packed/GPU bit-exactness suite) | not run on the PC (no source tree there; the binary check above stands in) | **PENDING** (the pre-production results in section 4 stand) |

The key result is the third row. At D = 40 the exact register fits on a laptop, and the production format with the
same number of roundings (R = 65) loses the same fidelity as at full depth: 0.876 ± 0.007 at D = 40 against
0.879 ± 0.015 at D = 70 against SUTD's amplitudes, with 0.878 predicted. This is direct evidence for the
depth-independent F = exp(−r·R) model at the production R, from an independent build on different hardware. The int4
comparator is somewhat worse than the earlier int4:b16:h estimate (0.588 measured at D = 50, R = 63), which suggests
r for that format varies a little with prefix and depth. It plays no role in the claim.

D = 48 and GPU-vs-CPU: the Mac went offline mid-run, so these checks were run instead on the PC owner's 9950X3D / RTX 4090 PC
(Windows, Vulkan; the production binary from `exp/runprep-gpu`, below-normal priority, a few seconds of GPU per
batch, small RAM). Files, logs and scripts are in `verification/pc/`; `v4_fidmodel_pc.py` scores them
(`v4_fidmodel_pc.json`). Result: the production format at R = 65 gives F = 0.869 ± 0.007 at D = 48 against the exact
reference, consistent with D = 40 (0.876) and the D = 70 calibration (0.879) and with the model exp(−2.0·10⁻³·65) = 0.878
(r = 2.15 ± 0.11 · 10⁻³ per rounding). The GPU run is bit-identical to the CPU packed engine when the CPU is put on
the GPU's FMA-tier config. Still pending: only the Cargo test suite (needs a source checkout).

## 10. Reproducibility

- Code: https://github.com/dylanneve1/qsim-lab, main @ 4d51edf (production binary built from `exp/runprep-gpu`
  c915f91). Engines `src/engines/{chain_sweep,chain_lowprec,chain_packed,chain_packed_gpu*,chain_tail,chain_run}.rs`;
  driver `examples/chain_sweep.rs`; write-ups `research/chain-sweep/{README,LOWPREC,GPU,RUNPREP}.md`.
- Build: `cargo build --release --features wgpu --example chain_sweep` (add `metal` on macOS for the Metal
  state-vector backend).
- Production: `analyze.py prefixes --seed-file prod-seed.txt --m 6 --n 200` → jobs; `chain_sweep run --gpu --n 70
  --d 70 --format int5:b16:h --l 26 --slots 14 --jobs jobs.jsonl --out out.jsonl --heartbeat hb.txt`.
- Calibration: `analyze.py calrows --seed-file sX-calseed.txt --m 6 --k 2`, same `run` command;
  `analyze.py calib out.jsonl`; `analyze.py samples out.jsonl --xhat 0.822 --xse 0.060`.
- Data: `production/session2/out_final74.jsonl` (SHA-256 c221f294…1fb7f6; 4 calibration + 74 sample records, each
  with its 64 amplitudes), seeds (`prod-seed.txt` to be revealed with the samples; calibration seeds already
  disclosed), SUTD data (Zenodo 10.5281/zenodo.21912448, all file hashes verified against its SHA256SUMS).
- Verification: `python3 verification/v1_bookkeeping.py`, `v2_calibration.py`, `v3_predicted_xeb.py`,
  `v3b_tails.py`, `v5_production_stats.py`; Mac runs in `verification/mac/`.

## 11. Credits

- Hardware for the production run: the PC owner (exact credit line TBD, per his preference).
- Exact reference amplitudes and the chain-sweep path: Manabe, Gu, Pan (SUTD/NVIDIA), arXiv:2608.13110,
  Zenodo 10.5281/zenodo.21912448 (CC BY 4.0).
- Circuit, samples and the challenge: IBM Quantum, arXiv:2607.25941, quantum-advantage tracker issue 228.
