# D = 70 sampling run: protocol, numbers, run-book (drafted 2026-10-07)

Target: tracker issue 228, `doped_random_graph_sampling_nq70_depth70_checks27` (70 qubits, CZ depth 70, 468 T).
Engine: qsim-lab chain sweep with a packed low-precision 2^35 bond register (exp/chain-sweep-lowprec).
Machine: the run PC (a contributor's desktop) (Ryzen 9 9950X3D, 64 GB DDR5, RTX 4090, Windows). Nothing in this file has been run on that PC.
Code for this plan (VPS, Python): `runplan/analyze.py` (seed commit, prefixes, calibration scoring, XEB bookkeeping),
`runplan/sampler_mc.py` (sampler designs, output `runplan/mc.md`), `runplan/calib_mc.py` / `calib_k.py`
(calibration precision, outputs `calib.md`, `calib_k.md`). SUTD's dataset is unpacked in `sutd/`.

## 0. Summary

1. **SUTD published exact D = 70 amplitudes.** Manabe/Gu/Pan's Zenodo record
   ([10.5281/zenodo.21912448](https://doi.org/10.5281/zenodo.21912448), CC BY 4.0, 4 MB) holds 2051 batches × 256
   complex64 amplitudes: for every IBM sample, all 256 completions of q0..q7 with q8..q69 fixed to the sample. That is
   **525,056 exact full-depth amplitudes**. From the file (computed here): IBM's samples have **linear XEB 0.342 ± 0.028**
   (log-XEB 0.3503, which reproduces SUTD's number), and the 525k values follow Porter–Thomas closely
   (E z = 1.0004, E z² = 2.007, E z³ = 6.04; PT gives 1, 2, 6).
2. **So the fidelity of our engine can be measured directly at D = 70, with no model.** Run the sweep on the
   mirrored chain and leave the last m processed qubits (= q0..q(m−1)) open. With q_m..q69 set to an IBM row, one sweep
   returns 2^m amplitudes that are all in SUTD's set. Pooled amplitude fidelity F̂ = |Σ e*l|² / (Σ|e|² Σ|l|²):
   **±0.05 takes 2 sweeps at F ≈ 0.43 and 1 sweep at F ≈ 0.85 (m = 6)**; ±0.02 takes 12 and 2 (table §1.3).
   The same sweeps give the sampler's **expected XEB, computed exactly** on those prefixes.
3. **Format: the highest precision that fits, int6:b64 (50 GiB, F ≈ 0.96).** A sweep is compute-bound (~1 h on CPU,
   to be measured). The packed store adds only ~1.5–2 min per sweep, so int6 costs about the same time per sweep as int4
   but needs ~5× fewer samples. int4 also cannot beat IBM's XEB at any N.
4. **Sampler: tail-open batching, 1 sweep per uncorrelated sample.** Stop the sweep at the cut before the last m
   qubits and contract the register against the m-qubit tail in one streaming read-only pass. This gives 2^m amplitudes
   for ≈ one sweep (m = 6: ~+3 min, while m fewer worldlines save ~5 min). Draw a uniform prefix, then a suffix
   ∝ |l|² inside the batch. That is SUTD's batched sampler, and the XEB loss is a factor (B−1)/(B+1) = 0.97 at m = 6.
   Pair rejection (m = 1) costs 2–4 sweeps per sample; single-amplitude rejection costs 3–10.
5. **Budget (int6, m = 6):** 64 production samples plus 6 calibration sweeps = **~70 sweeps ≈ 3 days on CPU** at
   1 h/sweep (1.5–6 days over the 0.5–2 h range). Predicted XEB 0.92 ± 0.17. That is ≥ 5σ over 0.044 and ≥ 3σ over
   IBM's 0.342.
6. **GPU:** feasible without CUDA (OpenCL ships with the NVIDIA driver). PCIe is not the bottleneck (~2–5 min per sweep
   for the host-resident part). Estimate 10–25 min per sweep, 3–6× faster than CPU. It does not pay off for the
   recommended 3-day run. It becomes worth it for a run of hundreds of samples, for int8 (needs VRAM as extra
   capacity), or for the "inside IBM's 5 h" stretch (§4).
7. **Claim we can support:** XEB significantly above 0.044, and above the device, from uncorrelated samples on one
   desktop. The fidelity is measured against independent exact amplitudes; with SUTD's help, the samples can be scored
   directly. **Not** supported: faster than the quantum run (17,900 s), or the fidelity-weighted equivalent of IBM's
   2051 samples (~750 of ours ≈ 1 month of CPU).

## 1. Scoring and verification (the crux)

### 1.1 What each option gives

| option | what it measures | cost | strength |
|---|---|---|---|
| (a) error model F = exp(−r·R) | prediction from r (D-independent, measured D = 24–48, tail windows) | 0 | extrapolation in D, like Google/Zhao. Fine as a prediction, weak alone |
| (b) re-score with higher precision of our own engine | int8 (F 0.996) needs 64.5 GiB: does not fit in 64 GB RAM, fits only RAM + VRAM split. int6 is the most that fits on CPU | one sweep per sample (no cheaper per bitstring: the 2^35 store is fixed) | self-scored, with correlated rounding. **Not recommended**: if int6 fits, sample with it |
| (b′) slice-sum / meet-in-the-middle | slicing narrows one cut only, so the store stays 2^35; MITM needs two 2^35 registers (2× memory) | — | **does not work** within 64 GB |
| (c) third-party scoring | exact p(x) of OUR samples (SUTD: one 256-batch contraction, 32 s on 8 H100, per sample; 64 samples ≈ 35 min of one node) | their time | **strongest**: model-free linear XEB of our samples |
| (d) as posed: our amplitudes on IBM's 2051 samples → implied XEB | E[z̃(x_IBM)] − 1 = F_ours · 0.342. All 2051 give F to ±0.09 at best; ±0.05 needs ~6800 samples | 2051+ sweeps | **useless** |
| **(d′) direct amplitude comparison against SUTD's 525k exact amplitudes** | F̂ of our engine at D = 70, plus the exact expected XEB of our sampler on those prefixes | 1–12 sweeps (§1.3) | **strong, model-free, independent implementation (SUTD's code, H100s)** |

The slice-sum fact (F = fraction of slices kept, exactly) trades time for fidelity at a fixed 2^35 store. It cannot
buy precision for a few bitstrings, because every one of the 69 cuts is 35 bonds wide (README, FOLD.md).

### 1.2 Recommended protocol (uses (a) + (d′) + (c), per Dylan's decision)

1. **Predict** (a): F_pred = exp(−r·R) with r(int6:b64) = 6.2e-4 and R = the planner's pass count on the mirrored
   chain (71 − m + 1 ≈ 66 with m = 6), so F_pred ≈ 0.96. Written down before any D = 70 sweep.
2. **Calibrate** (d′): K = 6 calibration sweeps, identical to production except for the prefix. The prefixes are
   q8..q69 from IBM rows picked by the committed seed, plus seeded bits for q_m..q7 (`analyze.py calrows`). Run 2 at
   the start, 2 in the middle and 2 at the end. They catch drift or hardware faults and give 6 × 64 = 384 amplitudes.
   Report F̂ (jackknife) and the sampler XEB ratio X̂/X_exact on the same prefixes (`analyze.py calib`).
   **Go/no-go after the first 2:** F̂ within 3σ of F_pred, and the phases of ⟨e|l⟩ consistent across records
   (a conjugation or bit-order bug shows up as F̂ ≈ 0).
3. **Sample**: N = 64 prefixes from a committed seed (sha256 published **before** production), one sweep each, every
   sweep logged. A crashed sweep is re-run with the same index, never skipped.
4. **Predicted XEB of the sample set** = X̂_ratio × (B−1)/(B+1), with SE = √((1+2X−X²)/N ⊕ calibration SE)
   (`analyze.py samples`). It rests on: the calibration prefixes being representative (IBM-derived high bits: their
   marginal weight averages 1.0004 ≈ uniform), and the noise being uncorrelated with the ideal amplitude. That second
   point was checked at D ≤ 48 (the "xeb ratio" column in LOWPREC.md ≈ F) and is checked again at D = 70 by X̂.
5. **Publish samples for third-party scoring** (c): bitstrings, seed, and per-sample amplitudes. Ask SUTD (Manabe/Pan)
   to score them: ~35 min of one 8×H100 node for 64 samples. `analyze.py score` turns their p(x) into linear XEB ± SE.
   This step is Dylan's decision (no posting from this plan).

Claim supported without (c), in the spirit of Zhao et al. 2406.18889. Zhao et al. verified their sampler's fidelity
against exact amplitudes from a previous study (F_num 0.0182% vs F_pred 0.0191%) and reported a predicted XEB for 3M
samples:

> Uncorrelated samples from a sampler whose D = 70 amplitude fidelity, measured against independent exact amplitudes,
> is F̂ = 0.96 ± 0.005. Its expected XEB, measured exactly on 6 prefixes, gives a predicted XEB for the 64 samples of
> 0.92 ± 0.17 (5σ over 0.044, 3σ over the device's 0.342).

With (c): "the 64 samples have linear XEB = x ± 0.17 against exact amplitudes", which is verification option 1/2
directly. Either way, credit SUTD for the reference amplitudes.

### 1.3 Calibration precision (from `calib_k.md`, `calib.md`; real SUTD amplitudes, noise model l = √F e + √(1−F) g)

Sweeps needed for sd(F̂) ≤ 0.05 / ≤ 0.02 (single amplitude ≈ 2× the pair row):

| F | m = 1 (2 amps) | m = 4 (16) | m = 6 (64) | m = 8 (256) |
|---|---|---|---|---|
| 0.43 | 63 / 457 | 8 / 51 | 2 / 12 | 1 / 3 |
| 0.85 | 10 / 51 | 2 / 7 | 1 / 2 | 1 / 1 |
| 0.96 | 2 / 5 | 1 / 1 | 1 / 1 | 1 / 1 |

The sampler XEB ratio is noisier, because it is an expectation over the prefix's Porter–Thomas fluctuation. With
K = 8 at m = 6: F = 0.43 → 0.40 ± 0.04, F = 0.85 → 0.82 ± 0.03, F = 0.96 → 0.94 ± 0.015 (synthetic test of
`analyze.py calib`). Without tail-open (single amplitudes only), the calibration costs ~125 sweeps at F = 0.43 and
~20 at F = 0.85. That is one more reason to build the tail.

### 1.4 Two late inputs: the exact RNS sweep and dithered averaging (exp/ideas c2a24a2)

**RNS exact sweep over Z[ζ8]** (~375 bits per amplitude, ~55 channel-sweeps per exact amplitude, ≥ 34 GB per channel).

- **The tail-open pass applies to RNS too.** It is linear ring arithmetic mod P, so one 55-channel set gives **2^m
  exact amplitudes**, not one. Costs at ~1 h per channel-sweep on CPU and 4–6 min on GPU (projected):

| use | channel-sweeps | CPU / GPU wall | what it adds over SUTD calibration |
|---|---|---|---|
| exact F̂ on one of OUR uniform prefixes (m = 6, 64 amps) | 55 | 55 h / 4–6 h | sd(F̂) 0.066 / 0.03 / 0.008 at F = 0.43 / 0.85 / 0.96, the same as **1 SUTD sweep**. Adds uniform (non-IBM) prefixes and independence from SUTD's code |
| exact sampler-XEB ratio on our production prefixes | 55 per prefix | — / 5 h per prefix | ±~0.04 per prefix at F = 0.96. Checks that the calibration prefixes are representative. **The one RNS use worth doing** (1–2 prefixes on GPU) |
| exact linear XEB of our own samples | 55 per sample | 3520 h / 230–350 h for 64 samples; 1265 h / ~105 h for the 23 samples of 3σ | model-free direct XEB, but SUTD's scoring gives the same in ~35 min of their node |
| IBM's samples | — | — | nothing: SUTD already published them |
| RNS as the sampler | 55 per sample (XEB 0.97) | 3200 channel-sweeps for 5σ | 50× the int6 sampler (63 sweeps). Dead |

- Verdict: not needed for the verification. SUTD direct (1–12 sweeps) ≫ RNS on our own prefixes (55 per batch) ≫ RNS
  scoring of samples (55 per sample) ≫ IBM-sample XEB (useless). Run 1–2 RNS batches on production prefixes as a
  capstone only if the GPU path exists. Add 2–3 redundant channels, because the 375-bit estimate is extrapolated from
  n ≤ 12 windows and a too-small CRT bound fails silently otherwise.

**Dithered K-run averaging**, F_K = 1/(1+(1/F_1−1)/K) (`runplan/dither.py`, `dither.md`).
- Assumes r_dz ≈ 1.2 × r_nearest, as measured for int4:b64 (F_1 0.33 vs 0.40). Tail m = 6, 5σ target.
- The best K is 1 for every format except int4:b64 (K = 2, 363 sweeps vs 270 for nearest int4 and 63 for nearest
  int6).
- It can never win at 64 GB: total cost is K·N(F_K) ≥ K·55, while nearest int6 already gives 63. Dithering only pays
  when it buys a format that otherwise would not fit, which is not the case at 64 GB. **Confirmed: no.**

## 2. Sampling scheme

Model (`sampler_mc.py`): ideal z ~ Porter–Thomas (verified at D = 70 above) and approximate z̃ = |√F e + √(1−F) g|².
XEB is measured against the **ideal** z. N = 9·Var/(XEB − 0.044)² for 3σ (25·… for 5σ). Var = E z² − (1+XEB)² of
the sampled z, ≈ 1 + 2X − X². The brief's 1.7·9/(F−0.044)² is this variance factor. Full table in `runplan/mc.md`;
best settings:

| F | design | sweeps/sample | XEB | N 3σ | sweeps 3σ | N 5σ | sweeps 5σ | N 3σ > IBM 0.342 |
|---|---|---|---|---|---|---|---|---|
| 0.43 | single rejection M = 3 | 3.16 | 0.36 | 135 | 426 | 375 | 1184 | ∞ |
| 0.43 | pair rejection M = 2 | 2.12 | 0.37 | 129 | 273 | 359 | 759 | ∞ |
| 0.43 | **tail m = 6, uniform prefix** | **1.00** | 0.41 | 109 | **109** | 302 | **302** | ∞ |
| 0.85 | single rejection M = 3 | 3.16 | 0.71 | 32 | 100 | 88 | 279 | 108 |
| 0.85 | pair rejection M = 2 | 2.12 | 0.74 | 31 | 65 | 86 | 181 | 98 |
| 0.85 | **tail m = 6, uniform prefix** | **1.00** | 0.81 | 28 | **28** | 79 | **79** | 78 |
| 0.96 | pair rejection M = 2 | 2.11 | 0.84 | 23 | 49 | 65 | 137 | 62 |
| 0.96 | **tail m = 6, uniform prefix** | **1.00** | 0.92 | 23 | **23** | 63 | **63** | 53 |
| 0.96 | tail m = 8, uniform prefix | 1.00 | 0.95 | 22 | 22 | 61 | 61 | 50 |

- **Frugal rejection (single amplitude):** acceptance (1 − e^−M)/M, so ~M sweeps per sample. Small M is cheaper but
  under-samples heavy outputs: XEB 0.81 instead of 0.96 at M = 3. The total-sweep optimum is M ≈ 2–3.
- **Pair (last qubit open, m = 1):** proposal ∝ z̃0 + z̃1, then pick b ∝ z̃b. Optimum M ≈ 2, ~2.1 sweeps per sample.
  Pair **without** rejection (batched, B = 2) has XEB only (B−1)/(B+1) = 1/3 × F. Do not use it.
- **Tail-open m (recommended):** a uniform prefix on q_m..q69, all 2^m suffixes from one sweep, suffix drawn ∝ z̃.
  XEB ≈ F·(B−1)/(B+1); TV error of the uniform-prefix approximation is 1/√(2πB) (0.05 at m = 6, 0.025 at m = 8, as
  in SUTD's §4.2). Prefix rejection (M = 1.2, 1.2 sweeps per sample) wins back the 3% and does not pay. One sample
  per sweep: several samples from one batch share a prefix, so they are **correlated** and do not count (Zhao et al.).
- **Meet-in-the-middle K × J:** the K·J outputs share halves, so they are correlated and useless for uncorrelated
  sampling. It also needs two 2^35 registers (68 GiB at int4). The tail-open pass is the useful form of it: a right
  "half" small enough to contract on the fly.
- **Sequential / marginal sampling:** marginals need ⟨ψ|·|ψ⟩, a doubled network of width ~70. Dead.
- **Top-k post-processing** (Zhao's XEB amplification: take argmax z̃ in each batch, XEB ≫ 1): legal under
  Zhao-style rules but it is XEB spoofing. Do not use it for the headline.

**Bias checks.**
- Rejection with an unnormalized z̃ (the low-precision norm drifts by a factor c) only changes the effective cutoff
  M/c. Accepted samples are still ∝ min(c·z̃, M). Estimate c as the mean z̃ over all proposals (they are uniform).
- The tail sampler needs no normalization: it is conditional within the batch.
- The approximate distribution enters only through XEB = F·(…), which is exactly what X̂ measures at D = 70.
- Selection bias is the real risk: never re-draw or drop a sweep because of its output; commit the seed; log every
  proposal (and its u) for rejection variants.

### 2.1 Tail-open pass (new code, ~200–400 lines Rust plus tests)

- `compile_prefix` already stops the sweep at a clean cut. The register then holds the 35 bonds of edge
  (69−m, 70−m) (on the mirrored chain the tail is q0..q(m−1), SUTD's open qubits).
- The tail tensor is T_b(β) = ⟨b| Π_t G_t Z_{w}^{β_t} |0^m⟩. Here Z^{β_t} acts on the tail wire adjacent to the cut
  (the right half of CZ = Σ P_b ⊗ Z^b), and G_t are the tail's gates between consecutive bond times.
- A = Σ_β R(β) T(β) is a tree reduction in time order. The local (gathered) bits are the earliest slots; each node
  holds a 2^m vector.
- Cost ≈ 2^34 · 3m · 2^m complex MACs. m = 6: ~2.4e13, **~2–4 min**. m = 8: ~1.2e14, ~8–16 min. A full sweep is ~1 h,
  and the m skipped worldlines save m/71 of it. Partial buffers: 2^(35−w)·2^m complex, i.e. MBs. No extra rounding
  (f32), so R drops to ≈ 71 − m + 1.
- Validate against the state vector at n ≤ 24 for m = 1..8, and against the plain sweep's amplitude at n = 70,
  D ≤ 48. Watch the WHT/[[1,1],[1,−1]] bond convention that `cut_tensors_cpu` uses.

## 3. Format choice

Time per sweep: compute ≈ 10907 ops × 2^35 ≈ 3.75e14 amplitude-ops (~2e15 FMA). 16 Zen 5 cores with AVX-512 at
15–30% of peak gives **~0.7–1.5 h**, which matches the brief's 0.5–2 h guess. The f32 inner stages run on 2^22-amplitude
blocks (32 MiB, which fits the 96 MB V-cache CCD). The **packed store traffic** is only 71 passes × read+write × size
at ~60 GB/s: int4 ~85 s, int6 ~125 s per sweep (+1.5–3.5%). Even the worst case (re-streaming the packed store on every
inner stage, ~500×) is 10–15 min. So **time per sweep ≈ independent of format**; the format only sets F and N.

F at R ≈ 66–71 (LOWPREC rates). XEB, N and sweeps use tail m = 6. Totals add 6 calibration sweeps. CPU hours assume
1.0 h per sweep (scale linearly).

| format | store | F (R≈66–71) | F if planner gives R≈250 | sampler XEB | N 3σ | N 5σ | N 3σ > IBM | sweeps (5σ + 6 cal) | CPU h @1 h | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| int4:b64 | 34 GiB | 0.43–0.45 | 0.05 | 0.41 | 109 | 302 | ∞ | 308 | ~320 (13 d) | yes |
| int4:b16:h | 36 GiB | 0.53–0.55 | 0.11 | 0.52 | 71 | 198 | 670 | 204 | ~210 (9 d) | yes |
| int5:b64 | 42 GiB | 0.83–0.84 | 0.52 | ~0.80 | 29 | 82 | 85 | 88 | ~90 | yes |
| int5:b16:h | 44 GiB | 0.87–0.88 | 0.61 | 0.83 | 28 | 79 | 78 | 85 | ~88 | yes (**fallback**) |
| **int6:b64** | **50 GiB** | **0.957–0.96** | 0.86 | **0.92** | 23 | **63** | 53 | **69** | **~72 (3 d)** | yes, ~8 GiB headroom (**recommended**) |
| int8:b256 | 64.5 GiB | 0.996 | 0.99 | 0.97 | 21 | 59 | 47 | 65 | — | only RAM + VRAM split (GPU path) |

- int4:b16:h vs int4:b64: same story, slightly better. Neither reaches the device's XEB.
- **R matters more than bits.** The default CPU blocked schedule has 212–281 passes. At R ≈ 250, int4 is dead, int5
  is marginal, and int6 still gives 0.86. Insist on the 71-pass big-buffer schedule (one rounding per worldline) and
  check R with `--count` on the mirrored chain.
- int7 (unmeasured, r ≈ 1.5e-4, ~58 GiB) is too tight on Windows CPU-only.

## 4. GPU (RTX 4090, 24 GB, no CUDA toolkit)

- **API.** OpenCL 3.0 ships inside the NVIDIA driver: `opencl3` crate, kernels in OpenCL C compiled by the driver,
  64-bit pointers, single allocations up to ~6 GiB. It is the most direct port of the existing Metal (MSL) kernels.
- **Alternatives.**
  - wgpu/WGSL on DX12/Vulkan: no 64-bit integers, binding-size limits of 128 MiB–4 GiB, so chunked indexing. Workable
    but more friction.
  - Vulkan via `ash` with SPIR-V (compiled on the VPS with glslang).
  - The CUDA *driver* API (nvcuda.dll, present) with PTX via `cudarc`: no toolkit on the PC, but the PTX must be
    produced elsewhere.
- **Residency.** ~20 GiB of the packed store in VRAM, the rest in host RAM, streamed both ways every pass.

  | format | host part | PCIe per sweep at ~20 GB/s | overlappable |
  |---|---|---|---|
  | int4 | 14 GiB | ~100 s | yes |
  | int6 | 30 GiB | ~210 s | yes |
  | int8 | 44.5 GiB | ~320 s | yes |

  **PCIe is not the bottleneck.** The compute (fused stages on L2-resident 32 MiB blocks, 72 MB L2, 1 TB/s VRAM) is
  ~5–15 min per sweep.
- **Estimate: 10–25 min per sweep**, 3–6× the CPU. VRAM also adds capacity: int8 (F 0.996) becomes possible
  (64.5 GiB = ~20 VRAM + ~45 host).
- **Is it worth it?** Not for the recommended run: it saves ~2 of 3 days and costs ~2–4 agent-days of kernel work and
  validation. It is worth it if:
  - CPU sweeps come in at ≥ 2 h;
  - we want hundreds of samples (IBM's dataset is ~750 of our samples under fidelity-weighted accounting:
    ~31 CPU-days, ~8 GPU-days);
  - we want the **stretch claim**: significant XEB inside IBM's 17,900 s on one desktop. That needs ~25 int6 samples
    in 5 h, i.e. ≤ 12 min per sweep, which only the GPU could reach.
- **Windows GPU risks.**
  - TDR kills any kernel longer than ~2 s: keep dispatches short or set `TdrDelay` in the registry and reboot.
  - Drive the display from the iGPU so VRAM and the scheduler are free.
  - Don't over-commit VRAM: WDDM silently pages VRAM to system RAM.

## 5. Risks and mitigations

- **Memory / commit.** Windows' commit limit is RAM + pagefile. Keep a system-managed pagefile (so the 50 GiB commit
  succeeds) but make the register **non-pageable**, so it never touches disk:
  - allocate with `VirtualAlloc(MEM_LARGE_PAGES)` (needs the "Lock pages in memory" right, secpol.msc; allocate right
    after boot, before memory fragments). This also cuts TLB misses on the 64 B gathered runs.
  - fallback: `SetProcessWorkingSetSizeEx` plus `VirtualLock`.
  - allocate **once** in a long-lived process and re-zero between sweeps.
- **No-disk rule, other leaks:**
  - `powercfg /hibernate off` (hiberfil.sys would write RAM to disk);
  - crash dump set to "Small memory dump" or none (a full/kernel dump writes RAM);
  - the register is never written to disk; checkpoints are the per-sweep JSONL lines only (KBs).
- **Sleep/power.** `powercfg /change standby-timeout-ac 0`, `monitor-timeout-ac 0`; the program also calls
  `SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)`. Use the Balanced plan (AMD's CCD parking logic expects
  it), and check in Task Manager that all 32 threads are busy, not just one CCD (Xbox Game Bar "game mode" can park
  CCD1). Benchmark 16 vs 32 rayon threads.
- **Windows Update.** Pause updates for the run window (Settings → Windows Update → Pause; it auto-restarts otherwise).
  Defender: add exclusions for the run folder and the exe (prevents scan stalls on the JSONL appends).
- **Detach.** An OpenSSH session's children die with the session. Launch via Task Scheduler ("run whether user is
  logged on or not") or `Start-Process`, and monitor by reading the log and heartbeat files.
- **Thermals and memory stability.** Sustained AVX-512 on 16 cores, ~200 W: watch Tctl (HWiNFO), keep it < 90 °C.
  Use stock or EXPO memory settings that are known stable (2 DIMMs). Non-ECC bit flips: one flip touches one of 2^36
  entries (negligible), or a block scale (one of 2^29 blocks, also negligible), or crashes the process (restart cost
  ≤ 1 sweep). The interleaved calibration sweeps catch gross corruption.
- **Shared machine.** the run PC (a contributor's desktop): needs an exclusive ~3–4 day window (no gaming, no big browser sessions; 8 GiB of
  headroom at int6). The other agent's jobs must be stopped. If the 50 GiB allocation fails, fall back to int5:b16:h
  (44 GiB).
- **Restart cost.** Sweeps are independent. A crash loses the sweep in progress (≤ ~1 h) plus the reallocation (if
  large pages fail after uptime, reboot first). There is no mid-sweep checkpoint, by design (the register can't go to
  disk; a RAM-only checkpoint would need a second 50 GiB). Resume with the first index that has a "start" marker but
  no "done".
- **Correctness.** The first calibration sweep is the D = 70 correctness test (F̂ ≈ 0 means a convention or bit-order
  bug). Before that: the small-n tests (§6 step 2).

## 6. Run-book

The engine CLI below is the **spec** for the build agent; flags may differ. Paths are on the PC unless noted.

**Prerequisites (engine deliverables):**
- E1: packed int6:b64 / int5:b16:h / int4 storage with f32 in-block compute; the 71-pass big-buffer schedule;
  `--count` prints R.
- E2: `--mirror` plus tail-open m (§2.1).
- E3: `run` subcommand. One process, one allocation (large pages, locked), loops over a JSONL job list. Per job it
  appends `{"kind":"start","i"}` before the sweep and the result record after it (fsync), and rewrites `heartbeat.txt`
  (time, job, pass index) every pass.
- E4: result record schema (`analyze.py` reads it):
  `{"kind":"calib"|"sample","i"|"row":…, "tail_m":6, "prefix_bits":"......<64 bits q6..q69>", "amps":[[re,im]×64],
  "format":"int6:b64", "R":66, "t_sweep_s":…}`.
  `prefix_bits` is q0-first with the tail as dots; `amps[j]` has j = int(bits q0..q5, 2), q0 = MSB (SUTD's rule).
  Sample records add `"u_tail"` and `"bitstring_q0_first"`.

**Steps:**

0. **Machine prep (once, admin PowerShell):**
   ```
   powercfg /hibernate off; powercfg /change standby-timeout-ac 0; powercfg /change monitor-timeout-ac 0
   # secpol.msc → Local Policies → User Rights → Lock pages in memory → add the run user; sign out/in
   # System → Advanced → Startup and Recovery → Write debugging information: Small memory dump
   Add-MpPreference -ExclusionPath C:\qsim-run
   # Settings → Windows Update → Pause updates (cover the window); reboot (fresh, unfragmented RAM)
   ```
1. **Build:** `cargo build --release --example chain_sweep` (qsim-lab branch with E1–E4).
2. **Small-n validation (~10 min):** `cargo test --release --test chain_sweep` (incl. new tail tests).
   `chain_sweep validate --n 22 --d 40 --mirror --tail 6 --backends cpu64,cpu32` must match the state vector to
   ~1e-6 relative.
3. **Plan check (seconds):** `chain_sweep lowprec --n 70 --d 70 --mirror --tail 6 --count` → expect R ≈ 66–71.
   **If R > 100, stop** and fix the schedule.
4. **Seeds (on the VPS):**
   ```
   head -c 32 /dev/urandom | xxd -p -c 64 > seed.txt
   python3 runplan/analyze.py commit --seed-file seed.txt
   ```
   Record the hash in qsim-lab (commit message) and PLAN.md **before** step 7. Then:
   ```
   python3 runplan/analyze.py calrows  --seed-file seed.txt --m 6 --k 6  > cal_jobs.jsonl
   python3 runplan/analyze.py prefixes --seed-file seed.txt --m 6 --n 64 > prod_jobs.jsonl
   ```
   Copy both job files to the PC. The seed stays private until publication.
5. **Smoke + calibration 1–2 (~2 h):**
   `chain_sweep run --format int6:b64 --mirror --tail 6 --jobs cal_jobs.jsonl --take 2 --out C:\qsim-run\cal.jsonl`.
   This gives the measured T_sweep and peak RAM (Task Manager: commit < 58 GiB).
6. **Go/no-go (VPS):** `python3 runplan/analyze.py calib cal.jsonl`. Need F̂ ≥ 0.85 and consistent overlap phases.
   - If F̂ ≈ 0: conventions (bit order, conjugation, mirror). Fix, then redo step 2.
   - If 0.1 < F̂ < 0.85: check R and the format and compare with F_pred.
   - Re-plan N with `sampler_mc.py <F̂>`.
7. **Production:** `chain_sweep run … --jobs prod_jobs.jsonl --out C:\qsim-run\samples.jsonl`. After job 32, run
   calibration jobs 3–4; at the end, jobs 5–6.
   **Monitoring:**
   - heartbeat age < 10 min;
   - one result line per ~T_sweep;
   - `analyze.py samples samples.jsonl --xhat … --xse …` lists missing indices to redo;
   - CPU temperature and memory in HWiNFO / Task Manager.
8. **Close:**
   - `analyze.py calib` on all 6 calibration records → F̂, X̂ ratio.
   - `analyze.py samples` → predicted XEB ± SE and σ over 0.044 / 0.342.
   - Archive the JSONL, seed and hash.
   - Third-party scoring request and write-up: Dylan's call. No tracker post without him.

Expected: ~70 sweeps. At 1 h each, ~3 days of wall time (1.5 days at 0.5 h, 6 days at 2 h).
