# Chain sweep in reduced precision (n = 70 doped-Clifford circuit)

Question: can the exact chain-sweep register (2^35 amplitudes at D = 70,
256 GiB in complex64) be stored in fewer bits per component, with every op
still computed in f32? What does that cost in fidelity?

Code: `src/engines/chain_lowprec.rs`, plus the `lowprec` subcommand of
`examples/chain_sweep.rs`. The storage format is **emulated**. The register
lives in f32, every memory pass runs in f32, and at the end of each pass
every component is rounded to the storage format and back. That is
bit-for-bit what a kernel would produce if it loaded `b`-bit values,
computed in f32 and stored `b`-bit values. The fidelity numbers carry over
to real packed storage; the packed kernels themselves are not written.

Formats (per real component): bf16, fp16, the OCP minifloats (fp8 E4M3 /
E5M2, fp6 E2M3 / E3M2, fp4 E2M1), and `intB`. `intB` is block floating
point: a symmetric `B`-bit integer per component and one scale per block
of amplitudes. Suffixes: `:bN` gives one scale per N amplitudes (minifloats:
a power-of-two scale, 8 bits; ints: an exact f32 scale, 32 bits); `:h`
stores the int scale as an f16 mantissa (rounded up) under one exponent
per pass, 16 bits; `:g` is one global scale per pass; `:sr` is stochastic
rounding. "bits" in the tables includes the scales.

All runs: VPS (4 shared EPYC vCPUs, 2 rayon threads, `prlimit --as=1.5e9`).

## Measurement

- Fidelity `F = |Σ e* l|^2 / (Σ|e|^2 Σ|l|^2)` over `k` uniform bitstrings
  (exact f64 amplitude `e`, emulated low-precision amplitude `l`), with
  jackknife errors. The "xeb ratio" is the linear XEB of sampling from
  `|l|^2`, divided by the ideal one, both estimated from the same uniform
  bitstrings. It is noisier than F.
- `--trace J` also tracks the register fidelity against an f64 register
  every J passes, on the first bitstring. It agrees with the amplitude F
  (e.g. int4:b16:h at D = 24: register 0.557 at pass 65, amplitude
  F = 0.535 ± 0.036). Above D = 40 the amplitude runs use few bitstrings
  (40 at D = 44, 12 at D = 48: ~15–50 s per amplitude and format on 2 VPS
  threads), so the register trace is the sharper number there.
- **Which truncation to use.** The first D layers of the circuit (`truncate`)
  have their T gates on only 19–31 qubits, starting at qubit 20 (D = 24–48).
  Rounding costs nothing while the register is a stabilizer-like state: the
  register trace is flat up to qubit ~20 in those runs. At D = 70 every qubit
  carries T gates (249 of the 468 sit in the last 7 layers). So the honest
  proxy is the **last** D layers (`--tail`, `truncate_window`). Those have
  346–414 of the 468 T gates, on all 70 qubits. All main-table numbers use
  tail windows; head-truncation numbers (more optimistic) are in the appendix.
- Rounding granularity (number of roundings `R` per amplitude):
  - `qubit`: once per worldline, R = 70. This is what an out-of-core or
    big-buffer executor gets. The blocked stage planner run on the D = 70
    plan with a 2^26-amplitude f32 work buffer (512 MiB), 14 gathered bits
    and 12 contiguous low bits (4 KiB runs at 1 byte per amplitude) needs
    **71 passes**. So does any buffer from 2^22 amplitudes with ≥ 16
    gathered bits (`lowprec --count` with `CS_BLOCK_BYTES` / `CS_SLOTS`).
  - `every:26`: every 26 ops, R ≈ 420 at D = 70 (196 at D = 32). This
    matches the ~6 passes per qubit of the in-RAM Metal schedule. The
    default CPU blocked schedule needs 212–281 passes at D = 70.

## Results (tail windows, n = 70, uniform bitstrings)

| format | bits | rounding | D | R | bitstrings | F | xeb ratio |
|---|---|---|---|---|---|---|---|
| fp16:b1024 | 16.004 | every:26 | 32 | 196 | 100 | 1.0000 ± 0.0000 | 1.000 ± 0.001 |
| bf16 | 16.000 | every:26 | 32 | 196 | 100 | 0.9994 ± 0.0001 | 1.000 ± 0.005 |
| int8:b256 | 8.062 | every:26 | 32 | 196 | 100 | 0.9875 ± 0.0017 | 0.984 ± 0.029 |
| e4m3:b256 | 8.016 | every:26 | 32 | 196 | 100 | 0.8632 ± 0.0182 | 0.846 ± 0.063 |
| int6:b64 | 6.250 | every:26 | 32 | 196 | 100 | 0.8828 ± 0.0174 | 0.854 ± 0.078 |
| int5:b64 | 5.250 | every:26 | 32 | 196 | 100 | 0.5550 ± 0.0444 | 0.501 ± 0.208 |
| int4:b16:h | 4.500 | every:26 | 32 | 196 | 100 | 0.1589 ± 0.0424 | -0.057 ± 0.213 |
| fp16:b1024 | 16.004 | qubit | 24 | 70 | 200 | 1.0000 ± 0.0000 | 1.000 ± 0.000 |
| fp16:b1024 | 16.004 | qubit | 32 | 70 | 200 | 1.0000 ± 0.0000 | 1.000 ± 0.000 |
| bf16 | 16.000 | qubit | 24 | 70 | 200 | 0.9998 ± 0.0000 | 1.000 ± 0.003 |
| bf16 | 16.000 | qubit | 32 | 70 | 200 | 0.9998 ± 0.0000 | 1.000 ± 0.002 |
| bf16 | 16.000 | qubit | 40 | 70 | 100 | 0.9998 ± 0.0000 | 1.000 ± 0.002 |
| int8:b256 | 8.062 | qubit | 24 | 70 | 200 | 0.9956 ± 0.0004 | 1.004 ± 0.010 |
| int8:b256 | 8.062 | qubit | 32 | 70 | 200 | 0.9964 ± 0.0003 | 0.998 ± 0.009 |
| int8:b256 | 8.062 | qubit | 40 | 70 | 100 | 0.9968 ± 0.0004 | 1.008 ± 0.009 |
| int8:b256 | 8.062 | qubit | 44 | 70 | 40 | 0.9965 ± 0.0006 | 0.984 ± 0.050 |
| e4m3:b256 | 8.016 | qubit | 24 | 70 | 200 | 0.9378 ± 0.0057 | 0.950 ± 0.046 |
| e4m3:b256 | 8.016 | qubit | 32 | 70 | 200 | 0.9521 ± 0.0044 | 0.952 ± 0.032 |
| int6:b64 | 6.250 | qubit | 24 | 70 | 200 | 0.9445 ± 0.0047 | 0.855 ± 0.050 |
| int6:b64 | 6.250 | qubit | 32 | 70 | 200 | 0.9596 ± 0.0036 | 0.953 ± 0.038 |
| int6:b64 | 6.250 | qubit | 40 | 70 | 100 | 0.9618 ± 0.0057 | 0.938 ± 0.033 |
| int6:b64 | 6.250 | qubit | 44 | 70 | 40 | 0.9669 ± 0.0070 | 0.943 ± 0.097 |
| int6:b64 | 6.250 | qubit | 48 | 70 | 12 | 0.9727 ± 0.0120 | 1.026 ± 0.349 |
| e2m3:b64 | 6.062 | qubit | 24 | 70 | 200 | 0.9413 ± 0.0058 | 0.939 ± 0.042 |
| e2m3:b64 | 6.062 | qubit | 32 | 70 | 200 | 0.9436 ± 0.0053 | 0.962 ± 0.035 |
| int5:b16:h | 5.500 | qubit | 24 | 70 | 200 | 0.8406 ± 0.0154 | 0.917 ± 0.097 |
| int5:b16:h | 5.500 | qubit | 32 | 70 | 200 | 0.8804 ± 0.0105 | 0.778 ± 0.045 |
| int5:b16:h | 5.500 | qubit | 40 | 70 | 100 | 0.8856 ± 0.0164 | 0.837 ± 0.056 |
| int5:b64 | 5.250 | qubit | 24 | 70 | 200 | 0.7980 ± 0.0187 | 0.713 ± 0.098 |
| int5:b64 | 5.250 | qubit | 32 | 70 | 200 | 0.8337 ± 0.0151 | 0.798 ± 0.066 |
| int5:b64 | 5.250 | qubit | 40 | 70 | 100 | 0.8536 ± 0.0166 | 0.848 ± 0.070 |
| int5:b64 | 5.250 | qubit | 44 | 70 | 40 | 0.8288 ± 0.0313 | 0.952 ± 0.358 |
| int5:b64 | 5.250 | qubit | 48 | 70 | 12 | 0.8297 ± 0.0524 | 1.273 ± 1.180 |
| int4:b16:h | 4.500 | qubit | 24 | 70 | 200 | 0.5347 ± 0.0358 | 0.631 ± 0.274 |
| int4:b16:h | 4.500 | qubit | 32 | 70 | 200 | 0.5139 ± 0.0327 | 0.492 ± 0.112 |
| int4:b16:h | 4.500 | qubit | 40 | 70 | 100 | 0.5771 ± 0.0467 | 0.725 ± 0.114 |
| int4:b16:h | 4.500 | qubit | 44 | 70 | 40 | 0.3971 ± 0.0673 | -0.385 ± 0.693 |
| int4:b16:h | 4.500 | qubit | 48 | 70 | 12 | 0.6316 ± 0.1167 | 0.945 ± 0.944 |
| e2m1:b16 | 4.250 | qubit | 24 | 70 | 200 | 0.3870 ± 0.0452 | 0.429 ± 0.220 |
| e2m1:b16 | 4.250 | qubit | 32 | 70 | 200 | 0.2955 ± 0.0389 | 0.247 ± 0.112 |
| e2m1:b16 | 4.250 | qubit | 40 | 70 | 100 | 0.4277 ± 0.0534 | 0.652 ± 0.105 |
| int4:b64 | 4.250 | qubit | 24 | 70 | 200 | 0.3897 ± 0.0413 | 0.372 ± 0.338 |
| int4:b64 | 4.250 | qubit | 32 | 70 | 200 | 0.3974 ± 0.0388 | 0.395 ± 0.138 |
| int4:b64 | 4.250 | qubit | 40 | 70 | 100 | 0.5156 ± 0.0464 | 0.540 ± 0.125 |
| int4:b64:sr | 4.250 | qubit | 24 | 70 | 200 | 0.1466 ± 0.0300 | -0.073 ± 0.200 |
| int4:b64:sr | 4.250 | qubit | 32 | 70 | 200 | 0.1393 ± 0.0333 | 0.045 ± 0.124 |
| int4:b256 | 4.062 | qubit | 24 | 70 | 200 | 0.2540 ± 0.0345 | -0.018 ± 0.197 |
| int4:b256 | 4.062 | qubit | 32 | 70 | 200 | 0.2910 ± 0.0371 | 0.217 ± 0.105 |
| int3:b16:h | 3.500 | qubit | 24 | 70 | 200 | 0.0120 ± 0.0104 | -0.169 ± 0.221 |
| int3:b16:h | 3.500 | qubit | 32 | 70 | 200 | 0.0285 ± 0.0172 | -0.025 ± 0.146 |

### The loss compounds per rounding and does not depend on D

- The register trace falls geometrically. `−ln F` grows linearly with the
  pass index once the register is non-Clifford, at a rate set by the format
  alone.
- The same R gives the same F at D = 24 to 48, within errors. At pass 65
  the register traces at D = 32 / 40 / 44 / 48 are int4:b16:h
  0.561 / 0.572 / 0.572 / 0.568, int5:b64 0.836 / 0.839 / 0.839 / 0.840 and
  int6:b64 0.959 / 0.959 / 0.960 / 0.960. Nothing grows with the register
  width, so the D = 70 value is set by R alone.
- The rate per rounding is the same for `qubit` and `every:26` placement.
  For int4:b16:h, `−ln F / R` is 0.0090 (R = 70) against 0.0094 (R = 196).
  So F(R) ≈ exp(−r·R).
- **Stochastic rounding is worse, not better.** At int4:b64, R = 70:
  F = 0.14 against 0.40 for round-to-nearest. The loss comes from the
  error variance, not from bias. SR is unbiased but has a larger variance,
  and independent errors already add up in quadrature.
- Smaller blocks help a little: int4 with b256 / b64 / b16:h gives
  F = 0.29 / 0.40 / 0.51 (D = 32). Ints beat minifloats at equal bits
  (int4:b64 0.40 vs e2m1:b16 0.30; int6 ≈ e2m3; int8 0.996 vs e4m3 0.95).
  The register is flat: Clifford-like magnitudes, no wide dynamic range
  inside a block.

Per-rounding rates r = −ln F / R. They are pooled (inverse-variance) over
the tail-window amplitude fidelities at D = 24–48. Register-trace rates
are shown for comparison: they are the same at every D to ±3 %.

| format | bits | r (amplitudes) | r (register trace, D = 24…48) | F(D=70), R = 71 | F(D=70), R ≈ 420 | register 2^35 |
|---|---|---|---|---|---|---|
| fp16:b1024 | 16.0 | 4.3e-8 | 4.6e-8 | 1.0000 | 1.0000 | 128 GiB |
| bf16 | 16 | 2.7e-6 | 2.5–2.8e-6 | 0.9998 | 0.9989 | 128 GiB |
| int8:b256 | 8.06 | 5.2e-5 | 4.9–5.2e-5 | 0.996 | 0.98 | 64.5 GiB |
| int6:b64 | 6.25 | 6.2e-4 | 6.3–6.6e-4 | 0.96 | 0.77 | 50 GiB |
| e4m3:b256 | 8.02 | 7.8e-4 | 6.8–7.0e-4 | 0.95 | 0.72 | 64 GiB |
| e2m3:b64 | 6.06 | 8.5e-4 | 8.3–8.4e-4 | 0.94 | 0.70 | 48.5 GiB |
| int5:b16:h | 5.5 | 2.0e-3 | 1.9–2.0e-3 | 0.87 | 0.44 | 44 GiB |
| int5:b64 | 5.25 | 2.6e-3 | 2.7–2.8e-3 | 0.83 | 0.33 | 42 GiB |
| int4:b16:h | 4.5 | 9.0e-3 | 8.6–9.0e-3 | 0.53 | 0.02 | 36 GiB |
| int4:b64 | 4.25 | 1.2e-2 | 1.24–1.25e-2 | 0.43 | 0.007 | 34 GiB |
| e2m1:b16 | 4.25 | 1.4e-2 | 1.33–1.36e-2 | 0.36 | 0.003 | 34 GiB |
| int4:b256 | 4.06 | 1.9e-2 | 1.6–1.7e-2 | 0.27 | 4e-4 | 32.5 GiB |
| int4:b64:sr | 4.25 | 2.8e-2 | 2.6e-2 | 0.14 | 0 | 34 GiB |
| int3:b16:h | 3.5 | 5.5e-2 | 4.6e-2 | 0.02 | 0 | 28 GiB |

The D = 70 columns are extrapolations, F = exp(−r·R). They are justified
by the D-independence above and by tail windows that carry 74–88 % of the
D = 70 T gates. Register sizes are 2^36 components × bits / 8 (1 GiB =
2^30 bytes).

## What fits on owned hardware

The budget is the M1 Pro (16 GB RAM, ~20 GB free SSD) and the VPS (7.7 GB
RAM, ~6 GB free disk, a few GB free in practice).

- The only formats with f ≳ 0.25 at D = 70 need **≥ 4 bits + scales** and
  the **71-pass big-buffer schedule**. int4:b16:h gives F ≈ 0.5 in 36 GiB
  (38.7 GB). int4:b64:h (4.125 bits) gives F ≈ 0.4 in 33 GiB (35.4 GB).
  int4:b256 gives F ≈ 0.28 in 32.5 GiB (34.9 GB). With the in-RAM GPU
  schedule (~420 roundings) 4 bits is dead, and only ≥ 6 bits works.
- Mac alone: 16 GB RAM (~11–12 GB usable for the store after the OS and a
  0.5 GiB f32 work buffer) plus ~20 GB SSD is ~31–32 GB. That is **2–7 GB
  short** of the 4-bit store. With ~30 GB of free SSD, int4:b16:h fits.
  With ~35–40 GB free, int5 (F ≈ 0.85–0.88, 42–44 GiB) fits, which needs
  3–4× fewer samples.
- Mac + VPS meet-in-the-middle does not help. Every cut of this circuit is
  35 bonds wide, so each half needs its own 2^35 register, and the
  left·right dot product needs both. Distributing one register over the
  two machines needs a power-of-two split (half = 16+ GiB on the VPS, which
  does not have it), plus a global-bit exchange over the internet link on
  most passes (GBs per pass at tens of MB/s, hours per pass). Not viable.
- 3 bits (28 GiB, would fit) loses too much: F ≈ 0.01–0.03 already at
  D = 24–32 with R = 70.

### Cost on the Mac, if the SSD space is found (estimate, not measured)

- Compute: 10907 ops × 2^35 amplitudes. At the measured M1 Pro Metal
  throughput (D = 58: 8912 ops × 2^29 in 71.6 s, 6.7e10 amplitude-ops/s)
  that is ≈ 5.6e3 s ≈ 1.6 h per sweep, assuming the out-of-core kernels
  (gather of 4 KiB runs, int4 decode/encode) keep that rate. They do not
  exist yet.
- SSD traffic: with ~11 GB of the store in RAM, the rest (int4:b16:h:
  ~28 GB; int5: ~34–36 GB; int6: ~43 GB) is read and written on each of
  71 passes. At 2–4 GB/s that is ~15–35 s per pass, ~20–40 min per sweep.
  It also means **~2–3 TB written to the SSD per sweep**. SSD wear is the
  real cost.
- Amplitudes per sweep: 1. It is 2 if the last qubit is swept forwards
  and its slot is read before the final projection (not implemented). Any
  further open output costs one more register bit, i.e. double the memory.
  Frugal rejection sampling costs ~8–10 sweeps per sample on single
  amplitudes (M ≈ 8–10 under Porter–Thomas), or ~3–4 on the pair
  (proposal ∝ p0 + p1, M ≈ 3–4).
- Samples: N ≈ 9/(F − 0.044)^2. With the Porter–Thomas variance of the
  estimator (≈ 1 + 2F − F^2) included, multiply by ~1.5–1.8.
  These counts use the pair trick; without it, multiply sweeps by ~2.5.
  - int4:b16:h, F ≈ 0.53, needs ~30 GB free SSD: N ≈ 39 (≈ 65 with the
    variance factor), so ~120–260 sweeps ≈ 250–550 Mac-hours, ~250–520 TB
    written. That is roughly the rated endurance of a 1 TB consumer SSD.
  - int5:b16:h, F ≈ 0.87, needs ~36 GB free SSD: N ≈ 13 (≈ 20), so ~45–80
    sweeps ≈ 100–170 h, ~120–210 TB written.
  - int6:b64, F ≈ 0.96, needs ~43 GB free SSD: N ≈ 11 (≈ 15), so ~35–60
    sweeps ≈ 75–130 h, ~110–180 TB written.
- Rough conclusion: technically feasible on one Mac with ≥ 30–40 GB of
  free SSD. Expect weeks of runtime and a large share of the SSD's write
  endurance. The kernels (packed int4/int5 storage, out-of-core gather
  with 4 KiB runs) still have to be written and timed.

## ~2–3 bits per component: Lloyd–Max and int2/int3 (tail windows)

Goal: a RAM-resident register on the 16 GB Mac (≤ ~14 GiB, i.e. ≤ ~1.75
bits per real component). `lmB` is the Lloyd–Max quantizer for a unit
Gaussian with 2^B levels, scaled by the block RMS of the nonzero
components. Exact zeros (structurally free register halves, known from
the plan) are kept and need no storage. `int2` has the levels {−1, 0, 1}.
One rounding per worldline (R = 70). Rates come from the register trace
slope (passes 20→65), which is the same at D = 24 and D = 32.

| format | bits | F (D=24, 200 bs) | F (D=32, 100 bs) | r per rounding | Gaussian scalar-LM MSE |
|---|---|---|---|---|---|
| int4:b16:h | 4.5 | 0.535 ± 0.036 | 0.514 ± 0.033 | 0.009 | |
| lm3:b64:h | 3.125 | 0.038 ± 0.019 | 0.17 ± 0.05 | 0.035 | 0.0345 |
| lm3:b256:h | 3.03 | 0.053 ± 0.023 | 0.09 ± 0.04 | 0.035 | 0.0345 |
| int3:b64:h | 3.125 | 0.007 ± 0.009 | 0.05 ± 0.03 | 0.069 | |
| lm2:b256:h | 2.03 | 0.009 ± 0.010 | 0.010 ± 0.014 | 0.12 | 0.1175 |
| lm2:b64:h | 2.125 | 0.006 ± 0.008 | 0.009 ± 0.013 | 0.12 | 0.1175 |
| lm2:g | 2.0 | 0.005 ± 0.007 | | ≫ (one scale is not enough) | |
| int2:b16:h / b64:h | 2.5 / 2.1 | ≈ 0 | | ≈ 0.4–0.5 | |

**The per-rounding loss equals the quantizer's relative MSE on a unit
Gaussian** (lm3 0.035 vs 0.0345, lm2 0.12 vs 0.1175). The register
entries behave like i.i.d. Gaussians, and the fidelity follows
F ≈ exp(−MSE(bits)·R) with no structure to exploit. Better quantizers can
therefore win at most the gap to the rate–distortion bound 2^(−2b):
0.0625 at 2 bits (E8 / trellis-coded quantization get to ~0.07), and
0.088 at 1.75 bits.

D = 70 extrapolations, F = exp(−r·R):

| quantizer (bits/comp) | R = 71 | R = 35 | R = 24 | R = 12 | register 2^35 |
|---|---|---|---|---|---|
| int4:b16:h (4.5) | 0.53 | 0.73 | 0.81 | 0.90 | 36 GiB |
| lm3:b256:h (3.03) | 0.08 | 0.29 | 0.43 | 0.66 | 24.3 GiB |
| R–D bound, 2.5 bits (r = 0.031) | 0.11 | 0.34 | 0.47 | 0.69 | 20 GiB |
| lm2:b256:h (2.03) | 2e-4 | 0.015 | 0.056 | 0.24 | 16.2 GiB |
| R–D bound, 2 bits (r = 0.0625) | 0.012 | 0.11 | 0.22 | 0.47 | 16 GiB |
| R–D bound, 1.75 bits (r = 0.088) | 0.002 | 0.046 | 0.12 | 0.35 | 14 GiB |

So a ≤ 14 GiB register needs both ideal (vector or entropy-coded)
quantization at ~1.75 bits and R ≲ 12–24 roundings. A ≤ 16.2 GiB register
with the simple lm2 quantizer needs R ≈ 12 for F ≈ 0.24. The planner's
current minimum is R = 71 (one per worldline). Fewer roundings would need
several worldlines folded into one pass, which is not done here.

## Appendix: head truncations (first D layers; T on qubits ≥ 20 only)

Optimistic (no loss before the first T qubit); kept for reference.

| format | rounding | D | R | bitstrings | F |
|---|---|---|---|---|---|
| bf16 | every:26 | 24 | 142 | 200 | 0.9997 ± 0.0000 |
| bf16:b1024 | every:26 | 24 | 142 | 200 | 0.9997 ± 0.0000 |
| fp16 | every:26 | 24 | 142 | 200 | 0 (underflow) |
| fp16:g | every:26 | 24 | 142 | 200 | 1.0000 ± 0.0000 |
| fp16:b1024 | every:26 | 24 | 142 | 200 | 1.0000 ± 0.0000 |
| e4m3:b256 | every:26 | 24 | 142 | 200 | 0.9300 ± 0.0078 |
| e5m2:b256 | every:26 | 24 | 142 | 200 | 0.7843 ± 0.0202 |
| bf16 | every:26 | 32 | 190 | 200 | 0.9997 ± 0.0000 |
| bf16:b1024 | every:26 | 32 | 190 | 200 | 0.9997 ± 0.0000 |
| fp16 | every:26 | 32 | 190 | 200 | 0 (underflow) |
| fp16:g | every:26 | 32 | 190 | 200 | 1.0000 ± 0.0000 |
| fp16:b1024 | every:26 | 32 | 190 | 200 | 1.0000 ± 0.0000 |
| e4m3:b256 | every:26 | 32 | 190 | 200 | 0.9078 ± 0.0084 |
| e5m2:b256 | every:26 | 32 | 190 | 200 | 0.7055 ± 0.0244 |
| int8:b256 | qubit | 32 | 70 | 200 | 0.9980 ± 0.0002 |
| int6:b64 | qubit | 32 | 70 | 200 | 0.9688 ± 0.0031 |
| int5:b64 | qubit | 32 | 70 | 200 | 0.8886 ± 0.0098 |
| int5:b16:h | qubit | 32 | 70 | 200 | 0.9123 ± 0.0080 |
| int4:b256 | qubit | 32 | 70 | 200 | 0.4701 ± 0.0330 |
| int4:b64 | qubit | 32 | 70 | 200 | 0.6187 ± 0.0291 |
| int4:b16:h | qubit | 32 | 70 | 200 | 0.6889 ± 0.0238 |
| int4:b64:sr | qubit | 32 | 70 | 200 | 0.2331 ± 0.0366 |
| int3:b16:h | qubit | 32 | 70 | 200 | 0.1708 ± 0.0307 |
| e2m3:b64 | qubit | 32 | 70 | 200 | 0.9632 ± 0.0033 |
| e2m1:b64 | qubit | 32 | 70 | 200 | 0.5336 ± 0.0321 |
| e2m1:b16 | qubit | 32 | 70 | 200 | 0.5901 ± 0.0292 |

## Reproduce

```text
# pass counts at D = 70 for a given work buffer
CS_BLOCK_BYTES=$((8<<26)) CS_SLOTS=14 cargo run --release --example chain_sweep -- lowprec --n 70 --d 70 --count
# fidelity table rows (tail windows, one rounding per worldline)
RAYON_NUM_THREADS=2 cargo run --release --example chain_sweep -- lowprec --n 70 --d 32 --tail --k 200 \
    --gran qubit --trace 5 --formats bf16,int8:b256,int6:b64,int5:b64,int4:b16:h,int4:b64:sr
```
