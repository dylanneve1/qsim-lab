# Boundary MPS: an approximate chain sweep for IBM's doped-Clifford circuit

Status: experiment (October 2026). **Negative result.** Truncating the
chain sweep's boundary does not bring the depth-70 circuit within reach of a
16 GB machine at any useful fidelity. At D = 70, fidelity 0.1 needs a bond
dimension of about 2^16.2, which is a larger object (about 320 GiB) than
the exact 2^35-entry register (256 GiB).

Engine: `src/engines/chain_mps.rs`. Driver: `examples/chain_sweep.rs`
(`dumpcut`, `exactamps`, `mpsfid`, `mpsprofile`). Tests:
`tests/engines/chain_mps.rs`. Raw logs: `boundary-mps-logs/`. Spectra script:
`boundary_spectra.py`. The exact engine and the circuit are described
in [README.md](README.md).

## Design

The exact sweep ([README.md](README.md)) carries a dense vector over the
`m ≈ D/2` bond bits of the current cut and updates it with a compiled stream
of single-bit matrices, controlled X's and one- and two-bit diagonal terms
(`KOp`s). `chain_mps` runs **the same op stream** on an MPS over the
register bits:

- One-bit ops act on the orthogonality centre, which is moved there with QR.
  Two-bit ops contract the two sites, apply the 4×4 matrix and split with an
  SVD. The bond is capped at `χ`, and singular values with
  `s²/Σs² < 1e-13` are dropped (the projectors in the stream produce exact
  zeros).
- The ops are not unitary. The tensors are kept normalised in mixed
  canonical form and the norm is carried as a log-scale, so every truncation
  is locally optimal and its discarded weight is exact. Truncation is a
  projection: the amplitude is that of the projected state, not
  renormalised. `fid_est` is the product of the kept weights.
- Site order: the exact sweep's slot allocation already puts the register
  bits in time order (qubit `i+1`'s bond at time `t+1` reuses the slot of
  qubit `i`'s bond at time `t`). Almost every two-bit op therefore acts on
  chain neighbours. The rest are routed with SWAPs, and the permutation is
  kept rather than undone (55–140 SWAPs per amplitude against 800–1300
  SVDs).

Why time order, and why not a mixed dense/MPS register: the boundary vector's
Schmidt rank across a time cut `t` is bounded by `2^min(t, m−t)` (and by
`2^(e+1)` for cut `e`). The measurements below show that it **saturates**
that bound, with a nearly flat spectrum. No ordering of the bits beats a
saturated "every cut maximally entangled" profile. A dense window of `k`
bits plus MPS tails has the same middle-cut rank. What any compressed
boundary must do is drop rank at the middle cuts, so the fidelity cost per
dropped bit (below) applies to all such designs.

## Validation

- `tests/engines/chain_mps.rs`: with `χ` at least the Schmidt ranks, every
  amplitude of random brickwork circuits (n ≤ 8, arbitrary rotations)
  matches the state vector to 1e-10 of the RMS amplitude. The MPS register
  after every qubit prefix matches the dense register of the exact sweep
  (truncated IBM circuit, n = 12). Truncated IBM circuit n = 20, D = 24:
  exact at `χ = 128`; at `χ = 8` it truncates and reports `fid_est < 0.99`.
  A unit test covers the Gram-matrix SVD fallback (below).
- n = 70: at `χ = χ_ex = 2^⌊D/4⌋` the overlap with the exact amplitudes is
  F = 1.00000 for D = 24, 28, 32 (400/400/256 bitstrings), with max error
  4e-13, 2e-5, 4e-5 of the RMS amplitude. The 1e-5 level comes from the
  relative SVD cutoff (1e-13 in weight, summed over ~1000 SVDs).
- faer's SVD failed to converge on a few exactly degenerate 128×128
  matrices at D = 36, even after `mps::robust_thin_svd`'s retries. The
  truncation step then falls back to diagonalising the smaller Gram matrix.
  Its error, ~1e-16 `s_max²` in `s²`, is far below the truncation error.

## Spectra of the exact boundary (VPS, f64)

`dumpcut` writes the exact left/right boundary tensors of a cut, and
`numpy.linalg.svd` gives the spectrum across every time cut `t` of the
`m = D/2` bonds (bits in time order). n = 70, middle cut `e = 34`, one
uniformly random bitstring:

| D | m | Schmidt rank at t | weight kept at `χ = rank/2` (middle t) | kept at `rank/4` |
|---|---|---|---|---|
| 24 | 12 | `2^min(t, m−t)` at every t | 0.891 | 0.619 |
| 32 | 16 | `2^min(t, m−t)` at every t | 0.890 | 0.615 |
| 40 | 20 | `2^min(t, m−t)` at every t (L); same (R) | 0.929 (L), 0.889 (R) | 0.690 (L), 0.604 (R) |

The spectrum is flat at cuts that see few T gates. At cut `e = 10`
(D = 40), every Schmidt value is equal: rank `2^r` and weight exactly `2^-k`
kept by `χ = 2^(r−k)`. The middle cut is only mildly non-flat.

## Fidelity against exact amplitudes (n = 70)

`mpsfid`: uniform random bitstrings, exact amplitudes from the chain sweep
(CPU f64 for D ≤ 36, Metal f32 for D = 40, 48), boundary-MPS amplitudes at
each `χ`. `F = |Σ a* a'|² / (Σ|a|² Σ|a'|²)` over the bitstrings (state
fidelity estimate, noise floor ~1/k), with a bootstrap standard error.
`fid_est` is the mean product of kept weights. `χ_ex = 2^⌊D/4⌋` is the
exact cap (F = 1 within 1e-5 at `χ_ex`). `j = log2(χ_ex/χ)` = bits cut.
Times are per amplitude, one CPU thread, on an M1 Pro (4–6 amplitudes in
parallel, machine shared with other jobs). Memory is the largest MPS (site
tensors, complex f64).

| D | χ_ex | k | χ | j | F ± se | fid_est (mean) | log10 fid_est (mean) | s / amp | MPS MiB |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 64 | 400 | 64 | 0 | 1.00000 (err 4e-13 rms) | 1 | 0 | 0.22 | 0.2 |
| 24 | 64 | 400 | 48 | 0.42 | 0.429 ± 0.026 | 0.429 | −0.37 | 0.22 | 0.1 |
| 24 | 64 | 400 | 32 | 1 | 0.022 ± 0.012 | 0.0114 | −1.95 | 0.17 | 0.1 |
| 24 | 64 | 400 | 24 | 1.42 | (noise) | 4.8e-5 | −4.32 | 0.11 | 0.1 |
| 24 | 64 | 400 | 16 | 2 | (noise) | 3.7e-8 | −7.44 | 0.07 | 0.0 |
| 24 | 64 | 400 | 8 | 3 | (noise) | 2.0e-15 | −14.96 | 0.02 | 0.0 |
| 28 | 128 | 400 | 128 | 0 | 1.00000 | 1 | 0 | 1.4 | 0.7 |
| 28 | 128 | 400 | 96 | 0.42 | 0.301 ± 0.029 | 0.374 | −0.43 | 1.2 | 0.5 |
| 28 | 128 | 400 | 64 | 1 | 0.012 ± 0.008 | 0.0098 | −2.01 | 0.98 | 0.4 |
| 28 | 128 | 400 | 48 | 1.42 | (noise) | 6.4e-5 | −4.20 | 0.61 | 0.3 |
| 28 | 128 | 400 | 32 | 2 | (noise) | 6.9e-8 | −7.17 | 0.34 | 0.2 |
| 28 | 128 | 400 | 16 | 3 | (noise) | 2.4e-13 | −12.70 | 0.10 | 0.1 |
| 32 | 256 | 256 | 256 | 0 | 1.00000 | 1 | 0 | 8.6 | 2.7 |
| 32 | 256 | 256 | 192 | 0.42 | 0.305 ± 0.035 | 0.384 | −0.42 | 9.5 | 2.2 |
| 32 | 256 | 256 | 128 | 1 | 0.011 ± 0.008 | 0.0112 | −1.95 | 6.0 | 1.7 |
| 32 | 256 | 256 | 96 | 1.42 | (noise) | 8.4e-5 | −4.08 | 3.5 | 1.1 |
| 32 | 256 | 256 | 64 | 2 | (noise) | 1.0e-7 | −7.00 | 1.9 | 0.7 |
| 32 | 256 | 256 | 32 | 3 | (noise) | 3.1e-13 | −12.53 | 0.50 | 0.2 |
| 36 | 512 | 160 | 384 | 0.42 | 0.572 ± 0.039 | 0.596 | −0.23 | 52 | 8.7 |
| 36 | 512 | 160 | 256 | 1 | 0.053 ± 0.027 | 0.059 | −1.23 | 36 | 6.7 |
| 36 | 512 | 160 | 128 | 2 | (noise) | 2.4e-7 | −6.62 | 12 | 2.7 |
| 36 | 512 | 160 | 64 | 3 | (noise) | 9.7e-13 | −12.02 | 2.7 | 0.9 |
| 40 | 1024 | 24 | 724 | 0.50 | 0.21 ± 0.11 | 0.281 | −0.55 | 338 | 33.3 |
| 40 | 1024 | 36 | 512 | 1 | 0.012 ± 0.033 | 0.0151 | −1.82 | 268 | 26.7 |
| 40 | 1024 | 60 | 256 | 2 | (noise) | 2.1e-7 | −6.69 | 76 | 10.7 |
| 40 | 1024 | 60 | 128 | 3 | (noise) | 1.0e-12 | −12.00 | 16 | 3.7 |
| 44 | 2048 | 2 | 512 | 2 | – | 5.4e-7 | −6.27 | 533 | 42.7 |
| 44 | 2048 | 2 | 256 | 3 | – | 4.2e-12 | −11.38 | 110 | 14.7 |
| 48 | 4096 | 24 | 256 | 4 | 0.08 ± 0.07 (noise floor ~1/k) | 4.7e-17 | −16.33 | 149 | 18.7 |
| 52 | 8192 | 2 | 512 | 4 | – | 1.6e-16 | −15.79 | 1099 | 74.7 |
| 52 | 8192 | 2 | 256 | 5 | – | 1.2e-20 | −19.91 | 185 | 22.7 |
| 56 | 16384 | 2 | 512 | 5 | – | 2.6e-20 | −19.59 | 1390 | 90.7 |
| 56 | 16384 | 2 | 256 | 6 | – | 1.7e-24 | −23.77 | 225 | 26.7 |
| 70 | 131072 | 2 | 512 | 8 | – | 4.1e-30 | −29.48 | 2385 | 146.7 |
| 70 | 131072 | 2 | 256 | 9 | – | 5.7e-35 | −34.26 | 356 | 40.7 |
| 70 | 131072 | 2 | 128 | 10 | – | 8.8e-42 | −41.07 | 58 | 11.2 |

Rows with `–` have no exact reference. At D ≥ 44 the χ that a
single-threaded run reaches gives a fidelity far below any measurable F
(the noise floor of F is ~1/k), so only the truncation estimate is
reported. It is validated against F at D ≤ 40 in the rows above. The D = 70
rows are the "pilot" on the full circuit. Timings for D = 52–70 are from two
amplitudes run in parallel with another 4-thread job.

## Scaling and the D = 70 extrapolation

### Where the fidelity goes (`mpsprofile`, one bitstring, fid_est after every qubit)

| D | χ | j | first truncating qubit q0 | kept weight at q0 | loss rate after (nats/qubit) | final fid_est | s (1 thread) |
|---|---|---|---|---|---|---|---|
| 24 | 32 | 1 | 10 | 0.5000 | 0.069 | 0.0110, 0.0115 (2 seeds) | 0.2 |
| 28 | 64 | 1 | 12 | 0.5000 | 0.069 | 0.0100, 0.0101 | 0.9 |
| 32 | 128 | 1 | 14 | 0.5000 | 0.069 | 0.0114, 0.0112 | 5.5 |
| 36 | 256 | 1 | 25 | 0.8536 | 0.069 | 0.0589, 0.0593 | 33 |
| 40 | 512 | 1 | 17 | 0.5000 | 0.069 | 0.0151, 0.0151 | 263 |
| 44 | 512 | 2 | 12 | 0.5000 | 0.256 | 5.3e-7 | 522 |
| 48 | 256 | 4 | 8 | 0.316 | 0.61 | 4.6e-17 | 158 |
| 56 | 256 | 6 | 8 | 0.164 | 0.86 | 2.4e-24 | 233 |
| 70 | 256 | 9 | 8 | 0.064 | 1.28 | 3.1e-35 | 358 |

The picture is simple and very regular:

- The first truncation happens at the qubit `q0` where the middle-cut
  rank first exceeds `χ`. The spectrum there is flat (Clifford), so it
  keeps `2^-j` for `j ≤ 1` (exactly 1/2 at `j = 1`). At D = 36 it keeps
  0.8536 = cos²(π/8), because a T gate makes the spectrum non-flat. At
  small χ and large D the rank overshoots `χ` by more than a bit at once,
  so less is kept (0.32, 0.16, 0.06 for D = 48, 56, 70 at χ = 256).
- After that, every further qubit drops a **constant fraction**. At `j = 1`
  that is 0.069 nats per qubit, identically for D = 24 to 40. The new qubit
  regenerates rank above `χ` and the truncation removes it again. The rate
  grows with `j`: about `0.069 j²` for `j ≤ 1` (this fits the `j = 0.42`
  and `j = 0.5` rows within their errors), then 0.26, 0.61, 0.86 and 1.28
  for `j = 2, 4, 6, 9`.
- Model (j ≲ 1.5): `ln F ≈ −j ln 2 − r(j) · (69 − q0)`. It reproduces the table: for
  example D = 40, `j = 0.5` predicts 0.29 against a measured fid_est of
  0.28 and F = 0.21 ± 0.11. The truncation estimate tracks the measured
  overlap F wherever F is above the ~1/k noise floor. It is slightly
  optimistic at `j = 0.42` for D = 28 and 32 (fid_est 0.37–0.38 against
  F = 0.30).
- D enters only through `q0` (≈ 0.44 D − 1 at `j ≈ 1`: 10, 12, 14, 17 for
  D = 24–40, with D = 36 an outlier at 25) and through `χ_ex = 2^⌊D/4⌋`.
  So **the χ needed for a given fidelity is a fixed fraction of the exact
  rank**: `χ(D, f) ≈ 2^(⌊D/4⌋ − j(f))`, with `j(f)` at most about 1 bit.

### D = 70

`χ_ex = 2^17` (`m = 35`; the two middle cuts reach `2^17`). With
`q0 ≈ 30 ± 5`, so 39 ± 5 truncating qubits, the model gives:

| target f | j(f) | χ | MPS size (complex64) | one `2χ × 2χ` SVD matrix |
|---|---|---|---|---|
| 0.5 | 0.40 ± 0.05 | 2^16.6 ≈ 99 k | 426 GiB | 294 GiB |
| 0.25 | 0.60 ± 0.06 | 2^16.4 ≈ 86 k | 366 GiB | 223 GiB |
| 0.1 | 0.81 ± 0.06 (model error: 0.7–0.9) | 2^16.2 ≈ 75 k | 317 GiB | 169 GiB |
| exact | 0 | 2^17 | 597 GiB | 512 GiB |

**Meet in the middle helps, but not enough.** The model also holds for
short chains, where few qubits truncate after `q0`. With D = 40 and
`q0 = 17`, n = 24 leaves 6 truncating qubits. The model predicts 0.33 at
`j = 1` (measured fid_est 0.34, F = 0.41 ± 0.09, k = 48) and about 0.1 at
`j = 1.5` (measured 0.097, F = 0.11 ± 0.07). For n = 30 (6 more truncating qubits) at `j = 2` it predicts the n = 24 value times `e^(−0.256·6)`, i.e. 0.0047 (measured 0.0047). At `j = 1.5` the drop from 0.097 to 0.035 (F = 0.11 ± 0.07, at its noise floor) gives `r(1.5) ≈ 0.17`. At `j = 1` the model gives 0.34 · e^(−0.069·6) = 0.23 (measured fid_est 0.22, F = 0.18 ± 0.08). Splitting D = 70 at the middle
edge, as `cut_tensors_cpu` does, gives two half-sweeps of 35 qubits. Each
truncates only after its own `q0 ≈ 30`, so about 9 qubits truncate in all
(0–19), against 39 for the one-way sweep, at the price of two first
truncations. With `ln F ≈ −2 j ln 2 − r(j) N` this gives
`j(0.1) ≈ 1.1` (0.9–1.7), `χ ≈ 2^15.9` (2^15.3–2^16.1), i.e. **~250 GiB**
per half-boundary (130–300 GiB). Bounding the overlap of the two halves by
the product of their fidelities is an assumption. This variant is not
implemented: it gains 0.3–0.9 bit of χ, and it would still need ≥ 8× the
Mac's memory per half.

The dense exact register is 256 GiB. **At fidelity 0.1 the boundary MPS is
larger than the exact dense vector.** The MPS of a state whose time cuts are
all saturated costs about 2.3× the dense vector (597 vs 256 GiB), and
truncation to f = 0.1 only removes 0.8 bit of χ, i.e. 1.6 bits of memory.

What fits in 16 GB is `χ ≤ 2^12.5` (6 GiB of tensors plus SVD workspace),
`j ≈ 4.5–5`. Measured directly at D = 70 (one or two bitstrings, M1 Pro,
one thread per amplitude): χ = 128 → fid_est 8.8e-42 (58 s, 11 MiB);
χ = 256 → 5.7e-35 (356 s, 41 MiB); χ = 512 → 4.1e-30 (2385 s, 147 MiB).
At j = 5 (χ = 2^12 at D = 70) the measured truncation estimate is 1.2e-20 at D = 52 and 2.6e-20 at D = 56. At fixed j only the number of qubits after q0 changes with D, and q0 ≈ 13 for this χ, so D = 70 at χ = 2^12–2^12.5 should land at about 1e-18 to 1e-20 (estimated, not run). **That is not a usable
fidelity.**

Time, had the memory existed: the measured per-amplitude time at D = 70
grows ×6–7 per doubling of χ (58 → 356 → 2385 s). Seven more doublings to
χ ≈ 2^16.2 give **~2e9–8e9 s per amplitude** on one M1 Pro core (2e8 s if
10 cores were perfectly used). For comparison, the exact dense sweep is
estimated at 1.6 h per amplitude (if 256 GiB existed).

Sampling budget for a 3σ XEB result, `N ≈ 9/(f − 0.044)²`: 2870 samples at
f = 0.1, 212 at 0.25, 43 at 0.5, times at least a few amplitudes per sample
(rejection sampling against a Porter–Thomas envelope needs ~e). Even the
most favourable point (f = 0.5, 43 samples × ~3 amplitudes × ≥1e9 s) is
beyond 1e11 CPU-seconds and needs ~400 GiB. **The boundary-MPS route is
hopeless for this circuit**, on a laptop and on a cluster alike: it is
dominated by the exact dense sweep in both memory and time.

### Why (and what it rules out)

The boundary is the left half of the circuit contracted against `<x_0..x_e|`.
As a function of the `m` bond bits it behaves like a random stabilizer
state that a few T gates have slightly tilted. Every time cut is maximally
entangled, and the spectrum is flat up to the T-gate tilt. Truncation is
therefore a projection that keeps a fraction `2^-j` of a flat spectrum. It
does not compress anything, and because each new qubit scrambles the kept
subspace again, the loss compounds along the chain. The same argument
applies to any time-ordered or mixed dense/MPS boundary. The memory of a
compressed boundary is set by the middle-cut rank, and that rank cannot drop
below `2^(D/4 − 1)` without losing most of the fidelity.

## Next steps

- **Clifford-augmented boundary.** The boundary's entanglement is mostly
  stabilizer entanglement, which a Clifford frame on the `D/2` bond bits
  could remove for free (CAMPS applied to the boundary rather than to the
  state). Only the T-induced part would then cost bond dimension. How much
  that is for the roughly 234 of the 468 T gates that sit in a half-chain
  is unknown. It is the one variant of this idea that the measurements
  above do not rule out.
- Otherwise the chain sweep's options are the exact 256 GiB register
  (distributed, as in Manabe et al.) or nothing. Approximation by slicing
  (F = fraction kept, measured exactly) and by boundary truncation (this
  note) both fail to shrink the memory at useful fidelity.
- Not done: amplitudes for IBM's published samples. At the χ that fits in
  16 GB their fidelity would be ~1e-20 or less, so the XEB diagnostic carries no
  information.

## Reproduce

```text
cargo test --release --test chain_mps
cargo run --release --example chain_sweep -- dumpcut --n 70 --d 40 --e 34 --k 1 --out d40
python3 research/chain-sweep/boundary_spectra.py d40_0_L.bin
cargo run --release --example chain_sweep -- mpsfid --n 70 --d 32 --k 256 --seed 5 --chis 128,192,256
cargo run --release --features metal --example chain_sweep -- exactamps --n 70 --d 40 --k 60 --seed 7 --backend metal > a.txt
cargo run --release --example chain_sweep -- mpsfid --n 70 --d 40 --amps a.txt --chis 512
cargo run --release --example chain_sweep -- mpsprofile --n 70 --d 70 --chi 256 --seed 3
cargo run --release --example chain_sweep -- mpsfid --n 70 --d 70 --k 2 --seed 7 --noexact --chis 128,256,512
```
