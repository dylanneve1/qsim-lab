# New ideas against IBM's doped-Clifford circuit on small hardware: sweep + probes

Agent agt_14307505, 2026-10-07. VPS only (prlimit 1.5 GB, nice, ≤ 2 threads).
Circuit: `../chain-sweep/nq70_depth70_checks27_doped.qasm` (tracker issue 228; 70 qubits,
open chain, depth 70, 468 T). Goal: XEB significantly > 0.044 from uncorrelated samples,
on a 16 GB laptop or the 7.7 GB VPS. Eleven routes were already closed (see PLAN.md).

**Bottom line: no route found that works at 16 GB.** Three new tools are quantified below.
Two of them are real memory↔time levers (dithered multi-run averaging, residue-number
arithmetic); both need ≥ ~28–34 GB per sweep and hundreds to thousands of Mac-hours.
The third closes every XEB-spoofing route with numbers.

## 1. XEB spoofing is dead here: the Walsh spectrum of p is flat (`paths.py`, `sa.py`, `validate_paths.py`)

Any sampler q has `XEB(q) = Σ_{S≠0} c_S d_S`, with `c_S = <Z_S>_ψ` and `d_S = <Z_S>_q`
(the Walsh–Fourier coefficients of p and q). Patch, light-cone, Pauli-path and low-degree
samplers, and marginal-based sequential samplers, all score by knowing some large `c_S`.

* Clifford part: the undoped output is uniform (X-rank 70), so `c_S = 0` for all S≠0.
  Every nonzero `c_S` comes from T branchings. A Pauli path from a Z-type string at t=0
  to a Z-type string at the end contributes `±2^{-a/2}`, with `a` = the number of
  anticommuting T it crosses. Over GF(2) the set of valid paths is
  `{J : supp(1_J) ⊆ supp(M 1_J)}`, with `a = |M 1_J|` and `S = S_M 1_J`. Here `M` is
  468×468, built from the Clifford tableau.
* Minimum `a` found (singles exhaustive, plus simulated annealing, 2 seeds):

| D (head truncation) | T | min a | per-path weight |
|---|---|---|---|
| 40 | 108 | 42 | 2^-21 |
| 63 | 219 | 88 | 2^-44 |
| 70 | 468 | 198 | 2^-99 |

  For a flat (Porter–Thomas) spectrum the typical `|c_S|` is `2^-35`. Every single path
  is far below that, so each `c_S` is an irreducible many-path sum.
* Exact check on 14–16-qubit windows (dense state vector, exact Walsh transform):
  - Full-depth window, qubits 20–35 (n=16, 186 T): `max|c_S| = 0.016 = 4.1 × rms` (rms = 2^-8).
    A Gaussian extreme over 65535 coefficients gives ~4.7×, so the spectrum is flat.
    The best single path has weight 2^-31, while the actual `|c_S|` on that S is ~1e-3 (many paths).
  - The structured regime exists only at shallow depth. In window 0–13 at D=30, a = 10 paths
    give `c_S` up to 0.41.
* At n=70: `max_S |c_S| ≈ 2^-35·sqrt(2 ln 2^70) ≈ 3e-10`. A sampler that fixes k parities
  scores `Σ_i |c_{S_i}| ≲ k·3e-10`. Reaching 0.03 needs ~2^30 exactly known coefficients,
  i.e. exact 30-qubit marginals. **Gao et al.-style spoofing (2112.01657), Pauli paths
  (2211.03999, 2407.16068) and patch samplers give XEB ≈ 0 on this circuit.**

## 2. Dithered multi-run averaging (new lever; probe: `--reps K`, format suffix `:dz`)

Rounding with unbiased errors (stochastic rounding `:sr`, or subtractive dither `:dz`,
where the dither is regenerated from a seed and costs no storage) gives an amplitude estimator
`l = a + noise`, with `E[l] = a` (the maps are linear). K runs with independent streams
average to `F_K = 1/(1 + (1/F_1 − 1)/K)`. Measured: n=70, tail window D=24, R=70
roundings (= the D=70 count), 100 uniform bitstrings:

| format | bits/comp | F_1 | K=2 | K=4 | K=8 | K=16 | prediction K=16 |
|---|---|---|---|---|---|---|---|
| int4:b64 (nearest, deterministic) | 4.25 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | — |
| int4:b64:sr | 4.25 | 0.14 | 0.30 | 0.40 | 0.56 | 0.75 ± 0.03 | 0.73 |
| int4:b64:dz | 4.25 | 0.33 | 0.51 | 0.70 | 0.82 | 0.89 ± 0.02 | 0.89 |
| int3:b16:h:dz | 3.5 | 0.016 | 0.03 | 0.15 | 0.25 | 0.37 ± 0.05 | ~0.3 |

The averaging law holds. It converts memory into time at a rate `K ≈ (1/F_1 − 1)·F/(1−F)`.
Since `F_1 = exp(−rR)` and r grows ×4 per bit lost, this is exponential:
* int3:b16:h (28 GiB = 30 GB, which just fits the Mac's RAM + ~20 GB free SSD). Minimising
  samples × runs gives F ≈ 0.5, K ≈ 40 and N ≈ 90 samples: ~3400 sweeps ≈ 5000 h at ~1.5 h per
  sweep (the M1 Pro compute floor for 3.7e14 amplitude-ops). SSD writes would be petabytes. **Not practical.**
* At 16 GB (≤ 2 bits/comp) `1/F_1 ~ 1e4–1e7`. **Dead.**
* Deterministic int4:b16:h (F=0.53, 39 GB, 1 run, ~80 samples) beats averaging int4:b64 (~200 runs).
  Averaging only pays when it buys a format that otherwise would not fit.
* int2:b16:h:dz produced one huge outlier (rms_rel 6e5). int2 is dead anyway; not chased.

## 3. Residue-number (RNS) exact sweep over Z[ζ8] (new; probe: `zomega.py`)

Every amplitude is `a = A/√2^k` with `A ∈ Z[ω]`, ω = e^{iπ/4}. Run the chain sweep with the
register stored modulo a prime ideal P of Z[ω] with √2 invertible (N(P) = 9, 17, 25, 41, …).
This is exact, carry-free and needs no rounding. A sweep mod P needs `2^35·log2 N(P)` bits
(13.6 GB for F_9, the smallest). Each channel gives `log2 N(P)` bits of A; CRT reconstructs a.
Channels are independent and embarrassingly parallel.

Needed bits = 2·(sde − n). Exact Z[ω] state vectors on 7 windows (n = 8–12, full depth,
31–176 T) give **sde − n = (0.36–0.42)·t**, about 0.40 per T gate (worst-case bound 1.5).
For t=468 that is **≈ 375 bits per amplitude**. Distinct prime ideals are scarce:
`Σ log2 N(P)` over ideals with N ≤ X is ≈ 1.44·X bits, so 375 bits needs channels up to
N ≈ 260 (8 bits, 32 GiB register).
* Mac (RAM + 20 GB SSD ≈ 31 GB → channels ≤ 7.2 bits): only **211 bits** available. **Infeasible.**
* With ≥ 34 GB per machine: F = 1 (exact) from ~55 channel-sweeps per amplitude, split across
  any number of machines (8 × 32 GB boxes replace one 256 GiB node). That is new, but it is not
  16 GB, and it is ~6× the f32 bit-volume.

## 4. Other ideas examined (short)

| idea | verdict |
|---|---|
| Noise shaping into ker(T_{i+1}) (store the register only modulo the next column's kernel) | Dead (argued, not measured): the register norm drops by exactly ½ bit per qubit (chain-sweep README), which is consistent with scaled-isometry columns, and the boundary has no stabilizers (exp/clifford-boundary). So there is no stabilizer-type kernel to exploit. |
| Rate-distortion limit at 16 GB | Shannon-optimal VQ: `F ≤ (1 − 2^{-2b})^70`. b = 1.86 (all 16 GB) gives F ≤ 0.004; b = 1.34 (11.5 GB usable) gives 7e-6; b = 3.6 (31 GB) gives 0.62. So "16 GB → F ≤ 0.01" is information-theoretic, not an engineering limit. |
| Top-K amplification of low-F amplitudes (argmax of K) | XEB ≈ F·(H_K − 1). F = 0.01 needs K ~ 2^14 amplitudes per sample. Dead. |
| Free open output bits | Only the last-swept qubit is free (2 amplitudes per sweep; ∝p choice gives XEB 1/3). Each further open bit doubles memory. |
| Trailing diagonal gates / light-cone trimming | 0 removable CZ, 0 removable T (checked). |
| Stabilizer TN + magic injection (2411.12482) | Same mechanism as CAMPS-OFD (≤ n = 70 T absorbed free); closed by exp/camps. |
| Clifford-disentangler learning (2609.27128, 2609.27565) | Needs T density ≲ 1–2 per qubit; here 6.7. Also a learning result, not a simulator. |
| LNN graph state + product-basis (Ising partition function) | Rank-width 32–35, the same wall. |
| Restricted-support (fixed prefix) sampling | XEB ≈ 1 but correlated samples. Not legit, and not cheaper anyway. |
| Out-of-core SSD | Already costed in FOLD.md; limited by free disk and SSD wear (~1 TB written per sweep). |

## Literature (sweep, 2026-10-07)

IBM 2607.25941 (v3 2026-09-02). Manabe/Gu/Pan 2608.13110 (v1 only). Citing works: 2608.15963
(Clifford obfuscation), 2609.13108 (spacetime mitigation), 2610.04666, 2603.09901 (survey).
No other classical response to issue 228 was found. Also relevant: 2112.01657 (Gao et al., XEB limits),
2211.03999 (Aharonov et al., noisy RCS), 2407.16068 (Pauli paths), 2411.12482 (STN + magic
injection), 2412.17209 (CAMPS), 2609.27128 / 2609.27565 (Clifford-scrambled product states,
t-doped learning), 2609.14252 (sum-over-Cliffords expectation values), 2606.01922 (path-sum
pruning), 1910.09534 (IBM's own secondary-storage argument, 2019).

## Files

`paths.py <qasm> D` builds M and S_M and lists single-branch paths. `sa.py D seed` anneals over valid branch sets.
`validate_paths.py <qasm> lo hi [D]` does the exact Walsh spectrum vs path weights on a window.
`zomega.py <qasm> lo hi` gives the exact Z[ω] state vector, sde and CRT bit sizes.
`examples/chain_sweep.rs lowprec --reps K` and the format suffix `:dz` are in `src/engines/chain_lowprec.rs`.
Command for the table in §2:
`chain_sweep lowprec --n 70 --d 24 --tail --gran qubit --k 100 --reps 16 --formats int4:b64,int4:b64:sr,int4:b64:dz,int3:b16:h,int3:b16:h:dz`.
