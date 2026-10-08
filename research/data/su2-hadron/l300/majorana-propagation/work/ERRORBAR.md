# Error bar for the near-free-fermion simulation (step 20)

Everything here comes from our own calculations: exact free fermions, exact second-order
perturbation theory, TEBD, and exact statevector on windows. Pauli-propagation (PP) numbers
from other groups are used only for the comparison at the end. They play no part in setting
the bar.

## Result

| observable (step 20) | free (g=0) | Hartree | **PT2 = free + a2** | error bar |
|---|---|---|---|---|
| n_f = stag_meson − stag_SCV | 0.1180375 | 0.117363 | **0.1168633** | **± 1.0e-4** |
| stag_SCV = Σ_r (−1)^r (n_i(r)+n_o(r)) | −2.6410778 | −2.630801 | **−2.6238502** | **± 2.0e-4** |
| stag_meson | −2.5230403 | −2.513439 | −2.5069868 | ± 2.0e-4 |
| Q (either circuit) | 60 (to 1e-12) | 60 | 60 (a2 < 1e-12) | exact |

## 1. Hartree is the wrong correction

The ladder is two free-fermion chains, coupled only through on-site rung phases
exp(i g n_i(r) n_o(r)) with g = 0.01 (1200 vertices). Expand ⟨O⟩(λ) in the coupling scale λ
(g → λg). Then:

- **The first-order coefficient a1 is exactly zero.** It is below 1e-16 at every step, for
  both circuits. This holds even though time-dependent Hartree is first-order exact, because
  its first-order term is zero as well.
- **The whole interaction effect is O(g²).** Hartree captures only part of it. At step 6,
  for example, the Hartree shift in stag_SCV is −0.0035 while the true shift is −0.0131.
- **All odd orders vanish numerically.** Exact windows give |E(λ) − E(−λ)| < 1.5e-14 at
  every step. Full-system TEBD at χ=64 gives |E(1) − E(−1)| ≤ 1.5e-13 for steps ≤ 5 and 1.3e-11 at step 6.
  After that it grows only with truncation error.
  Hartree is even in λ to 1e-14. We don't have a proof of the underlying (presumably
  antiunitary) symmetry; we rely only on the numerical checks.

So Hartree's error is O(g²) and it grows with depth (table below). At its worst it reaches
0.030 in stag_SCV (step 17) and 1.3e-3 in n_f. At step 20 the Hartree error is 7.0e-3
(stag_SCV) and 5.0e-4 (n_f).

## 2. Exact second order (PT2)

`wick_pt2.py`, with the circuit adapter in `pt2.py`, computes a0, a1 and a2 exactly.

- **Method.** In the interaction picture, U = F_tot Π_k (1 + c_k P_k) with
  P_k = n_i(r_k,t_k) n_o(r_k,t_k) and c_k = e^{iλg_k} − 1. This is exact, since P_k² = P_k.
  Each correlator factorises into chain-0 × chain-1 free-fermion correlators. These are
  evaluated with Wick's theorem as 1200×1200 matrices, with no sampling and no truncation.
- **Runtime.** About 15 s for step 20 (both circuits, one core, numpy).

Checks:

- **Against exact statevector** on 5- and 6-site windows (10–12 qubits): a2 agrees with
  finite differences in λ to all printed digits (validate_pt2.py), and a1 = 0.
- **Against full-system TEBD.** TEBD(g) − TEBD(0) at χ=128 cancels the truncation error.
  - stag_SCV: agrees with a2 to ≤ 1.7e-6 through step 9 and 4.4e-6 at step 10.
  - n_f: agrees to ≤ 4e-7 through step 10.
  - Steps 11–12: the agreement is about 1e-5, matching the χ-convergence of TEBD itself.
  - The step ≤ 10 differences are the O(g⁴) term (see §3), not a PT2 error.
  - Hartree − TEBD is up to 2.3e-2 in stag_SCV and 1.1e-3 in n_f over the same steps.
- **Conservation.** Q is conserved: a2[Q] < 1e-12.
- **Short-depth match.** The pipeline reproduces the published short-depth n_f values
  (1.825336, 1.347301, 0.690350 at steps 1–3), which were computed independently from the
  QASM.

## 3. Bound on the remainder O(g⁴) at step 20

Write R = E − (a0 + a2). Its leading term is a4 g⁴ (λ⁴ in the scan).

**(a) Full system, steps ≤ 10.** TEBD λ-scan at χ=64 with λ ∈ {0, 1, 2, 4, 8}, ±1 for the
parity check (lam_scan.py):

- R(λ) follows λ⁴ exactly: R(2)/R(1) = 16 and R(4)/R(1) = 256. The fitted a6 term is
  negligible even at λ=8.
- |a4| ≤ 3.7e-6 for stag_SCV through step 10.
- The n_f combination (meson − SCV) is ≤ 2.3e-7.
- |a4/a2| grows roughly linearly in depth, reaching about 2.3e-4 at step 10.

**(b) Exact statevector to step 20 on windows** of L = 6, 8, 10, 12 sites (12–24 qubits).
Gates crossing the window are dropped; the scan uses λ ∈ {0, ±1, 2, 4, 8} (window_lam.py):

- Per-site |R| ≤ 2.0e-6 at every step ≤ 20 in every window, for both circuits. It does not
  grow with L (max 2.0e-6, 1.7e-6, 1.2e-6, 1.2e-6 for L = 6, 8, 10, 12).
- Per site, |R/a2| ≤ 2e-3.
- Staggered window sums of R at step 20 are ≤ 6e-6.
- The n_f combination of window residuals is ≤ 5e-6.
- The λ⁶ coefficients are below 1.5e-9, so the series is well behaved even at 8× coupling.

**Bounds used:**

- **stag_SCV.** Assume all 60 sites carry the worst per-site residual with the same sign:
  60 × 2.0e-6 = 1.2e-4. We quote ±2e-4. A realistic estimate is about 1e-5, from the
  |a4/a2| ~ 1e-3 window ratio times a2(20) = 0.017. The bound covers that by more than 10×.
- **n_f.** Outside the meson's light cone the two circuits have identical local dynamics, so
  their residuals cancel. Inside it (24 sites where the free |Δn| > 1e-6 at step 20), each
  site is worst-case ≤ 2 × 2.0e-6. That gives 24 × 4e-6 ≈ 1e-4, so we quote ±1e-4. The
  measured window values (≤ 5e-6) and full-system values (≤ 2.3e-7 at step ≤ 10) are 20× or
  more smaller.
- **Other errors.** Free-fermion and Wick arithmetic are float64, below 1e-11.

## 4. Why the bar is conservative, and what could undermine it

- **Conservative choices.**
  - The bar is a worst-case coherent sum of the largest per-site residual seen anywhere.
  - Measured residuals are 10–100× smaller.
  - The full-system TEBD at the depths where it is exact agrees with the window-derived
    magnitude.
- **Window size.** The window evidence comes from small open windows (≤ 12 sites). At step
  20 the light cone of a site is about ±12 sites, so the windows truncate it. The per-site
  bound is an empirical statement that the residual doesn't grow with L between 6 and 12,
  not a proof.
- **Reach of the full-system check.** The full-system exact check of the λ⁴ scaling reaches
  only step ~10. The raw ladder MPS loses fidelity beyond step ~12 (needs χ ≳ 500 at step
  20).
- **Odd orders.** Their vanishing is established numerically (to 1e-14), not by a proof.
  The bar doesn't depend on it: an O(g³) term would have shown up in R(λ) and in
  E(λ) − E(−λ), and it doesn't.
- **Observable choice.** n_f and stag_SCV are not the tracker's recorded observable. The
  tracker records Q, which is exactly 60 for any noiseless simulation.

## 5. Comparison with Pauli propagation (after the fact)

- **n_f.** Issue #250 (monoprop) reports n_f(20) = 0.1169 ± 0.0007, with atol=1e-7 →
  0.116917 and 3e-7 → 0.116910. Ours is 0.1168633 ± 0.0001. The PP central value sits
  5e-5 above ours, which is inside both bars. The PP sequence is still drifting down as atol
  decreases.
- **stag_SCV.** Our own PP runs (NOTES) give −2.6355 / −2.6375 / −2.6277 at
  atol = 1e-4 / 1e-5 / 1e-6. They are not converged at the 1e-2 level. Ours is −2.62385 ± 2e-4.

## Files

- `wick_pt2.py`: reusable exact O(g²) PT for two free-fermion blocks coupled by
  density-density phases. Input is a generic gate list; there is an adapter from
  `gauss.blocks`.
- `pt2.py`: the same, written directly for the circuit (produced `pt2.json`).
- `validate_pt2.py`, `sv_lam.py`: statevector validation with λ-scaled interaction.
- `tebd_lam.py`: ladder TEBD with λ-scaled interaction. Outputs are in `tebd_runs/`.
- `lam_scan.py`: full-system λ⁴ scaling (`lam_scan.json`).
- `window_lam.py`: exact windows with λ scan to step 20 (`window_runs/`).
- `hartree_lam.py`: Hartree with λ scaling.
- `analyze.py`: builds `RESULTS_TABLES.md` and `results_summary.json`.

## Appendix: per-step tables (from RESULTS_TABLES.md, regenerate with `python3 analyze.py`)

TEBD ref uses χ=256 for SCV steps ≤ 9 and χ=128 otherwise (meson χ=128). Beyond step 12, TEBD is not converged; there, |Hartree − PT2| stands in for the Hartree error (step 13–20 maximum: 3.0e-2 for stag_SCV and 1.3e-3 for n_f; at step 20: 7.0e-3 and 5.0e-4).


## stag_SCV

| step | free | Hartree | PT2 | TEBD ref | χmax | Hartree − TEBD | PT2 − TEBD | χ-conv | trunc (raw) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -54.845406 | -54.845406 | -54.845406 | -54.845406 | 256 | +0.0e+00 | -2.8e-14 | 0.0e+00 | -1.4e-14 |
| 2 | -40.722018 | -40.722420 | -40.722478 | -40.722478 | 256 | +5.9e-05 | -3.8e-09 | 0.0e+00 | -5.8e-12 |
| 3 | -21.263602 | -21.265281 | -21.265902 | -21.265902 | 256 | +6.2e-04 | -5.6e-08 | 0.0e+00 | +7.9e-11 |
| 4 | -1.352675 | -1.356027 | -1.358549 | -1.358548 | 256 | +2.5e-03 | -2.7e-07 | 0.0e+00 | +2.0e-09 |
| 5 | 14.265179 | 14.260947 | 14.254973 | 14.254973 | 256 | +6.0e-03 | -7.3e-07 | 0.0e+00 | +1.3e-08 |
| 6 | 22.291968 | 22.288506 | 22.278867 | 22.278868 | 256 | +9.6e-03 | -1.3e-06 | 2.2e-12 | +4.7e-08 |
| 7 | 21.725907 | 21.724870 | 21.713781 | 21.713783 | 256 | +1.1e-02 | -1.7e-06 | 4.1e-11 | +1.3e-07 |
| 8 | 14.008750 | 14.011070 | 14.002762 | 14.002763 | 256 | +8.3e-03 | -1.1e-06 | 3.2e-10 | +2.8e-07 |
| 9 | 2.451309 | 2.456707 | 2.455765 | 2.455764 | 256 | +9.4e-04 | +9.0e-07 | 1.5e-08 | +5.0e-07 |
| 10 | -8.862955 | -8.856102 | -8.846881 | -8.846885 | 128 | -9.2e-03 | +4.4e-06 | 7.9e-06 | -6.1e-05 |
| 11 | -16.343357 | -16.337523 | -16.318932 | -16.318943 | 128 | -1.9e-02 | +1.1e-05 | 2.0e-05 | -1.2e-03 |
| 12 | -17.934088 | -17.931641 | -17.908604 | -17.908637 | 128 | -2.3e-02 | +3.3e-05 | 5.4e-06 | -7.2e-03 |
| 13 | -13.624578 | -13.626835 | -13.607112 | — | — | — | — | — | — |
| 14 | -5.297206 | -5.303854 | -5.295247 | — | — | — | — | — | — |
| 15 | 4.006156 | 3.997112 | 3.989935 | — | — | — | — | — | — |
| 16 | 11.138229 | 11.129872 | 11.107720 | — | — | — | — | — | — |
| 17 | 13.870732 | 13.866134 | 13.835759 | — | — | — | — | — | — |
| 18 | 11.563909 | 11.564978 | 11.537197 | — | — | — | — | — | — |
| 19 | 5.284127 | 5.290839 | 5.276803 | — | — | — | — | — | — |
| 20 | -2.641078 | -2.630801 | -2.623850 | — | — | — | — | — | — |

## n_f

| step | free | Hartree | PT2 | TEBD ref | χmax | Hartree − TEBD | PT2 − TEBD | χ-conv | trunc (raw) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.825336 | 1.825336 | 1.825336 | 1.825336 | 128 | +0.0e+00 | +3.8e-14 | 0.0e+00 | -7.1e-15 |
| 2 | 1.347278 | 1.347298 | 1.347301 | 1.347301 | 128 | -2.9e-06 | +1.9e-10 | 0.0e+00 | +6.5e-13 |
| 3 | 0.690237 | 0.690319 | 0.690350 | 0.690350 | 128 | -3.0e-05 | +2.7e-09 | 0.0e+00 | -1.3e-11 |
| 4 | 0.021001 | 0.021162 | 0.021282 | 0.021282 | 128 | -1.2e-04 | +1.2e-08 | 6.6e-14 | -3.2e-10 |
| 5 | -0.499040 | -0.498842 | -0.498568 | -0.498568 | 128 | -2.7e-04 | +3.2e-08 | 9.5e-12 | -2.1e-09 |
| 6 | -0.758898 | -0.758744 | -0.758322 | -0.758322 | 128 | -4.2e-04 | +5.1e-08 | 3.5e-10 | -7.9e-09 |
| 7 | -0.726812 | -0.726787 | -0.726334 | -0.726334 | 128 | -4.5e-04 | +4.9e-08 | 2.2e-09 | -3.0e-08 |
| 8 | -0.454707 | -0.454864 | -0.454570 | -0.454570 | 128 | -2.9e-04 | -2.7e-09 | 2.0e-09 | -1.9e-07 |
| 9 | -0.057730 | -0.058058 | -0.058119 | -0.058119 | 128 | +6.1e-05 | -1.2e-07 | 1.5e-07 | -8.7e-07 |
| 10 | 0.323996 | 0.323587 | 0.323064 | 0.323065 | 128 | +5.2e-04 | -4.1e-07 | 2.5e-06 | +1.0e-05 |
| 11 | 0.569045 | 0.568697 | 0.567767 | 0.567769 | 128 | +9.3e-04 | -2.4e-06 | 1.2e-05 | +1.9e-04 |
| 12 | 0.610146 | 0.609999 | 0.608895 | 0.608910 | 128 | +1.1e-03 | -1.5e-05 | 9.7e-06 | +1.1e-03 |
| 13 | 0.450873 | 0.451008 | 0.450079 | — | — | — | — | — | — |
| 14 | 0.159282 | 0.159681 | 0.159266 | — | — | — | — | — | — |
| 15 | -0.158348 | -0.157806 | -0.157519 | — | — | — | — | — | — |
| 16 | -0.394362 | -0.393868 | -0.392936 | — | — | — | — | — | — |
| 17 | -0.474876 | -0.474622 | -0.473363 | — | — | — | — | — | — |
| 18 | -0.382019 | -0.382123 | -0.381026 | — | — | — | — | — | — |
| 19 | -0.156704 | -0.157162 | -0.156713 | — | — | — | — | — | — |
| 20 | 0.118038 | 0.117363 | 0.116863 | — | — | — | — | — | — |

Full-system λ-scan (χ=64), stag residual R(λ) = E(λ) − E(0) − a2 λ²:
```
SCV lams [1, 2, 4, 8]
SCV step  1 a2 -2.753e-14 R(1) +2.75e-14 R(2) +1.10e-13 R(4) +4.41e-13 R(8) +1.76e-12 | fit a4 +2.18e-15 a6 -2.74e-17 | E(1)-E(-1) +0.0e+00
SCV step  2 a2 -4.607e-04 R(1) +3.83e-09 R(2) +6.14e-08 R(4) +9.83e-07 R(8) +1.57e-05 | fit a4 +3.84e-09 a6 -1.21e-14 | E(1)-E(-1) +0.0e+00
SCV step  3 a2 -2.300e-03 R(1) +5.58e-08 R(2) +8.93e-07 R(4) +1.43e-05 R(8) +2.29e-04 | fit a4 +5.58e-08 a6 -6.79e-13 | E(1)-E(-1) -3.6e-15
SCV step  4 a2 -5.873e-03 R(1) +2.68e-07 R(2) +4.29e-06 R(4) +6.86e-05 R(8) +1.10e-03 | fit a4 +2.68e-07 a6 -6.67e-12 | E(1)-E(-1) -5.3e-15
SCV step  5 a2 -1.021e-02 R(1) +7.32e-07 R(2) +1.17e-05 R(4) +1.87e-04 R(8) +2.99e-03 | fit a4 +7.32e-07 a6 -3.12e-11 | E(1)-E(-1) +1.5e-13
SCV step  6 a2 -1.310e-02 R(1) +1.33e-06 R(2) +2.14e-05 R(4) +3.42e-04 R(8) +5.45e-03 | fit a4 +1.34e-06 a6 -9.10e-11 | E(1)-E(-1) +1.3e-11
SCV step  7 a2 -1.213e-02 R(1) +1.63e-06 R(2) +2.63e-05 R(4) +4.22e-04 R(8) +6.72e-03 | fit a4 +1.65e-06 a6 -1.70e-10 | E(1)-E(-1) +3.0e-10
SCV step  8 a2 -5.988e-03 R(1) +9.25e-07 R(2) +1.63e-05 R(4) +2.67e-04 R(8) +4.24e-03 | fit a4 +1.05e-06 a6 -1.54e-10 | E(1)-E(-1) +4.3e-09
SCV step  9 a2 +4.456e-03 R(1) -7.31e-07 R(2) -1.35e-05 R(4) -2.32e-04 R(8) -3.87e-03 | fit a4 -8.93e-07 a6 -8.10e-10 | E(1)-E(-1) -2.8e-08
SCV step 10 a2 +1.607e-02 R(1) +3.49e-06 R(2) -4.16e-05 R(4) -9.51e-04 R(8) -1.50e-02 | fit a4 -3.72e-06 a6 +9.73e-10 | E(1)-E(-1) -4.9e-09
meson lams [1, 2, 4, 8]
meson step  1 a2 +1.021e-14 R(1) -1.02e-14 R(2) -4.09e-14 R(4) -1.63e-13 R(8) -6.54e-13 | fit a4 -8.10e-16 a6 +1.02e-17 | E(1)-E(-1) +nan
meson step  2 a2 -4.376e-04 R(1) +3.64e-09 R(2) +5.83e-08 R(4) +9.34e-07 R(8) +1.49e-05 | fit a4 +3.65e-09 a6 -1.17e-14 | E(1)-E(-1) +nan
meson step  3 a2 -2.187e-03 R(1) +5.31e-08 R(2) +8.50e-07 R(4) +1.36e-05 R(8) +2.18e-04 | fit a4 +5.31e-08 a6 -6.46e-13 | E(1)-E(-1) +nan
meson step  4 a2 -5.592e-03 R(1) +2.56e-07 R(2) +4.09e-06 R(4) +6.54e-05 R(8) +1.05e-03 | fit a4 +2.56e-07 a6 -6.36e-12 | E(1)-E(-1) +nan
meson step  5 a2 -9.734e-03 R(1) +7.01e-07 R(2) +1.12e-05 R(4) +1.79e-04 R(8) +2.86e-03 | fit a4 +7.01e-07 a6 -2.98e-11 | E(1)-E(-1) +nan
meson step  6 a2 -1.253e-02 R(1) +1.28e-06 R(2) +2.06e-05 R(4) +3.29e-04 R(8) +5.24e-03 | fit a4 +1.29e-06 a6 -8.72e-11 | E(1)-E(-1) +nan
meson step  7 a2 -1.165e-02 R(1) +1.58e-06 R(2) +2.56e-05 R(4) +4.10e-04 R(8) +6.52e-03 | fit a4 +1.60e-06 a6 -1.65e-10 | E(1)-E(-1) +nan
meson step  8 a2 -5.851e-03 R(1) +9.29e-07 R(2) +1.65e-05 R(4) +2.68e-04 R(8) +4.26e-03 | fit a4 +1.05e-06 a6 -1.66e-10 | E(1)-E(-1) +nan
meson step  9 a2 +4.067e-03 R(1) -7.63e-07 R(2) -1.26e-05 R(4) -2.03e-04 R(8) -3.38e-03 | fit a4 -7.82e-07 a6 -6.73e-10 | E(1)-E(-1) +nan
meson step 10 a2 +1.514e-02 R(1) +1.36e-06 R(2) -4.14e-05 R(4) -8.94e-04 R(8) -1.42e-02 | fit a4 -3.49e-06 a6 +3.72e-10 | E(1)-E(-1) +nan
```
