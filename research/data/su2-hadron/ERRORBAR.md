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
- **Against full-system TEBD.** TEBD(g) − TEBD(0) cancels the truncation error. It agrees
  with a2 to 1e-6 through step 9 at χ=128 (table). The remaining difference is the O(g⁴)
  term, not a PT2 error.
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
