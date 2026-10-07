# DRAFT: not submitted. Two issue bodies follow the tracker's
# `01-submission-observable-estimations.yml` template, one per circuit instance.

**Convention note (read before filing).** Every existing entry for these circuits records the
conserved charge **Q** in `observableValue`: #149 QPU, #202 PP_CPU, #227 MPS, #250 PP. Q is
exactly 60 for any noiseless simulation, so it carries no information. The physically
meaningful numbers are n_f(20) (meson − SCV, as defined in #149/#250) and the vacuum's
staggered occupation stag_SCV(20).

Two options for draft A:

- **A1 (follows existing practice).** Record Q = 60 in the field, and give n_f and stag_SCV
  with error bars in the method text.
- **A2 (gives the field real content).** Put n_f(20) in the field. Ask the maintainers first
  whether they'll accept n_f as the recorded observable.

Draft B (SCV) uses stag_SCV. No observable has been defined for the SCV instance yet, so it
needs the same confirmation.

---

## Draft A: meson instance

### Name

SU(2) LSH hadron dynamics as free fermions + exact 2nd-order interaction (Wick PT)

### Circuit

su2_hadron_dynamics_lsh_x_100_meson

### Observable value

A1: `60.0000000` (Q, meson circuit; exact by construction, deviation < 1e-11)
A2 (alternative): `0.11686` (n_f at step 20)

### Error bound (low)

A1: `0` (Q is conserved exactly by the method).
A2: `-0.0001`

### Error bound (high)

A1: `0`. A2: `+0.0001`

### Method

Free-fermion (Gaussian) simulation + exact second-order perturbation theory in the inter-chain coupling

### Method proof

**Main result (step 20).**

| quantity | value |
|---|---|
| n_f(20) = Σ_r (−1)^r [(n(2r)+n(2r+1))_meson − (…)_SCV] | **0.11686 ± 0.00010** |
| stag_SCV(20) = Σ_r (−1)^r (n(2r)+n(2r+1))_SCV | **−2.62385 ± 0.00020** |
| Q (meson), Q (SCV) | 60 (exact; deviation < 1e-11) |

These use the observable definitions of #149/#250. stag_SCV is equivalent to the vacuum's
total staggered fermion number Σ_r n_f^SCV(r) = 60 + stag_SCV = 57.37615 ± 0.00020.

**Structure of the circuit.** After undoing the SWAP network, the 120 qubits form two
60-site chains, i and o. The fused gates fall into three groups:

- **2360 nearest-neighbour hopping blocks** within one chain. Each is an exact
  number-conserving free-fermion (Gaussian) gate: the residual on the check is ≤ 3.3e-16.
- **Single-qubit Z phases.**
- **1200 diagonal on-site blocks.** These couple the two chains at the same rung only, through
  a phase exp(i g n_i(r) n_o(r)) with g = 0.010.

At g = 0 the circuit is exactly two free-fermion chains, simulated exactly with 60×60
correlation matrices.

**Method.** We expand in the coupling g:

- **Second order.** a2 is computed exactly in the interaction picture:
  - We write U = F_free Π_k (1 + c_k P_k), with P_k = n_i(r_k,t_k) n_o(r_k,t_k) as a
    Heisenberg-picture operator and c_k = e^{ig}−1.
  - Every correlator factorises into chain-i × chain-o free-fermion correlators.
  - These are evaluated with Wick's theorem as 1200×1200 matrices, with no sampling or
    truncation.
- **First order.** a1 is exactly zero, at all steps and for both circuits. All odd orders
  vanish numerically, to 1e-14. So the result free + a2 is correct up to O(g⁴).
- **Mean field is not enough.** Time-dependent Hartree is wrong at O(g²), by 5e-4 in n_f(20)
  and 7e-3 in stag_SCV(20).

**Validation.**

- a2 matches exact-statevector finite differences in g on 10–12-qubit windows to all printed
  digits.
- a2 matches full 120-qubit ladder-MPS TEBD. The comparison uses TEBD(g) − TEBD(0), so the
  truncation error cancels; the difference is converged in χ to 1.5e-8.
  - stag_SCV: agreement ≤ 2e-6 through step 9 (χ = 256).
  - n_f: agreement ≤ 4e-7 through step 10 (χ = 128).
  - The residual is the O(g⁴) term.
- The pipeline reproduces the published n_f at steps 1–3 (1.825336, 1.347301, 0.690350).

**Error bar.** It is set by the neglected O(g⁴) remainder:

- **Full system, steps ≤ 10.** A TEBD scan with g × {1, 2, 4, 8} shows the remainder scales
  exactly as g⁴, with |a4| ≤ 4e-6 in stag_SCV.
- **Windows to step 20.** Exact statevector on 6–12-site windows with the same g scan, run to
  step 20, gives a per-site remainder ≤ 2e-6. It shows no growth with window size, and the
  g⁶ terms are negligible even at 8× coupling.
- **Bounds.** We bound the remainder by a worst-case coherent sum:
  - stag_SCV: 60 sites × 2e-6 = 1.2e-4, quoted as ±2e-4.
  - n_f: 24 light-cone sites × 2 × 2e-6, quoted as ±1e-4. The two circuits are identical
    outside the meson's light cone.
- **Measured size.** The residuals we actually measure are 10–100× below these bounds.

**Caveat.** The step-20 bound uses small exact windows and full-system checks up to step ~10.
There is no full-system exact reference at step 20.

**Comparison (after the fact).** #250 reports n_f(20) = 0.1169 ± 0.0007, and 0.116917 at
atol = 1e-7. That is consistent with ours to 5e-5.

**Per-step values.** PT2, with the same bars at every step:

| step | n_f | stag_SCV |
|---:|---:|---:|
| 1 | 1.825336 | −54.845406 |
| 2 | 1.347301 | −40.722478 |
| 3 | 0.690350 | −21.265902 |
| 4 | 0.021282 | −1.358549 |
| 5 | −0.498568 | 14.254973 |
| 6 | −0.758322 | 22.278867 |
| 7 | −0.726334 | 21.713781 |
| 8 | −0.454570 | 14.002762 |
| 9 | −0.058119 | 2.455765 |
| 10 | 0.323064 | −8.846881 |
| 11 | 0.567767 | −16.318932 |
| 12 | 0.608895 | −17.908604 |
| 13 | 0.450079 | −13.607112 |
| 14 | 0.159266 | −5.295247 |
| 15 | −0.157519 | 3.989935 |
| 16 | −0.392936 | 11.107720 |
| 17 | −0.473363 | 13.835759 |
| 18 | −0.381026 | 11.537197 |
| 19 | −0.156713 | 5.276803 |
| 20 | 0.116863 | −2.623850 |

**Code.** `<link to repo/gist — TODO>`

- `gauss.py`: decoding and free-fermion simulation.
- `wick_pt2.py` and `pt2.py`: exact O(g²).
- `tebd_lam.py`, `window_lam.py`, `lam_scan.py`: validation and error bar.
- `ERRORBAR.md`: full derivation and per-step tables.

All of it is numpy only.

### Quantum runtime (seconds)

_No response_

### Classical runtime (seconds)

15

(Free + exact second order for both circuits at step 20, on one core. The validation runs
are not included.)

### Compute resources (quantum)

_No response_

### Compute resources (classical)

1 core of a shared 4-vCPU Linux VPS, numpy

### Notes

n_f(20)=0.11686±1e-4; stag_SCV=−2.62385±2e-4

### Authors

`<TODO>`

### Institutions

`<TODO>`

---

## Draft B: SCV instance

### Name

SU(2) LSH vacuum (SCV) as free fermions + exact 2nd-order interaction (Wick PT)

### Circuit

su2_hadron_dynamics_lsh_x_100_SCV

### Observable value

`-2.62385` (stag_SCV at step 20: Σ_r (−1)^r ⟨n(2r)+n(2r+1)⟩, with n(w) = (1−⟨Z_w⟩)/2. This is
equivalent to Σ_r n_f^SCV(r) = 60 + stag_SCV = 57.37615.)

### Error bound (low)

`-0.0002`

### Error bound (high)

`+0.0002`

### Method

Free-fermion (Gaussian) simulation + exact second-order perturbation theory in the inter-chain coupling

### Method proof

Same as draft A; refer to it there.

- **Contributions to stag_SCV(20) = −2.62385 ± 0.0002.**
  - Free part: −2.6410778.
  - Exact O(g²) correction: +0.0172276.
  - O(g) and all odd orders vanish.
  - O(g⁴) remainder: bounded by 1.2e-4 (60 sites × 2e-6 per site), quoted as ±2e-4.
- **Validation.** Against full-system TEBD through step 9 (agrees to ≤ 2e-6), and against
  exact windows to step 20.
- **Pauli propagation is not converged for this quantity.** atol = 1e-4 / 1e-5 / 1e-6 gives
  −2.6355 / −2.6375 / −2.6277, a spread of 1e-2.

### Classical runtime (seconds)

8

### Compute resources (classical)

1 core of a shared 4-vCPU Linux VPS, numpy

### Notes

Q_SCV = 60 exactly; vacuum stag occupation, see method

### Authors / Institutions

`<TODO>`
