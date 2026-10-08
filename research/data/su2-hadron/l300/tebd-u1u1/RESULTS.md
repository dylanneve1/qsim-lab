# RESULTS — su2-254 (SU(2) LSH hadron, l_i=300, g=0.226125), 120 qubits, 20 Trotter steps

**Method:** exact-gate U(1)xU(1) charge-conserving TEBD (TeNPy) on the 60-rung ladder (rung = 4-dim site, n_i,n_o
conserved separately), `tebd_sym.py`. The circuit is literally a brick-wall MPS circuit (even hops, odd hops,
on-site diagonal), so the only approximation is SVD truncation. χ = 256, 512, 1024, 2048, 2800; svd_min 1e-10.

**Observables:** stag = Σ_r (−1)^r (n_i(r)+n_o(r)); n_f = stag_meson − stag_SCV (brief's definition).

## Per-step results (best = χ=2800; err = max(|χ2800−χ2048|, |terr-extrapolation shift|), floor added from g=0 calibration)

| step | n_f | ±err | stag_SCV | ±err | n_f (free, g=0) | stag_SCV (free) | discarded wt (SCV, χ2800) |
|---|---|---|---|---|---|---|---|
| 1 | +3.650671 | 0e+00 | -54.845406 | 0e+00 | +3.6507 | -54.8454 | 0.0e+00 |
| 2 | +2.715205 | 0e+00 | -41.091820 | 0e+00 | +2.6957 | -40.7394 | 7.7e-19 |
| 3 | +1.480150 | 0e+00 | -23.068853 | 0e+00 | +1.3869 | -21.3586 | 5.0e-18 |
| 4 | +0.286461 | 0e+00 | -5.838538 | 0e+00 | +0.0608 | -1.6308 | 9.3e-18 |
| 5 | -0.589960 | 0e+00 | +6.669189 | 0e+00 | -0.9585 | +13.6778 | 2.0e-17 |
| 6 | -1.007995 | 0e+00 | +12.651646 | 0e+00 | -1.4507 | +21.2928 | 3.2e-17 |
| 7 | -0.973485 | 0e+00 | +12.426263 | 0e+00 | -1.3569 | +20.2801 | 5.7e-17 |
| 8 | -0.610328 | 0e+00 | +7.796420 | 0e+00 | -0.7873 | +12.1752 | 8.0e-17 |
| 9 | -0.107441 | 0e+00 | +1.219707 | 0e+00 | +0.0219 | +0.3755 | 1.2e-16 |
| 10 | +0.345667 | 0e+00 | -4.971765 | 0e+00 | +0.7877 | -10.9912 | 1.8e-16 |
| 11 | +0.618242 | 0e+00 | -9.171390 | 0e+00 | +1.2688 | -18.3536 | 2.5e-16 |
| 12 | +0.671348 | 4e-14 | -10.799762 | 1e-13 | +1.3365 | -19.7392 | 1.2e-15 |
| 13 | +0.551077 | 8e-13 | -10.222008 | 3e-12 | +1.0065 | -15.2594 | 6.2e-14 |
| 14 | +0.348704 | 2e-11 | -8.341120 | 9e-11 | +0.4227 | -6.9159 | 2.4e-12 |
| 15 | +0.152739 | 3e-10 | -6.097735 | 1e-09 | -0.1975 | +2.1773 | 6.2e-11 |
| 16 | +0.018138 | 2e-09 | -4.135817 | 1e-08 | -0.6395 | +8.8777 | 1.2e-09 |
| 17 | -0.038691 | 1e-06 | -2.741080 | 5e-06 | -0.7619 | +11.0427 | 1.7e-08 |
| 18 | -0.028144 | 1e-06 | -1.962374 | 5e-06 | -0.5386 | +8.1729 | 1.9e-07 |
| 19 | +0.027170 | 1e-06 | -1.751897 | 5e-06 | -0.0617 | +1.4823 | 1.7e-06 |
| 20 | +0.102851 | 2e-06 | -2.032802 | 9e-06 | +0.4963 | -6.5997 | 1.2e-05 |

Step 20: **n_f = 0.10285 ± 1e-5**, **stag_SCV = −2.03280 ± 3e-5** (quoted bars are ~5x the χ2048→χ2800 change,
1.7e-6 and 8.7e-6, to be conservative). Free values for comparison: 0.4963, −6.5997 — interactions matter at O(1).
Steps 1–16 are converged to ≤1e-8 (discarded weight ≤1e-9).

## Validation
* Code vs exact statevector (exact.py, number-conserving sectors): window TEBD == exact to 5e-14 (L=10, all 20 steps)
  and 5e-13 (L=14 window, χ=8192, all 20 steps). Truncated window run χ=128 errs 2e-2 at step 20 (terr 0.18) — shows
  error scale ~ discarded weight.
* Full-system TEBD vs exact L=12/14 windows, central rungs 28–31: 1e-14 at steps 1–2; at steps 3–8 the residual
  (L=12: 5e-8…8e-4; L=14: 2e-10…4e-5) shrinks ~100x per +2 sites => it is the window's hard-wall error, TEBD is exact
  there (discarded weight <1e-16 through step 11).
* g=0 control vs exact free fermions (gauss.py), step 16/18/20:
  χ=1024: stag err 4e-6/3e-4/1e-3, n_f err 7e-7/6e-5/5e-5 (terr 5.5e-3);
  χ=2048: stag err 1e-8/1e-6/2.2e-5, n_f err 3e-9/3e-7/5e-6 (terr 1.3e-4).
  The free state is MORE entangled than the interacting one (Smax 5.21 vs 4.81 nats at step 20; interacting discarded
  weight at χ2800 = 1.2e-5), so interacting χ=2800 errors should be below free χ=2048 errors (~2e-5).
* χ-convergence interacting, step 20: n_f 0.0994/0.1121/0.1032/0.10285/0.10285; stag −2.0299/−2.0939/−2.0341/−2.03279/−2.03280.

## Cost (this shared 4-vCPU VPS, OMP 2 threads, nice 10)
χ=1024: ~4 min, 0.39 GB; χ=2048: ~14 min, 1.0 GB; χ=2800: ~24 min, 1.8 GB (per circuit). Both circuits at χ=2800 ≈ 50 min.
16 GB Mac (M-series): χ=2048 in ~5-10 min, χ=4096 (~4 GB est.) in ~1 h — more than enough.

## Other routes evaluated (see NOTES.md)
G|MPS> natural-orbital: dead (natural occupations → 0.45/0.55, correlation entropy 17/20 bits on L=10).
Interaction picture: ~2.4 bits less entanglement, similar growth slope. Heisenberg MPO: worse than state TEBD.
Transverse/folded influence matrix: temporal entanglement ≈ spatial (T=4: 1.44 nats, χ_t≥128) and identical at g=0.

## Verdict
l_i=300 (θ=0.226) is classically easy at 20 steps: an off-the-shelf symmetric MPS reaches discarded weight 1e-5 and
observables to ~1e-5 in under an hour on 2 CPU threads. The physics is a Trotterized Hubbard chain, U/J≈1.5, Jt=3;
step-20 entanglement 4.8 nats needs χ~2000–3000. Earlier "χ≈1000–2000, VPS cannot hold that" was too pessimistic
because U(1)xU(1) block sparsity cuts memory/time by >10x.

## Cross-check vs issue's Pauli propagation (PP) and hardware (issue254.md §5–6)
Issue convention: N_X = Σ_r n_f^X(r) = 60 + stag_X; n_f = N_MID − N_SCV (= our n_f). Checked:
* Mapping: N_SCV(0)=0, N_MID(0)=4 by construction (decoded X prefix: MID moves both chains' rung-29 particle to 30);
  orientation/parity confirmed by N_SCV(20) agreeing with PP to 0.01 (a parity flip would give ~62). Decoded op list
  ends exactly at the step-20 marker (no gates dropped); SCV/MID gate lists identical op-by-op.
* Q conservation in our MPS: max|Q−60| over all 20 steps = 3e-13 (SCV), 3e-12 (MID) at χ=2800 (U(1)xU(1) exact).
* Early-step per-site densities vs exact L=12 window, every site outside the wall's causal cone: ≤3e-14 (steps 1–3).

Per-step N_X (χ=2800):
N_SCV: 5.15459 18.90818 36.93115 54.16146 66.66919 72.65165 72.42626 67.79642 61.21971 55.02823 50.82861 49.20024 49.77799 51.65888 53.90227 55.86418 57.25892 58.03763 58.24810 57.96720
N_MID: 8.80527 21.62338 38.41130 54.44792 66.07923 71.64365 71.45278 67.18609 61.11227 55.37390 51.44685 49.87159 50.32907 52.00758 54.05500 55.88232 57.22023 58.00948 58.27527 58.07005

| step 20 | N_SCV | N_MID | n_f | δQ_SCV | δQ_MID |
|---|---|---|---|---|---|
| TEBD χ=2800 (this work) | 57.96720 | 58.07005 | 0.10285 ± 1e-5 | 3e-13 | 3e-12 |
| PP atol 1e-4 | 58.64799 | 58.72830 | 0.08032 | 0.249 | 0.187 |
| PP atol 1e-5, weight≤20 | 57.95750 | 58.03731 | 0.07980 | 0.115 | 0.077 |
| hardware 3 Aug 2026 | 58.94984 | 59.04936 | 0.09952 | 0.195 | 0.269 |

Diagnosis of the 0.023 gap: it is PP truncation. PP is not converged: N_SCV/N_MID move by 0.69 between its last two
settings, toward our values (N_SCV 58.648 → 57.958 vs ours 57.967; N_MID 58.728 → 58.037 vs ours 58.070), its charge
is off by 0.08–0.12 (i.e. 4–5x larger than the n_f gap), and the 1e-5 run also imposes a Pauli-weight cutoff of 20 on a
120-qubit, 20-step circuit. Its n_f stability (0.0803→0.0798) reflects correlated SCV/MID truncation, not convergence
(the MID error −0.033 ≠ SCV error −0.010). Our side: monotone χ-convergence 0.10316/0.10285/0.10285 (χ 1024/2048/2800),
discarded weight 1.2e-5, exact Q, g=0 control matching exact free fermions to 5e-6 at χ=2048, exact-window agreement 1e-13.
Hardware (0.0995) happens to be within 0.003 of the converged value.
