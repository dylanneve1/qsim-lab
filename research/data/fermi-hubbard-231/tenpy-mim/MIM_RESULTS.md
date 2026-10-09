# MIM_RESULTS — late times (t=4…6) via Heisenberg MPO + meet-in-the-middle (supersedes the "t=5,6 not converged" rows of RESULTS.md)

Research only, nothing posted. Same target as RESULTS.md: ideal output of the paper's 30-layer Trotter circuit, L=60, U=-2, Néel start, centre site i=29.

## 1. Headline
* **t=5 and t=6 are now converged to ~1e-4 (⟨n↑⟩) and 1e-4…2e-4 (⟨n↑n↓⟩)** for the centre site 29 with a cheap method (≤1.5 GB RSS, 10–75 min per point),
  where forward TEBD at χ=2048–3072 had spreads of 4e-2…7e-2. ⟨n↓⟩_29 = 1 − ⟨n↑⟩_29 (particle–hole symmetry; verified ≤1e-5 on the converged forward data
  up to t=4; not independently computed at t=5,6).
* Method: ⟨ψ0|U†OU|ψ0⟩ = ⟨ψ(t1)| O_H |ψ(t1)⟩ with t1 = k0 = 10 layers (t=2): forward TEBD state ψ(10) (χψ=256/512; exact to ≲1e-5 there), and
  O_H = (G_{10+1}…G_k)† O (G_{10+1}…G_k) evolved backwards as a vectorised MPO (d=16 doubled sites, charge q_ket−q_bra, U(1)² conserved) with the same Trotter
  gates, truncated at χO. The sandwich ⟨ψ|O_H|ψ⟩ is contracted site by site (left-to-right transfer, charge-blocked npc).
* Errors are the χO spread (last step) + χψ effect (256 vs 512 at matched χO); this is *not* a rigorous bound but successive differences are small and the
  χO sequence is consistent (see §3).

## 2. Final centre-site table (site 29; t≤3 from forward TEBD χ=3072, t≥4 from MIM)
| t | quantity | method | value | error bar | TDVP (paper, cont.) | HW mm+dr | HW − ours | TDVP − ours |
|---|---|---|---|---|---|---|---|---|
| 1 | ⟨n↑⟩ | forward TEBD χ=3072 (spread vs χ=2048) | 0.381606 | 1.0e-13 | 0.388595 | 0.385336 | +0.0037 | +0.0070 |
| 1 | ⟨n↑n↓⟩ | forward TEBD χ=3072 (spread vs χ=2048) | 0.178244 | 1.0e-13 | 0.177560 | 0.179281 | +0.0010 | -0.0007 |
| 2 | ⟨n↑⟩ | forward TEBD χ=3072 (spread vs χ=2048) | 0.535107 | 1.0e-13 | 0.527211 | 0.533754 | -0.0014 | -0.0079 |
| 2 | ⟨n↑n↓⟩ | forward TEBD χ=3072 (spread vs χ=2048) | 0.195278 | 3.4e-13 | 0.197408 | 0.194505 | -0.0008 | +0.0021 |
| 3 | ⟨n↑⟩ | forward TEBD χ=3072 (spread vs χ=2048) | 0.472562 | 9.3e-07 | 0.477674 | 0.489801 | +0.0172 | +0.0051 |
| 3 | ⟨n↑n↓⟩ | forward TEBD χ=3072 (spread vs χ=2048) | 0.190683 | 6.9e-06 | 0.192203 | 0.203463 | +0.0128 | +0.0015 |
| 4 | ⟨n↑⟩ | MIM k0=10 χψ=512 χO=512 | 0.522942 | 1.0e-04 | 0.517328 | 0.519818 | -0.0031 | -0.0056 |
| 4 | ⟨n↑n↓⟩ | MIM k0=10 χψ=512 χO=512 | 0.184515 | 1.0e-04 | 0.185598 | 0.196395 | +0.0119 | +0.0011 |
| 5 | ⟨n↑⟩ | MIM k0=10 χψ=256 χO=768 | 0.486191 | 1.0e-04 | 0.477373 | 0.511717 | +0.0255 | -0.0088 |
| 5 | ⟨n↑n↓⟩ | MIM k0=10 χψ=256 χO=768 | 0.187058 | 2.4e-04 | 0.159058 | 0.200440 | +0.0134 | -0.0280 |
| 6 | ⟨n↑⟩ | MIM k0=10 χψ=256 χO=1024 | 0.509031 | 1.0e-04 | nan | 0.509610 | +0.0006 | +nan |
| 6 | ⟨n↑n↓⟩ | MIM k0=10 χψ=256 χO=1024 | 0.187071 | 2.2e-04 | nan | 0.205978 | +0.0189 | +nan |

Interpretation (target = ideal Trotter-circuit output; "TDVP − ours" is mostly Trotter error plus TDVP truncation):
* TDVP (continuous time, χ=2048) misses the circuit by 0.005–0.008 on ⟨n↑⟩ (t≤4), by **0.009 on ⟨n↑⟩ and 0.028 on ⟨n↑n↓⟩ at t=5**, and has no value at t=6.
* Hardware (mm+dr) vs the exact circuit: ⟨n↑⟩ +0.0037, −0.0014, +0.0172, −0.0031, **+0.0255 (t=5)**, +0.0006 (t=6); ⟨n↑n↓⟩ is systematically high:
  +0.0010, −0.0008, +0.0128, +0.0119, **+0.0134**, **+0.0189 (t=6)** — hardware's double occupancy overshoots by ≈6–10 % at t≥3. The t=6 ⟨n↑⟩ agreement (+6e-4)
  is probably coincidental (hardware error at t=5 is 2.5e-2).
* Against the paper's TDVP accuracy: for the *circuit* target we beat it at t=5 (error 1e-4 vs TDVP offset 9e-3/28e-3) and supply t=6 which TDVP lacks. For the
  *continuous-time* target we have no Δt-extrapolation beyond t=2, so no claim.

## 3. All MIM runs (value vs χO, χψ)
| obs | t (k) | k0 | chi_psi | chi_O | value | cum. discarded wt (psi / O) | time Heis (s) |
|---|---|---|---|---|---|---|---|
| dd | 3.0 (15) | 10 | 512 | 256 | 0.190678 | 8.4e-06 / 3.4e-08 | 3 |
| dd | 4.0 (20) | 10 | 512 | 256 | 0.184507 | 8.4e-06 / 4.2e-04 | 31 |
| dd | 4.0 (20) | 10 | 512 | 512 | 0.184515 | 8.4e-06 / 4.2e-05 | 104 |
| dd | 5.0 (25) | 10 | 256 | 512 | 0.186879 | 4.3e-04 / 2.9e-03 | 283 |
| dd | 5.0 (25) | 10 | 256 | 768 | 0.187058 | 4.3e-04 / 1.1e-03 | 587 |
| dd | 5.0 (25) | 10 | 512 | 256 | 0.186317 | 8.4e-06 / 1.1e-02 | 92 |
| dd | 5.0 (25) | 10 | 512 | 512 | 0.186822 | 8.4e-06 / 2.9e-03 | 307 |
| dd | 6.0 (30) | 10 | 256 | 512 | 0.187249 | 4.3e-04 / 2.2e-02 | 553 |
| dd | 6.0 (30) | 10 | 256 | 768 | 0.187250 | 4.3e-04 / 1.1e-02 | 1397 |
| dd | 6.0 (30) | 10 | 256 | 1024 | 0.187071 | 4.3e-04 / 6.7e-03 | 2133 |
| dd | 6.0 (30) | 10 | 512 | 256 | 0.186402 | 8.4e-06 / 5.9e-02 | 163 |
| dd | 6.0 (30) | 10 | 512 | 512 | 0.187209 | 8.4e-06 / 2.2e-02 | 588 |
| nu | 3.0 (15) | 10 | 512 | 256 | 0.472569 | 8.4e-06 / 2.0e-09 | 1 |
| nu | 4.0 (20) | 10 | 256 | 256 | 0.523039 | 4.3e-04 / 1.1e-04 | 37 |
| nu | 4.0 (20) | 10 | 512 | 256 | 0.522973 | 8.4e-06 / 1.1e-04 | 27 |
| nu | 4.0 (20) | 10 | 512 | 512 | 0.522942 | 8.4e-06 / 8.4e-06 | 84 |
| nu | 5.0 (25) | 10 | 256 | 512 | 0.486264 | 4.3e-04 / 1.3e-03 | 250 |
| nu | 5.0 (25) | 10 | 256 | 768 | 0.486191 | 4.3e-04 / 4.6e-04 | 518 |
| nu | 5.0 (25) | 10 | 512 | 256 | 0.486017 | 8.4e-06 / 6.1e-03 | 91 |
| nu | 5.0 (25) | 10 | 512 | 512 | 0.486285 | 8.4e-06 / 1.3e-03 | 265 |
| nu | 6.0 (30) | 10 | 256 | 512 | 0.508962 | 4.3e-04 / 1.4e-02 | 500 |
| nu | 6.0 (30) | 10 | 256 | 768 | 0.508976 | 4.3e-04 / 6.6e-03 | 1117 |
| nu | 6.0 (30) | 10 | 256 | 1024 | 0.509031 | 4.3e-04 / 3.8e-03 | 4248 |
| nu | 6.0 (30) | 10 | 512 | 256 | 0.510102 | 8.4e-06 / 3.9e-02 | 165 |
| nu | 6.0 (30) | 10 | 512 | 512 | 0.508941 | 8.4e-06 / 1.4e-02 | 556 |

Notes: the first column pair gives the order of magnitude of truncation: cumulative discarded weight of the O-run is 1e-2…6e-2 at χO=256/512 for t=6 but the
observable changes by only ≲1e-3 → the observable converges much faster than the operator norm (the dominant discarded weight sits in far-from-centre /
identity-like operator strings that the product-state-like ψ(t1) barely sees).
Validation: sandwich vs exact statevector (L=6, k0=3, k=4,6,9; n↑ and n↑n↓): ≤2.4e-14. Heisenberg MPO alone vs exact literal circuit (L=4, k up to 30): ≤8e-14. At L=60,
MIM (χψ=512, χO=256) reproduces forward TEBD at t=3 to 7e-6 (n↑) / 5e-6 (n↑n↓) and at t=4 to 4e-5 / 4e-4 (within the forward error 3e-4 / 6e-3).

## 4. Operator entanglement (pure Heisenberg MPO of O=n_{29↑}, all 30 layers, HS-normalised, max bond entropy in nats)
| j (layers = time t=0.2 j) | S_op(chi=128) | S_op(chi=256) | S_op(chi=512) | cum. discarded wt chi=256 / 512 | forward-state S (chi=3072 for j<=20, 2048 beyond) |
|---|---|---|---|---|---|
| 3 (t=0.6) | 0.746 | 0.746 | 0.746 | 9.8e-31 / 9.8e-31 | 1.14 |
| 5 (t=1.0) | 1.073 | 1.073 | 1.073 | 2.0e-09 / 6.0e-12 | 1.80 |
| 6 (t=1.2) | 1.191 | 1.191 | 1.191 | 5.7e-08 / 4.8e-10 | 2.14 |
| 9 (t=1.8) | 1.481 | 1.485 | 1.485 | 2.9e-05 / 1.5e-06 | 3.07 |
| 10 (t=2.0) | 1.569 | 1.579 | 1.580 | 1.1e-04 / 8.4e-06 | 3.38 |
| 12 (t=2.4) | 1.711 | 1.746 | 1.753 | 8.4e-04 / 1.1e-04 | 3.99 |
| 15 (t=3.0) | 1.828 | 1.927 | 1.961 | 6.1e-03 / 1.3e-03 | 4.89 |
| 18 (t=3.6) | 1.830 | 2.026 | 2.109 | 2.1e-02 / 6.4e-03 | 5.79 |
| 20 (t=4.0) | 1.783 | 2.054 | 2.177 | 3.9e-02 / 1.4e-02 | 6.36 |
| 21 (t=4.2) | 1.745 | 2.055 | 2.200 | 5.0e-02 / 1.9e-02 | 6.53 |
| 24 (t=4.8) | 1.572 | 2.000 | 2.223 | 9.3e-02 / 4.1e-02 | 6.96 |
| 27 (t=5.4) | 1.363 | 1.884 | 2.191 | 1.5e-01 / 7.3e-02 | 7.08 |
| 30 (t=6.0) | 1.137 | 1.725 | 2.113 | 2.1e-01 / 1.1e-01 | 7.03 |


Reading: for j ≲ 12 (converged in χ) S_op ≈ 0.75, 1.07, 1.19, 1.49, 1.58, 1.75 at j=3,5,6,9,10,12 — **about +0.5 per doubling of j, i.e. log-like**, whereas the forward-state entropy
doubles when j doubles (1.14 → 2.14 → 3.99: linear, 0.33/layer). At larger j S_op is truncation-limited (the χ-dependent rows) but the χ=512 values (1.96 at j=15, 2.2 at j=21–24) still show
strongly sublinear growth/saturation. However the *discarded weight* grows quickly with j (χ=512: 1e-6 at j=9 → 1.3e-3 (15) → 1.4e-2 (20) → 0.11 (30)), so entropy alone underestimates the
χ needed for the full operator; this is why the meet-in-the-middle split (only 20 operator layers at k=30) is much better than the pure Heisenberg run: pure Heisenberg χ=512 at t=6 gives
⟨n↑⟩=0.5116 (χ=128: 0.5140, 256: 0.5190) vs MIM 0.50903 ± 1e-4.
Pure-Heisenberg (k0=0) vs forward TEBD at L=60, χ=512: t=1: 7.5e-8, t=2: 6e-5 (n↑); i.e. less accurate than forward at early times, as expected; it is the late-time tool.

## 5. Practical notes / incidents
* The first sandwich attempt at (χψ=768, χO=1024) reached 5.6–6.3 GB RSS (full-size χψ²·χO·d environment tensors) and was killed by the supervisor; all later heavy jobs ran with
  `prlimit --as=3000000000` and χψ=256 (χO=768: 0.92 GB, χO=1024: 1.5 GB RSS). No job exceeded 3 GB after that.
* Files: heis.py (pure Heisenberg), heis2.py/tebd2.py (variants returning states), sandwich.py, mim.py, mim_summary.py, final_table.py, gen_docs.py, heis_validate.py, test_sandwich.py,
  mim/*.json (every run), heis/*.json/log (pure Heisenberg), mim_queue*.sh (drivers).
* Only site 29 (and implicitly 30 by symmetry) was done for MIM; all-site RMSE at t=5,6 would need a Heisenberg run per site (not done).
