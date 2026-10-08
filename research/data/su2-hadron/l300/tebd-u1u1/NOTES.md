# su2-254 (l_i=300) — classical simulation notes (agent su2-254-opus)

## Physics after decoding
* Circuit = two 60-site hard-core-boson (= JW fermion) chains i,o, brick-wall Floquet step:
  even-bond hops (both chains) | odd-bond hops | on-site diagonal. All 20 steps identical (checked).
  Hop gate = exp(i*0.15*(c†c+h.c.)) per bond per step (|V01| = sin 0.15); staggered on-site phase ±0.03/step;
  rung vertex exp(i g n_i n_o), g = 0.226125 (the chain-o −g single-site phase is a uniform chemical potential, irrelevant).
* => Trotterized 1D Fermi-Hubbard chain (i,o = spin up/down), J t = 3, U/J ≈ 1.5, tiny staggered mass,
  quench from a doublon CDW (doublons on odd rungs). Meson = doublon moved 29 -> 30.
  SCV and meson circuits have identical gates (only initial X differ) — checked op-by-op.
* "large staggered potential" in the brief is wrong: the interaction g is the largest per-step scale.

## Diagnostic 1: G|MPS> / natural orbitals (diag.py, diag_L10.json; exact L=10 window)
step:      2     6     10    14    20
max min(ν,1-ν): 0.006 0.14 0.27 0.38 0.44      (SCV; meson similar, 0.45 at step 20)
corr. entropy Σh(ν) [bits, max 20]: 0.7 9.7 14.1 16.7 17.3
S(i|o) [bits]: 0.19 2.8 4.5 5.4 5.6
max cut S, site basis [bits]: 0.83 2.38 3.80 5.06 6.78
max cut S, NO basis (occ-ordered, interleaved): 0.19 2.58 4.16 5.76 7.27
=> natural occupations drift to ~1/2: strongly non-Gaussian; NO rotation only helps through step ~4.
   i|o entanglement is untouchable by any per-chain Gaussian G. Route dead.
Interaction picture ψ_I = U_free† ψ (diag_ip.py): max cut 4.33 vs 6.78 bits at step 20 (L=10), slope similar
(~0.27 vs 0.30 bits/step) -> only constant-factor χ saving, and dressed vertices become nonlocal MPOs. Not pursued.

## Diagnostic 2: Heisenberg MPO (heis.py)
Local n_r(t) MPO: D=64 truncation error (Frobenius) 3e-10,2e-8,3e-7,3e-6 at k=3..6 (x10/step) — worse than
state TEBD at equal cost. Dropped.

## Diagnostic 3: transverse / folded influence-matrix contraction (tim.py, diag_tim.py)
Validated vs exact L=10 window (3e-9, Tr=1.000000 at T=3). Temporal entanglement at bulk cut:
T=2: S_t=0.64-0.67 nats, χ_t=16 exact;  T=4: S_t=1.44, χ_t>=128 (disc 2.5e-9 at 128).
Spatial MPS same steps: step2 S=0.58 χ=76; step4 S=1.17 χ=206. g=0 gives IDENTICAL temporal numbers
-> temporal entanglement is free-fermion dominated and not lower than spatial; with 2T sites of d=16 it is
more expensive than TEBD here. Dropped.

## Winner: U(1)xU(1) charge-conserving TEBD on rung sites (tebd_sym.py, TeNPy 1.1.1 in ./pylib)
Gates exact (16x16 rung-pair hop⊗hop, diagonal absorbed into the next even layer); only error = truncation.
Cost on this shared VPS (2 threads, nice 10): χ=256 2 min, 512 6 min, 1024 4 min*, 2048 14 min (RSS 1.0 GB).
(*timings depend on box load.) Memory ~χ^2: 2048 -> 1.0 GB, 2800 -> ~1.9 GB (est), 4096 -> ~4 GB (est).
Validation: window TEBD == exact statevector to 5e-14 (L=10, all 20 steps) and 5e-13 (L=14, χ=8192, all steps);
full-system TEBD vs exact L=12/14 windows at central rungs: 1e-14 at steps 1-2, then differences shrink
100x from L=12 to L=14 (window hard-wall effect, not TEBD error).
g=0 control vs exact free fermions: see RESULTS.md.

## Final TEBD numbers (see RESULTS.md): step 20 n_f = 0.10285 ± 1e-5, stag_SCV = −2.03280 ± 3e-5
(N_SCV 57.96720, N_MID 58.07005). PP in the issue (n_f 0.0798) is unconverged (δQ ≈ 0.1, weight cutoff 20).
Gotcha: `pkill -f <pattern>` kills your own shell if the pattern appears in the command line — kill by PID.
