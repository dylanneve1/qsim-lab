import subprocess, re
PY='/tmp/pk/research/data/peaked-circuits/.venv/bin/python'
subprocess.run([PY,'-W','ignore','/tmp/fh-231-t/final_table.py'],capture_output=True)
subprocess.run([PY,'-W','ignore','/tmp/fh-231-t/mim_summary.py'],capture_output=True)
final=open('/tmp/fh-231-t/FINAL_TABLE.md').read(); mt=open('/tmp/fh-231-t/mim_table.md').read(); oe=open('/tmp/fh-231-t/heis/oe_table.md').read()
doc='''# MIM_RESULTS — late times (t=4…6) via Heisenberg MPO + meet-in-the-middle (supersedes the "t=5,6 not converged" rows of RESULTS.md)

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
''' + final + '''

Interpretation (target = ideal Trotter-circuit output; "TDVP − ours" is mostly Trotter error plus TDVP truncation):
* TDVP (continuous time, χ=2048) misses the circuit by 0.005–0.008 on ⟨n↑⟩ (t≤4), by **0.009 on ⟨n↑⟩ and 0.028 on ⟨n↑n↓⟩ at t=5**, and has no value at t=6.
* Hardware (mm+dr) vs the exact circuit: ⟨n↑⟩ +0.0037, −0.0014, +0.0172, −0.0031, **+0.0255 (t=5)**, +0.0006 (t=6); ⟨n↑n↓⟩ is systematically high:
  +0.0010, −0.0008, +0.0128, +0.0119, **+0.0134**, **+0.0189 (t=6)** — hardware's double occupancy overshoots by ≈6–10 % at t≥3. The t=6 ⟨n↑⟩ agreement (+6e-4)
  is probably coincidental (hardware error at t=5 is 2.5e-2).
* Against the paper's TDVP accuracy: for the *circuit* target we beat it at t=5 (error 1e-4 vs TDVP offset 9e-3/28e-3) and supply t=6 which TDVP lacks. For the
  *continuous-time* target we have no Δt-extrapolation beyond t=2, so no claim.

## 3. All MIM runs (value vs χO, χψ)
''' + mt + '''

Notes: the first column pair gives the order of magnitude of truncation: cumulative discarded weight of the O-run is 1e-2…6e-2 at χO=256/512 for t=6 but the
observable changes by only ≲1e-3 → the observable converges much faster than the operator norm (the dominant discarded weight sits in far-from-centre /
identity-like operator strings that the product-state-like ψ(t1) barely sees).
Validation: sandwich vs exact statevector (L=6, k0=3, k=4,6,9; n↑ and n↑n↓): ≤2.4e-14. Heisenberg MPO alone vs exact literal circuit (L=4, k up to 30): ≤8e-14. At L=60,
MIM (χψ=512, χO=256) reproduces forward TEBD at t=3 to 7e-6 (n↑) / 5e-6 (n↑n↓) and at t=4 to 4e-5 / 4e-4 (within the forward error 3e-4 / 6e-3).

## 4. Operator entanglement (pure Heisenberg MPO of O=n_{29↑}, all 30 layers, HS-normalised, max bond entropy in nats)
''' + oe + '''

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
'''
open('/tmp/fh-231-t/MIM_RESULTS.md','w').write(doc); print('ok', len(doc))
