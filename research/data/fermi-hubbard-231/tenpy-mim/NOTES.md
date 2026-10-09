# NOTES — issue 231, 1D Fermi-Hubbard Fock-state quench (L=60, U/t=-2, dt=0.2, 30 layers) — TEBD study

Research only; nothing was posted anywhere. Everything lives in /tmp/fh-231-t.

## 1. Spec as verified from the paper source (arXiv:2605.04025, /tmp/fh-231/arxiv/latex/{body,supplement}.tex)
* H = -t_h Σ_{i,σ}(c†_{iσ}c_{i+1σ}+h.c.) + U Σ n_{i↑}n_{i↓} − μ Σ n ; open boundaries, t_h=1, **U=-2**, μ=0 (the
  H_1Q term is ∝ total N and therefore only a global phase in every particle-number sector).
* Initial state: Néel Fock state |↓↑↓↑…⟩, **site 0 = ↓, site 1 = ↑, … i.e. even sites ↓, odd sites ↑ (site 59 = ↑)**. This agrees
  with the t=0 entries of the data files (e.g. hw/TDVP n_{29,↑}(0)=1, n_{30,↓}(0)=1). The plain Néel (no-vacancy) file set is the
  one for this issue (the vacancy data are not in the pickles). "Centre" sites: **i=29 (initially ↑) and i=30 (initially ↓)**; the
  paper's central vacancy/defect is site 29. By the particle–hole/spin structure the two are related by n_{29,↑}↔n_{30,↓} etc.
  (numerically n_up[30]≈n_dn[29] to ≤3e-5 at all steps for χ=2048; the Trotter ordering breaks reflection×spin-flip only at the edges.)
* **Correction to the brief:** "30 second-order Trotter steps" is not what the paper runs. The circuit has **30 Trotter layers** of
  Δt=0.2 (T=6); each layer is a *first-order* step; the layers alternate between the orderings (supp. Eqs. first_order_trotter and its
  mirror) so that an odd–even pair is second order (=15 second-order steps of 0.4; paper table footnote says exactly this).
  Odd step counts therefore carry an extra O(Δt²) "unpaired step" error (this is why t=1 (step 5), t=3 (15), t=5 (25) are
  Trotter-wise worse than t=2,4,6).
* Gate sequence per layer in pair-interleaved JW ordering {c0↓,c0↑,c1↑,c1↓,c2↓,…}: U_step = A·S·F·U·S·A (right-to-left in time)
  with A=exp(-iΔt/2 H_1Q), S=exp(-iΔt H_S) (RXX+RYY on qubits (2J+1,2J+2)), U=exp(-iΔt H_U2Q) (RZZ on (2J,2J+1)), F=fSWAP layer on
  (2J,2J+1); the fSWAP relabels spin up↔down on every site, so the second S is the "long hop" H_L, and the labelling toggles each layer
  (the final virtual permutation P for odd step counts is absorbed in the observables).
  Fermion-frame equivalent used by the TEBD (verified, see §2):
  * **odd layer:** S(Δt) → U(Δt) → L(Δt);  **even layer:** L(Δt) → U(Δt) → S(Δt) (time order).
  * S = ↑-hops on even bonds (2j,2j+1) + ↓-hops on odd bonds; L = ↓-hops on even bonds + ↑-hops on odd bonds; U = e^{-iΔt U n↑n↓}.
* Data: /tmp/fh-231/*.pkl; hardware arrays have 31 entries (index k = step k, t=0.2k); **TDVP arrays have only 30 entries
  (steps 0..29, t≤5.8) → no TDVP value at t=6**. TDVP is *continuous-time* H evolution (ITensor TDVP, Δt_TDVP=Δt_Trotter=0.2, ℓ=1, cutoff
  1e-8, χ up to 4096 in the paper; the pickles are the χ=2048 run per the issue), **not** the Trotter circuit. Hardware = the Trotter circuit
  + noise, so the exact Trotter-circuit result is the right benchmark for the hardware; TDVP differs from it by Trotter error.

## 2. Validation (all in /tmp/fh-231-t, `validate.py`, `exact_circuit.py`)
* `exact_circuit.py`: dense statevector (2^{2L}) of the **literal paper circuit** (Pauli-rotation RXX/RYY/RZZ/RZ gates, fSWAP layers,
  JW pair-interleaved ordering, relabelling) — completely independent of the fermionic d=4 gates. A second implementation without fSWAP
  (direct XZZX/YZZY long hops, alternating S,U,L / L,U,S) agrees with the literal circuit to 2e-15 (L=4, 6) → my reading of the layer
  structure is internally consistent.
* d=4 U(1)×U(1) TEBD (`tebd_fh.py`, TeNPy 1.1.1) at χ=512, svd_min=1e-16 vs the literal circuit, **30 steps, all sites, n↑, n↓, n↑n↓**:
  L=4: 4.9e-14, L=6: 9.1e-14, L=8: 1.1e-13 (max abs deviation). N↑ and N↓ conserved to ≤3e-15 (U(1)² exact by construction),
  truncation weight ≤1e-31. Target was ≤1e-10 — met by 3 orders of magnitude. JW sign handling: hopping is built from a 4-mode
  site-major Jordan–Wigner embedding, so the ↑ hop across a site carries (−1)^{n↓} and the ↓ hop (−1)^{n↑ of the right site}.
* Sign conventions: the overall sign of t and of the time direction cannot change any density observable for this Néel state
  (gauge c_i→(−1)^i c_i, and complex conjugation of a real problem). The sign of U does matter and is taken as in the paper
  (+U n↑n↓ with U=−2). The previous agent's "hopping sign fix" is therefore physically immaterial for these observables; I rebuilt the
  gates from scratch rather than reusing it.
* Trotter-vs-continuous cross-check at L=60: re-running the TEBD with Δt=0.1 and 0.05 (same layer structure, χ=256; converged for t≤2)
  and Richardson-extrapolating (Δt²) reproduces the ITensor-TDVP continuous-time values at the centre site to 2e-4 (t=1) and 3e-5 (t=2)
  (`runs/dtscan_report.txt`). So (i) the TEBD machinery is right at full size, (ii) the paper's TDVP numbers are accurate to ~1e-4
  at t≤2, and (iii) the 0.005–0.01 difference "TDVP − TEBD(Δt=0.2)" in the tables is **Trotter error of the circuit**, not truncation.
* Also: L=8 Trotter-vs-exact-continuous comparison (`val/trotter_vs_cont_L8.json`): differences of order 1e-2 at the centre sites
  for t≥3, same order as the L=60 differences.

## 3. Production runs (L=60, 30 layers, separate detached background jobs, one at a time, `nice -n 10`, OMP_NUM_THREADS=2)
* Algorithm: TEBD with *every hop applied as its own two-site gate* (59 per hop group, 118 SVDs per layer + a free single-site U layer),
  SVD truncation chi_max with svd_min=1e-10, no other approximation. Logs `runs/log_chi*.txt` (per step: t, n↑,n↓,n↑n↓ of site 29, max
  bond entropy, max bond dim, cumulative discarded weight, wall, RSS); JSONs `runs/tebd_L60_chi*.json` hold **all 60 sites** for n↑, n↓,
  n↑n↓ at every step.
* χ=128, 256, 512, 1024, 2048 complete (30 steps); χ=3072 run to step 20 (t=4.0) only (job stopped there by me — it was launched for 22 steps — because t≤4 was all it could settle) — see RESULTS.md. Peak RSS 0.11, 0.13,
  0.19, 0.41, 1.17 GB; 3072 ≈ 2.1–2.2 GB. Wall: 39 s, 97 s, 271 s, 937 s, 3477 s (χ=3072: ≈6–7 s/gate-layer growing to ~400 s/step).
  χ=4096 would need ≈4–5 GB (> 3 GB limit) and was not attempted.
* Entanglement: max bond entropy grows ~linearly in t as expected: 1.80 (t=1), 3.38 (2), 4.89 (3), 6.3 (4), 7.0 (5), 7.0 (6) nats
  (χ=2048, truncated; saturation at t≥5 is a truncation artefact of χ=2048, the χ=1024 values are already smaller).
* "Converged" is judged by the χ-sequence of the observable: spread between the two largest χ and the ratio of successive spreads.

## 4. Files
`exact_circuit.py`, `tebd_fh.py`, `validate.py` (+`val/`), `trotter_err.py`, `queue.sh`/`queue2.sh` (job drivers), `analyze.py`,
`make_results.py` (→ `results_tables.md`), `dtscan_report.py`, `runs/` (logs + JSON), `RESULTS.md`.
Run recipe: `OMP_NUM_THREADS=2 nice -n 10 /tmp/pk/research/data/peaked-circuits/.venv/bin/python -W ignore tebd_fh.py 60 <chi> 30`
(TeNPy from /tmp/su2-254-opus/pylib).

## 5. Caveats / honest limitations
* Truncation is local (per-gate SVD, svd_min=1e-10, chi_max); the χ-sequence spread is the only error bar used. Where successive spreads
  shrink geometrically (t≤4) the true error of the best χ is likely smaller than the quoted spread; at t=5,6 the spread is *not* shrinking
  (7.6e-2 → 6.6e-2 at t=5 for 512→1024→2048) so those points carry no certified accuracy.
* Cheap internal diagnostics are NOT reliable error estimates: e.g. |n↑+n↓−1| at the centre (zero for exact dynamics) is ≤2e-4 for
  χ=2048 at t=5 while the actual χ-error is ~7e-2; cumulative discarded weight is O(1) at t≥5 (2.1 at t=5, 5.9 at t=6 for χ=2048).
* The previous agent's /tmp/fh-231 TEBD tables (quimb, χ≤128, "order=2" step) use a different gate sequence (their χ=64 t=1 value
  0.3896 vs the exact-circuit 0.3816 here) and are not comparable; nothing from production_run.py was used.
* Hardware columns: mm+dr = measurement-mitigated + decay-recovered; no hardware error bars are given in the pickles.
* Trotter-vs-continuous: the TDVP pickle is continuous time. Everything here targets the Trotter circuit (the hardware's ideal output).
  A continuous-time TEBD value was only produced for t≤2 (Δt scan). I did not run Δt scans at t≥3.
* χ=3072 job stopped at t=4.0 (step 20); χ=4096 exceeds the 3 GB RSS cap (estimated 4–5 GB).

## 6. Update: Heisenberg MPO and meet-in-the-middle (parent's request)
* heis.py: vectorised operator MPS (doubled d=16 sites; gate = G† ⊗ Gᵀ; charge q_ket−q_bra), conjugates O by layers k, k−1, …, 1 (Heisenberg order: reverse of the time order
  of each layer), then ⟨Néel|O_H|Néel⟩ by product-state overlap; HS-norm prefactor 2^{L−1}‖o‖ and truncation-norm loss tracked. Validation: ≤8e-14 vs exact circuit (L=4, k≤30).
* mim.py/sandwich.py: ⟨ψ(k0)|O_H(layers k0+1..k)|ψ(k0)⟩ with npc left-to-right transfer contraction; validated to 2.4e-14 (L=6) and against L=60 forward TEBD (t=3: ≤7e-6, t=4: ≤4e-4).
* Memory incident: the unrestricted (χψ=768, χO=1024) sandwich blew past 6 GB (supervisor killed it). All later heavy jobs used prlimit --as=3e9 and χψ=256 (χO≤1024: ≤1.5 GB).
* Results/analysis: MIM_RESULTS.md (final table, χ sequences, operator-entanglement growth table).
* Mistake worth knowing: `pkill -f`/`ps|grep`-based kills in my own shell matched my own command line and aborted the shell several times (exit 144); use PIDs.
