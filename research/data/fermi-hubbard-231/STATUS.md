# Fermi-Hubbard quench (tracker issue 231): status, SOLVED for the Trotter circuit (centre site, t=1…6)

**Final:** see `tenpy-mim/MIM_RESULTS.md`. Forward U(1)×U(1) TEBD for t ≤ 3 (χ=3072), and meet-in-the-middle (forward state at t=2 sandwiched with the Heisenberg MPO of the remaining layers) for t ≥ 4. The result is converged to ~1e-4 at t=4–6, validated to 2e-14 against the exact statevector. The TDVP-vs-circuit gap is Trotter error, since TDVP is continuous-time. The notes below describe the earlier Gemini attempt and are kept for the record.


**Problem.** 1D Fermi-Hubbard quench from a Néel state with U/t = −2. The circuit is a second-order Trotter product with dt = 0.2 and 30 steps (T = 6). The paper's system is 60 sites (120 qubits). The observables are ⟨n_{c,σ}⟩ and ⟨n_{c↑} n_{c↓}⟩ at the centre site.

**What is solid:**
- The d = 4 local Hamiltonian and fermionic-sign convention (after fixing a hopping sign) agree with exact evolution for L = 4 and L = 6 to 1e-13 (`exact_val*.py`).
- At t = 1, χ = 64 matches the paper's TDVP to ~1e-3.

**What is NOT valid (do not reuse as results):**
- `production_run.py` simulates **L = 30** sites, not the paper's 60.
- It also integrates the continuous Hamiltonian rather than the hardware's Trotter product.
- Its logged entropy diagnostics are wrong (negative values).

**Next.** Redo the run with L = 60 and the exact second-order Trotter product. Sweep χ (with U(1)×U(1) charge conservation, as in `../su2-hadron/l300/tebd-u1u1/tebd_sym.py`), then compare against TDVP and the hardware data in the issue's Archive.zip, which is not committed.
