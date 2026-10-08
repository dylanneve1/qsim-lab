# Fermi-Hubbard quench (tracker issue 231): status, PARKED

**Problem.** 1D Fermi-Hubbard quench from a Néel state with U/t = −2. The circuit is a second-order Trotter product with dt = 0.2 and 30 steps (T = 6). The paper's system is 60 sites (120 qubits). The observables are ⟨n_{c,σ}⟩ and ⟨n_{c↑} n_{c↓}⟩ at the centre site.

**What is solid:**
- The d = 4 local Hamiltonian and fermionic-sign convention (after fixing a hopping sign) agree with exact evolution for L = 4 and L = 6 to 1e-13 (`exact_val*.py`).
- At t = 1, χ = 64 matches the paper's TDVP to ~1e-3.

**What is NOT valid (do not reuse as results):**
- `production_run.py` simulates **L = 30** sites, not the paper's 60.
- It also integrates the continuous Hamiltonian rather than the hardware's Trotter product.
- Its logged entropy diagnostics are wrong (negative values).

**Next.** Redo the run with L = 60 and the exact second-order Trotter product. Sweep χ (with U(1)×U(1) charge conservation, as in `../su2-hadron/l300/tebd-u1u1/tebd_sym.py`), then compare against TDVP and the hardware data in the issue's Archive.zip, which is not committed.
