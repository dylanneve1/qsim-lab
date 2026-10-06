# Small-circuit tools (exact solve + certificate)

For peaked circuits that are small or shallow enough to simulate exactly. These are a different regime from the structural
solvers (`../solve_peaked.py`, `../solve_peaked_v2.py`), which target the 98-qubit HQAP circuits.

- `sv.py FILE.qasm`: exact state vector (complex64; rz, sx, x, h, u/u3, ry, cz, cx). Fuses single-qubit runs and updates the state
  in chunks, so peak RAM is about 1.5× the state (28 qubits ≈ 3 GB). Prints the top 5 bitstrings (qubit 0 first). Checked against
  quimb's dense simulation on a random 10-qubit circuit (fidelity 1.0).
- `direct.py FILE.qasm`: exact single-qubit marginals by tensor-network light cones (quimb/cotengra), then the per-qubit majority
  string and its exact probability. Saves the marginals to `FILE.zs.npy`.
- `cert.py FILE.qasm S P`: certificate that S (probability P) is the global peak. Any S' with p(S') ≥ P must agree with S on every
  wire whose minority marginal is below P. Its configuration x on the remaining "ambiguous" wires must also have exact marginal
  P_amb(x) ≥ P. So compute the exact joint marginal over the ambiguous wires. If only S's configuration reaches P, S is the
  unique peak (up to float error).
- `forced_diag.py`: diagnostic for `../bound-attempt/zipper2.py`. Prints the eigenphase structure of the residue at forced peels.
  This is how the exp(iπ/4·P) (CZ-class) residues were found.
