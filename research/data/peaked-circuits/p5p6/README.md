# P5 / P6 structure work and our MPO + unswapping grower

The notes are in `NOTES.md`. In short:
- **P5** (44q): R ▷ U ▷ swap network ▷ π·U†·π⁻¹ ▷ P. π (4- and 2-cycles) is inferred from Makhlin invariants of mirrored
  layers (`invmatch.py`, `perm2.py`), and the swap network equals P_π per 4-qubit group to ~1e-2 (`netcheck.py`). The submitted
  P5 answer came from a sibling solver (core R ▷ π ▷ P plus a compressed operator); see `../PORTAL.md`.
- **P6** (62q, unsolved): three rings A/B/C (20/20/22 qubits), with A and B twins under a reversed map. There is no
  global mirror, no balanced graph cut below the trivial 62-bond time cut (`cutprobe.py`), and no collapse under exact CZ
  cancellation (`czsimp.py`) or pyzx (`zxred.py`).
- **`mpou/`**: our own MPO-with-unswapping grower, after the idea of Kremer–Dupuis (arXiv 2604.21908) and Galda's
  public solver; this is not their code. Routing is by relabelling MPO legs, unswapping is greedy per bond, plus a
  **matching unswap**: Pauli-transfer score S[i,j] = (1/3)·Σ|Tr(W P_i W† Q_j)|²/d² followed by an assignment solve
  (`match.py`). `matching_unswap_proto.py` is the dense validation: the hidden permutation is recovered exactly down to
  83% trace fidelity. The exact tests are `test_mpou.py` and `test_match.py`.
- Status: mpou absorbs ~270/1892 P5 units (truncation 1e-5) before blowing up at the swap-network boundary, and stalls on P6 at
  ~50/3494 units (bond 256).
