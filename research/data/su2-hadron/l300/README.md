# SU(2) hadron dynamics at l_i = 300 (tracker issue 254): investigation log

This instance uses the same 120-qubit circuit as l_i = 12, but the rung interaction phase is 22× stronger: g = θ = 0.226125, against 0.0101 before. The instance was built to sit beyond the perturbative regime. Status: **in progress**. The numbers below are interim and are not submitted.

## What was tried

| route | dir | outcome |
|---|---|---|
| Structure decode | `structure/` | Hops are still exact free-fermion gates (3.3e-16). The rung phase is g = 0.226125 on all 1200 vertices. MID moves both chains' site-29 pair to site 30, so free n_f(1) = 3.650671. `graph_util.py` needs its ZZ angle constant set to θ/2 = 0.1130625. |
| Free fermions, Hartree, PT2 | `windows/` | All fail at step 20. Free gives n_f 0.496, Hartree 0.226, PT2 −0.11. In the λ-series the odd orders are nonzero and ~10 orders are needed. |
| Exact windows, L = 6–12 | `windows/` | Converged to 1e-4 only through step ~2–4 (raw densities) or ~8 (central-site shift). Useful only as an early-step reference. |
| Site-basis Heisenberg monomial propagation (C) | `majorana-propagation/` | **Negative.** The term count doubles per vertex: 2M terms by step 9 at eps = 1e-3, with ~8e-3 truncation error already at step 2. Expanding in site monomials pays for the free hopping as operator spreading. |
| Natural-orbital diagnostic | `tebd-u1u1/diag*.py` | **Negative for G·MPS.** Natural occupations reach 0.25–0.75 by step 20, so the state is strongly correlated. The natural-orbital basis is not less entangled than the site basis. |
| Folded / temporal-entanglement contraction | `tebd-u1u1/` | **Negative.** Temporal entropy ≈ spatial entropy (step 4: 1.44 vs 1.17 nats), and it is identical at g = 0. Free-fermion dominated, so it is no better than TEBD. |
| **U(1)×U(1) TEBD (TeNPy)** on 60 rung sites, d = 4 | `tebd-u1u1/tebd_sym.py` | **Works.** The circuit is compiled exactly into two-site gates; the only error is SVD truncation. χ = 2048: 14 min, 1.0 GB. |

## Interim U(1)×U(1) TEBD results at step 20

| χ | discarded weight | stag_SCV | stag_MID | n_f |
|---|---|---|---|---|
| 512 | 7.8e-2 | −2.0939 | −1.9817 | 0.1121 |
| 1024 | 2.8e-3 | −2.0341 | −1.9309 | 0.1032 |
| 2048 | 1.2e-4 | −2.0328 | −1.9299 | 0.1029 |

The g = 0 control at χ = 1024 reproduces exact free fermions to 9.6e-4 (stag) and 5e-5 (n_f). Note that the issue text reports Pauli propagation with n_f(20) = 0.0798 (atol 1e-5, 101M terms, Q drift 0.08–0.12) and a hardware value of 0.0995 (3 Aug). The 0.023 gap to Pauli propagation is being investigated before any claim is made.

Dependencies: TeNPy (vendored locally, not committed), numpy. Large outputs (npy/pkl, TEBD JSON over 2 MB) are not committed.
