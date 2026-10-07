# SU(2) hadron dynamics (LSH) circuits as near-free fermions

Target: the `su2_hadron_dynamics_lsh` observable-estimation circuits on the
[quantum advantage tracker](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io)
(issue 149), at upstream commit `1db844f1540a198c5620af49247e09fc28e7f61b`.

## What the circuits are

After undoing the SWAP network, the 120 qubits form two 60-site chains (a
ladder). The gates fuse into exactly three kinds of block:

- **2360 hopping blocks** on adjacent sites of one chain. Each conserves
  particle number and is an exact free-fermion (Gaussian) gate: off-block
  leakage and determinant mismatch are below 3.3e-16.
- **1200 on-site blocks** coupling the two chains at the same rung. They are
  diagonal, with an interaction phase |g| ≤ 0.010.
- **Single-qubit Z phases.**

With g = 0 the circuit is exactly free fermions, and evolving a 60×60
correlation matrix per chain simulates it exactly. With g ≠ 0,
time-dependent Hartree captures the first-order correction.

## Results (step 20)

| method | stag_SCV | stag_meson | n_f = meson − SCV |
|---|---|---|---|
| free fermion (exact for g=0) | −2.641078 | −2.523040 | 0.118038 |
| time-dependent Hartree | −2.630801 | −2.513439 | 0.117363 |
| Pauli propagation, converged (tracker issue 250) | — | — | 0.1169 |

Q = 60 exactly in both modes. The free-fermion result is within 1.2e-3 of the
converged n_f.

The SCV-only staggered occupation is less settled. Pauli propagation at
atol 1e-4 / 1e-5 / 1e-6 gives −2.6355 / −2.6375 / −2.6277, so it is not
converged at the 1e-2 level. The free-fermion value sits within that spread.
A ladder TEBD at chi=32 (fidelity > 0.9996 up to step 6) puts the interaction
shift in stag_SCV at about 1e-2 by step 6.

Both n_f and stag_SCV come from correlated truncations. Their errors largely
cancel in the difference n_f, which is why n_f is far better converged.

## Run

```
git clone https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io qat
cd qat && git checkout 1db844f1540a198c5620af49247e09fc28e7f61b && cd -
export SU2_CIRCUITS=$PWD/qat/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh
python3 gauss.py      # free + Hartree, SCV + meson; ~8 s total on one core, numpy only
```

Files:

- `parse.py`: minimal QASM reader.
- `graph_util.py`: SWAP-network unwinding to logical (site, chain) wires.
- `gauss.py`: block fusion, Gaussian checks, and the free / Hartree evolution.
- `ladder.py`, `tebd.py`: ladder MPS TEBD cross-check.
- `ent.py`: entanglement diagnostics.
