# Operator Loschmidt echo: exact value for `operator_loschmidt_echo_49x648`

**Result.** f = 0.822488318196 for the 49-qubit, L=3, δ=0.15 instance with observable Z52 Z59 Z72.

**Method.** Exact tensor-network contraction. Nothing is truncated or sampled, and every initial state is included.

**Formula.** By the tracker's definition, f = 2^-n Σ_z σ_z ⟨z|W† O W|z⟩ with W = U† V U. This equals 2^-n Σ_{a,b} s_a s_b |⟨b|W|a⟩|² for diagonal O, which is the doubled network contracted here (`tnexact_build.py`).

## Checks

- **Circuit identity.** `gen.py` builds W_L from the L=6 tracker QASM. Its L=3 output matches the provided `49Q_OLE_circuit_L_3_b_0.25_delta0.15.qasm`:
  - 4756 gates each;
  - identical per-qubit gate sequences on all 49 qubits, CZ partners included.
- **Pipeline validation.** On a dense 12-qubit sub-circuit, the TN pipeline gives 0.9754460616, matching exact state-vector evolution.
- **Two independent contraction paths.** Both are in complex128 and agree to all 12 digits:

  | path | log file | flops | intermediate cap | slices | f | imag |
  |---|---|---|---|---|---|---|
  | 1 | `run_49x648_L3_path1.log` | 2^37.1 | 2^23 | 32 | 0.822488318196 | 1e-15 |
  | 2 | `run_49x648_L3_path2.log` | — | 2^22 | 1024 | 0.822488318196 | −5.5e-16 |

  Path 1 used a KaHyPar tree; path 2 came from a fresh greedy+KaHyPar search.
- **Same family, other depths.** f(L=1) = 1 and f(L=2) = 0.897254392906.

## Existing tracker entries

| method | f |
|---|---|
| BP-TN, BD 192 | 0.8203 |
| BP-TN, BD 512 | 0.8217 |
| Hardware (global rescaling) | 0.824 |
| Single-path MC | 0.808 |

## Structure

These are notes on the circuit; they are not needed for the number.

- U2 = Z_S U1 Z_S exactly, where S is the set of b=0.25 sites. Hence W = Zt V Zt, with Zt = U1† Z_S U1.
- Pauli propagation is hopeless here: the operator fully scrambles by L=3.
- Exact cost grows by about 25 bits per Trotter step (2^16, 2^37, 2^63 for L = 2, 3, 4). L=6 is out of reach exactly.

## Run

```
git clone https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io qat
python3 search.py 49 3 --target 23 --contract   # needs quimb, cotengra, kahypar
```
