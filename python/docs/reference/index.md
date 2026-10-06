# API reference

Generated from the docstrings in [`python/qsimlab`](https://github.com/dylanneve1/qsim-lab/tree/main/python/qsimlab).
The normative contract (conventions, error types, threading and stability guarantees) is the
[API contract](../API.md); where the two disagree, the contract wins and the docstring is a bug.

| Module | What it holds | Status |
|---|---|---|
| [`qsimlab.circuit`](circuit.md) | `Circuit` builder, gate set, instructions | stable |
| [`qsimlab.sim`](sim.md) | `simulate`, requests (`statevector`, `amplitudes`, `samples`, `expectation`), results, budgets, noise | stable |
| [`qsimlab.errors`](errors.md) | the `QsimError` hierarchy | stable |
| [`qsimlab.interop`](interop.md) | conversion to and from Qiskit, Cirq, Stim and OpenQASM 2 | stable |
| [`qsimlab.qec`](qec.md) | memory circuits, detector sampling, error models, decoders | provisional |
| [`qsimlab.shor`](shor.md) | factoring runs, gate-level oracle circuits, resource counts, cost laws | provisional |
| [`qsimlab.analysis`](analysis.md) | magic, stabilizer rank, simulability, monitored dynamics | provisional |

**Provisional** modules may change between minor versions; see §7 of the contract.
