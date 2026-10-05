# `qsimlab` core — circuits, `simulate`, the planner, interop

**Status: stable** (phase 1; contract in [`../API.md`](../API.md)). This page is a runnable
tutorial: every `>>>` block is executed by `pytest` in CI, so if it is here, it works.

## 1. Build a circuit

Gate methods return the circuit, so they chain. Qubit 0 is the leftmost character of a
bitstring.

```python
>>> import numpy as np
>>> import qsimlab as qs
>>> bell = qs.Circuit(2).h(0).cx(0, 1)
>>> bell.num_qubits
2

```

## 2. Ask for one thing with `simulate`

`simulate(circuit, request)` takes one request — `statevector()`, `amplitudes([...])`,
`samples(shots)` or `expectation([...])` — and the planner picks the engine.

```python
>>> r = qs.simulate(bell, qs.statevector())
>>> r.state.round(4)
array([0.7071+0.j, 0.    +0.j, 0.    +0.j, 0.7071+0.j])
>>> r.probabilities().round(4)
array([0.5, 0. , 0. , 0.5])
>>> qs.simulate(bell, qs.amplitudes(["00", "11"])).amplitudes.round(4)
array([0.7071+0.j, 0.7071+0.j])
>>> qs.simulate(bell, qs.expectation(["Z0 Z1", "X0 X1", "Z0"])).values.round(6)
array([1., 1., 0.])

```

Sampling needs measurements; `seed=` makes it reproducible.

```python
>>> s = qs.simulate(bell.copy().measure_all(), qs.samples(2000), seed=7)
>>> s.shots, sorted(s.counts())
(2000, ['00', '11'])

```

## 3. Ask *why*: `explain=True` and `plan`

`explain=True` attaches the planner's reasoning to the result. `plan()` gives the same
prediction without running anything — use it before a big job.

```python
>>> r = qs.simulate(bell, qs.statevector(), explain=True)
>>> r.engine
'statevector'
>>> wide = qs.Circuit(40).h(0).cx(0, 1).measure_all()   # 40 qubits, but all Clifford
>>> qs.plan(wide, qs.samples(10)).engine
'tableau'

```

A 40-qubit state vector would need 16 TiB; the planner sees the circuit is Clifford and routes
it to the stabilizer tableau, which costs microseconds.

## 4. Move circuits in and out

OpenQASM 2 works with no extra dependencies; Qiskit, Cirq and Stim converters live in
`qsimlab.interop` and raise `MissingDependencyError` if the package is absent.

```python
>>> from qsimlab import interop
>>> print(interop.to_qasm(bell))
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
<BLANKLINE>
>>> interop.from_qasm(interop.to_qasm(bell)).num_qubits
2

```

## Next

- [`qec.md`](qec.md) — memory circuits, detector sampling, decoders, logical error rates.
- [`shor.md`](shor.md) — gate-level Shor, oracles, support prediction.
- [`analysis.md`](analysis.md) — magic, simulability, monitored circuits.
