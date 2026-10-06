# Getting started

## Install

qsimlab is not on PyPI yet. Build it from source; you need Python ≥ 3.9 and a Rust toolchain
([rustup.rs](https://rustup.rs)).

```sh
git clone https://github.com/dylanneve1/qsim-lab
cd qsim-lab
pip install -e python          # builds the Rust extension with maturin
```

For an optimised development build inside `python/`: `pip install maturin && maturin develop --release`.

Optional extras pull in the frameworks the converters talk to:

```sh
pip install -e "python[qiskit]"   # or [cirq], [stim], [all]
```

## First simulation

```python
>>> import qsimlab as qs
>>> c = qs.Circuit(3).h(0).cx(0, 1).cx(1, 2)          # GHZ
>>> qs.simulate(c, qs.amplitudes(["000", "111"])).amplitudes.round(4)
array([0.7071+0.j, 0.7071+0.j])
>>> qs.simulate(c, qs.expectation(["Z0 Z2", "X0 X1 X2"])).values.round(6)
array([1., 1.])
>>> r = qs.simulate(c.copy().measure_all(), qs.samples(1000), seed=42)
>>> sorted(r.counts())
['000', '111']

```

Every call has the same shape: **build a `Circuit`, pick a request, call `simulate`**.

| Request | Returns |
|---|---|
| `qs.statevector()` | the full state (`result.state`) |
| `qs.amplitudes([...])` | selected amplitudes |
| `qs.samples(n)` | `n` measurement shots (`result.counts()`) |
| `qs.expectation([...])` | Pauli-string expectation values |

The planner chooses the engine. To see its reasoning, pass `explain=True`; to force an engine,
pass `engine=...` (a forced engine that can't do the job raises instead of falling back).

!!! tip "Conventions"
    Bitstrings and state-vector indices are **little-endian** in qubit index; the full set of
    conventions is in §5 of the [API contract](API.md#5-conventions).

## Next steps

- [Core simulation guide](simulate.md): requests, budgets, noise, engines
- [Quantum error correction](qec.md), [Shor's algorithm](shor.md), [Analysis](analysis.md)
- [API reference](reference/index.md)
