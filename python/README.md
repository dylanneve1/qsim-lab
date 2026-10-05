# qsimlab

Python API for [qsim-lab](https://github.com/dylanneve1/qsim-lab): exact quantum circuit
simulation (state vector, stabilizer tableau, sparse, MPS, hybrid Schrödinger–Feynman,
compressed Clifford+T, batched noisy-Clifford sampling) with a cost-model planner that picks
the engine for you. The Rust crate does the work; this package is a thin, typed layer with
numpy in and out. The full contract is in [API.md](API.md).

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

Install for development: `pip install maturin && maturin develop --release` in this
directory (or `pip install -e python` from the repository root).

Tutorials (all executed in CI): [core simulation](docs/simulate.md), [QEC](docs/qec.md),
[Shor](docs/shor.md), [analysis](docs/analysis.md).
