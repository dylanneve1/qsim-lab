---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# qsimlab

<p class="tagline">Exact quantum circuit simulation for Python. A Rust core with seven engines and a
cost-model planner that picks the fastest exact one for each circuit. numpy in, numpy out.</p>

<div class="hero-buttons" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[API reference](reference/index.md){ .md-button }
[GitHub](https://github.com/dylanneve1/qsim-lab){ .md-button }
</div>

</div>

```python
import qsimlab as qs

# GHZ state on 3 qubits
c = qs.Circuit(3).h(0).cx(0, 1).cx(1, 2)

qs.simulate(c, qs.amplitudes(["000", "111"])).amplitudes
# array([0.7071+0.j, 0.7071+0.j])

qs.simulate(c, qs.expectation(["Z0 Z2", "X0 X1 X2"])).values
# array([1., 1.])
```

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } __Exact by default__

    ---

    No truncation or approximation unless a result says so. Every engine is
    differential-tested against an independent reference state vector.

-   :material-engine:{ .lg .middle } __The planner picks the engine__

    ---

    State vector, stabilizer tableau, sparse, MPS, hybrid Schrödinger–Feynman,
    compressed Clifford+T and batched noisy-Clifford sampling, chosen per connected
    component by a cost model. Pass `explain=True` to see why.

-   :material-shield-half-full:{ .lg .middle } __Error correction__

    ---

    Surface, repetition and colour-code memory circuits, detector sampling,
    detector error models, decoders and logical error rates.

    [:octicons-arrow-right-24: QEC guide](qec.md)

-   :material-key-variant:{ .lg .middle } __Shor at gate level__

    ---

    Factor semiprimes by exact simulation of the full X/CNOT/Toffoli circuit, with
    resource counts and noise studies.

    [:octicons-arrow-right-24: Shor guide](shor.md)

-   :material-chart-bell-curve:{ .lg .middle } __Why is it hard?__

    ---

    Magic, stabilizer rank and simulability analysis that explain a circuit's cost,
    mostly without simulating it.

    [:octicons-arrow-right-24: Analysis guide](analysis.md)

-   :material-swap-horizontal:{ .lg .middle } __Works with your stack__

    ---

    Convert to and from Qiskit, Cirq, Stim and OpenQASM 2. Typed API, abi3 wheels,
    no runtime dependencies beyond numpy.

    [:octicons-arrow-right-24: Interop reference](reference/interop.md)

</div>

## Built on research

qsimlab is the Python face of [qsim-lab](https://github.com/dylanneve1/qsim-lab), where every
headline result goes through an independent audit before it is merged. A few of them:

| Result | Number |
|---|---|
| Gate-level Shor, generic semiprime | 31-bit N factored by exact simulation of the full 132-qubit circuit |
| Detector sampling vs Stim 1.16 | 9–12× faster on x86 at ≥ 10⁵ shots, identical output distribution |
| Weight-6 qLDPC codes | [[224,18,12]] and [[288,34,8]], beyond every published weight-6 code |
| Engine planner | held-out regret 1.07–1.11 when choosing the exact engine |

More in [Research](research.md) and the full [results summary](https://github.com/dylanneve1/qsim-lab/blob/main/RESULTS.md).

!!! note "Status"
    `qsimlab` is **alpha** (v0.1). The core API (`circuit`, `sim`, `errors`, `interop`) is stable
    under the [API contract](API.md). `qec`, `shor` and `analysis` are provisional.
