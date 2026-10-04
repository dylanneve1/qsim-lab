# `qsimlab.qec` — memory circuits, detector sampling, error models, decoders

**Status: provisional** (phase 2; see [`../API.md`](../API.md) §1, §8 and the `qsimlab.qec`
section). The pipeline is *build → sample → decode → count*, every heavy step runs in Rust with
the GIL released, and everything is cross-checked against Stim and PyMatching
(`python/tests/test_qec.py`).

| what | function / class | engine underneath |
|---|---|---|
| surface code memory (rotated, X or Z basis) | `surface_code_memory(d, rounds, basis, p=, noise=)` | new generator (`python/src/qec.rs`), Stim's layout and hook-safe CNOT order |
| repetition code memory | `repetition_code_memory(d, rounds, p=, noise=)` | new generator |
| colour code memory (6.6.6, K–F rounds, flags) | `color_code_memory(d, rounds, basis, schedule=, flags=, p=, noise=)` | `qsim_lab::qec::color::memory_flagged` |
| detection events | `sample_detectors(c, shots, seed=, engine=)`, `DetectorSampler` | `FastSampler` (`research/fast-sampler.md`) / SymPhase |
| detector error model | `detector_error_model(c)` → `DetectorErrorModel` (`to_stim_dem`, `from_stim_dem`, `matrices`, `graphlike`) | SymPhase symbolic frame |
| exact circuit distance + count | `circuit_distance(c, max_weight=, detectors=, timeout=)` | `qsim_lab::qec::distance` (branch and bound) |
| decoding | `decode(dem, dets, "bposd" \| "pymatching" \| "tesseract")`, `make_decoder` | `qsim_lab::qec::bposd`; PyMatching / Tesseract adapters |
| logical error rate | `logical_error_rate(c, shots, decoder=, seed=, max_errors=, rounds=)` | fused sample + BP+OSD in Rust, or Python chunks for other decoders |

Circuits are ordinary `qsimlab.Circuit`s: detectors and observables are absolute measurement
indices (`c.detectors`, `c.observables`), every noise location is an explicit op, and
`c.to_stim()` is exactly the circuit that is sampled. `return_layout=True` also returns a
`CodeLayout` (qubit roles and coordinates, per-detector `(x, y, round)`, stabilizer basis, flag
detectors, and `memory_detectors`, the sector the observable lives in).

## Tutorial: a surface code memory, end to end

Build a distance-3 Z-basis memory with three rounds under SI1000 noise at p = 0.3%:

```python
>>> import numpy as np
>>> import qsimlab.qec as qec
>>> c, layout = qec.surface_code_memory(3, rounds=3, basis="Z", p=3e-3, noise="si1000",
...                                     return_layout=True)
>>> c.num_qubits, len(c.detectors), len(c.observables), len(layout.memory_detectors)
(17, 24, 1, 16)
>>> print(c.to_stim().splitlines()[0])           # the exact sampled circuit, in Stim format
R 0 1 2 3 4 5 6 7 8 9

```

Sample detection events. A `DetectorSampler` compiles once (the SymPhase frame and the
FastSampler's hit tables) and samples many times; seeds are reproducible for any thread count:

```python
>>> sampler = qec.DetectorSampler(c)
>>> sampler
DetectorSampler(engine='fast', detectors=24, observables=1)
>>> dets, obs = sampler.sample(100_000, seed=2026)
>>> dets.shape, dets.dtype, round(float(dets.mean()), 3)
((100000, 24), dtype('bool'), 0.075)
>>> packed, _ = sampler.sample(100_000, seed=2026, packed=True)   # Stim's bit_packed layout
>>> bool((np.unpackbits(packed, axis=1, count=24, bitorder="little") == dets).all())
True

```

The detector error model is read off the same symbolic frame. Its text is Stim's DEM format
(and it equals Stim's `detector_error_model()` of `c.to_stim()`, see the tests):

```python
>>> dem = qec.detector_error_model(c)
>>> dem
DetectorErrorModel(detectors=24, observables=1, errors=219)
>>> dem.to_stim_dem().splitlines()[0]
'error(0.006791033333739869) D0'

```

The circuit-level distance is exact: the smallest number of error mechanisms that flip the
logical observable without firing a detector, plus the number of such minimum-weight logicals:

```python
>>> r = qec.circuit_distance(c)
>>> r.distance, r.count, r.complete
(3, 136, True)

```

Decode the samples. BP+OSD is built in; PyMatching and Tesseract are used if installed:

```python
>>> pred = qec.decode(dem, dets, "bposd")
>>> int((pred != obs).any(axis=1).sum())          # failures out of 100,000
2502
>>> pred = qec.decode(dem, dets, "pymatching")   # doctest: +SKIP
>>> int((pred != obs).any(axis=1).sum())          # doctest: +SKIP
2798

```

BP+OSD decodes the full hypergraph DEM (it sees the correlation between the X and Z parts of a
Y error); matching needs graphlike edges and loses that information, hence the higher count.

Finally, the logical error rate against p for d = 3, 5, 7 (each point stops after 1000 failures;
`rounds=d` also reports a per-round rate). This is how `qec_ler.png` below was made, with
`shots=1_000_000`:

```python
>>> rows = []
>>> for d in (3, 5):
...     for p in (1e-3, 2e-3, 4e-3):
...         circuit = qec.surface_code_memory(d, d, p=p, noise="si1000")
...         r = qec.logical_error_rate(circuit, 10_000, seed=1, decoder="bposd", rounds=d)
...         rows.append((d, p, r.rate, r.ci))
>>> [round(rate, 4) for d, p, rate, ci in rows]                # doctest: +SKIP
[0.0028, 0.0098, 0.0416, 0.001, 0.0064, 0.0394]
>>> import matplotlib                                          # doctest: +SKIP
>>> matplotlib.use("Agg")                                      # doctest: +SKIP
>>> import matplotlib.pyplot as plt                            # doctest: +SKIP
>>> fig, ax = plt.subplots()                                   # doctest: +SKIP
>>> for d in (3, 5):                                           # doctest: +SKIP
...     x = [p for dd, p, _, _ in rows if dd == d]
...     y = [rate for dd, _, rate, _ in rows if dd == d]
...     err = [[rate - ci[0] for dd, _, rate, ci in rows if dd == d],
...            [ci[1] - rate for dd, _, rate, ci in rows if dd == d]]
...     _ = ax.errorbar(x, y, yerr=err, marker="o", label=f"d = {d}")
>>> ax.set_xscale("log"); ax.set_yscale("log"); _ = ax.legend()  # doctest: +SKIP
>>> fig.savefig("ler.png")                                      # doctest: +SKIP

```

![Logical error rate per shot of the rotated surface code memory (d rounds, Z basis) under SI1000
noise with PyMatching, d = 3, 5, 7; the curves cross near p = 0.4%.](qec_ler.png)

(1,000,000 shots per point or 1000 failures, Wilson 95% intervals; 80 s for all 18 points on the
loaded VPS.)

## Noise models

`noise=` picks the model and `p=` its strength. Every location is an explicit op of the circuit
(the readout flip is `circuit.readout_error`, Stim's `M(p)`).

| model | after CNOT | idle, gate layer | idle, measure/reset layer | readout flip | reset flip |
|---|---|---|---|---|---|
| `"cnot"` | `DEPOLARIZE2(p)` | – | – | – | – |
| `"uniform"` | `DEPOLARIZE2(p)` | `DEPOLARIZE1(p)` | `DEPOLARIZE1(p)` | p | p |
| `"si1000"` | `DEPOLARIZE2(p)` | `DEPOLARIZE1(p/10)` | `DEPOLARIZE1(2p)` | 5p | 2p |

Measurement and reset in the X basis are `MX = H M H` and `RX = R H` with the flip applied in
the right basis (`Z_ERROR` after `RX`); these basis changes carry no gate noise. SI1000 follows
Gidney et al. (the auxiliaries are reset/measured in their stabilizer's basis instead of using
explicit Hadamards, so it has no single-qubit gate layers). For the colour code `"cnot"` and
`"uniform"` are exactly the engine's `ColorNoise` models used in `research/qec-r4.md` and
`research/colour-global.md`; `"si1000"` is the uniform circuit with each location rescaled.

## Colour codes: schedules and flags

`schedule=` accepts `"kf"` (Kishony–Fowler, the default), `"tri"` (Lee et al.'s tri-optimal),
`"global"` (the exact schedules of `research/colour-global.md`: d = 9 with circuit distance 8
against K–F's 7, and d = 11 with 7 + 7 layers and distance 10), a `.sched` file, or a
`(plaquettes, 6)` array. `flags=True` puts a flag qubit on every boundary-touching plaquette
(`research/colour-flags.md`). The sector search makes these distances cheap:

```python
>>> kf, lk = qec.color_code_memory(9, 1, schedule="kf", p=1e-3, return_layout=True)
>>> r = qec.circuit_distance(kf, detectors=lk.memory_detectors)
>>> r.distance, r.certified
(7, True)

```

## Engines, seeds and performance

* `engine="auto"` (default) uses the FastSampler and falls back to the plain SymPhase sampler
  (same distribution, slower) with `DetectorSampler.note` set if the FastSampler rejects the
  circuit; `"fast"` / `"symphase"` force one. No circuit expressible in `qsimlab` has needed the
  fallback so far: the FastSampler handles every SymPhase variable group (flips, depolarizing
  channels, coins).
* Shots are drawn in chunks of 1024 from independent wyrand streams seeded by `(seed, chunk)`,
  in parallel; the same seed gives the same samples for any `threads=`.
* `logical_error_rate(..., max_errors=k)` checks the stopping rule every 65,536 shots, so its
  result is also seed-reproducible.

THROUGHPUT_TABLE

## Limitations

* Circuits must be Clifford with Pauli noise (`x/y/z_error`, `depolarize1/2`, readout flips) and
  no classically controlled gates; `UnsupportedOperationError` otherwise.
* At most 64 observables for the decoders (BP+OSD returns a 64-bit mask).
* `circuit_distance`: exponential in the distance; the full DEM of large colour codes takes
  minutes (use `detectors=layout.memory_detectors`, which is exact when `certified`).
  `timeout=` is approximate (the node budget per weight is set from the measured search rate).
* BP+OSD (the engine's decoder) runs OSD on many shots of large surface codes (~0.6 ms per shot
  at d = 5, p = 0.3% on the loaded VPS); for surface codes PyMatching is much faster.
* The PyMatching adapter decomposes hyperedges into graphlike mechanisms that exist in the model
  (`DetectorErrorModel.graphlike()`); mechanisms it cannot decompose are dropped with a warning.
* `si1000` for the colour code rescales the uniform circuit's locations by moment kind; for a
  hand-written schedule whose first CNOT layer is empty the idle noise of that layer would be
  classified as a reset layer's (`2p` instead of `p/10`).
* Detector coordinates live in the `CodeLayout`, not in the circuit (`Circuit` has no coordinate
  annotations; `to_stim()` writes no `QUBIT_COORDS`).
