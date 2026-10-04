# qsimlab — Python API contract (v0.1)

`qsimlab` is the Python face of the qsim-lab Rust engine (exact state-vector, stabilizer,
sparse, MPS, hybrid Schrödinger–Feynman, compressed Clifford+T and noisy-Clifford samplers,
chosen per circuit by the planner). The Rust crate does the work; the Python layer is thin,
typed and documented. This file is the contract other modules are implemented against.

Install (dev): `cd python && maturin develop --release` (or `pip install -e python`).
Wheels: abi3, CPython ≥ 3.9, numpy ≥ 1.22.

## 1. Modules

| module | status | contents |
|---|---|---|
| `qsimlab.circuit` | **stable** (phase 1) | `Circuit`, `Instruction`, `GATES` |
| `qsimlab.sim` | **stable** (phase 1) | `simulate`, `plan`, request constructors, result classes, `Budget`, `NoiseModel`, `ENGINES` |
| `qsimlab.interop` | **stable** (phase 1) | `from_qiskit`, `to_qiskit`, `from_cirq`, `to_cirq`, `from_stim`, `to_stim` |
| `qsimlab.errors` | **stable** (phase 1) | exception hierarchy (§4) |
| `qsimlab.qec` | provisional (phase 2) | codes, detector sampling, decoders, logical error rates |
| `qsimlab.shor` | provisional (phase 2) | modular-exponentiation circuits, period finding, resource counts |
| `qsimlab.analysis` | provisional (phase 2) | observables, fidelities, entropies, distributions, simulability |
| `qsimlab._native` | private | the PyO3 extension; never import it directly from user code |

Top level re-exports: `Circuit`, `simulate`, `plan`, `statevector`, `amplitudes`, `samples`,
`expectation`, `Budget`, `NoiseModel`, `set_num_threads`, `get_num_threads`, `__version__`,
and the submodules.

## 2. `qsimlab.circuit`

```python
c = Circuit(3)                       # all qubits start in |0>
c.h(0).cx(0, 1).cx(1, 2)             # builder methods return self
c.rz(2, 0.3).ccx(0, 1, 2)
c.measure(0); c.measure_all()        # measurement k writes classical bit k (program order)
c.x(2, c_if=0)                       # apply X if measurement 0 read 1
c.z(2, c_if=(1, False))              # ... if measurement 1 read 0
c.depolarize1(0, 1e-3); c.depolarize2(0, 1, 1e-3)
c.x_error(1, 0.01); c.y_error(1, 0.01); c.z_error(1, 0.01)
c.reset(2)
c.repeat(body, 1000)                 # unrolled; marks the circuit so the repeat pass is used
c.detector([3, 7]); c.observable_include(0, [7])   # QEC metadata (absolute measurement indices)
```

Gates (every gate of the Rust IR; `GATES` maps name → `(num_qubits, num_params)`):
`i h x y z s sdg t tdg sx sxdg rx ry rz p u` (1 qubit), `cx cz swap iswap iswapdg cp` (2),
`ccx` (3). Aliases: `cnot=cx`, `phase=p`, `cphase=cp`, `toffoli=ccx`, `id=i`.
Generic form: `c.append(name, qubits, params=(), *, c_if=None)`.

Other members: `num_qubits`, `num_measurements`, `global_phase` (float, radians, settable),
`readout_error` (global measurement flip probability, from Stim `M(p)`), `detectors`,
`observables`, `len(c)`, `instructions()` → `list[Instruction(name, qubits, params, c_if)]`,
`copy()`, `compose(other, qubits=None)`, `c + other`, `inverse()`, `without_measurements()`,
`stats()` → dict (`num_qubits, total_ops, total_gates, depth, gates_1q, gates_2q, gates_3q,
clifford_gates, t_gates, measurements, noise_channels, is_clifford, is_unitary`),
`draw()` → str, `to_qasm()` / `Circuit.from_qasm(src)`, `to_stim()` / `Circuit.from_stim(src)`,
`==`, pickling.

## 3. `qsimlab.sim`

```python
simulate(circuit, request, *, engine="auto", precision="f64", seed=None,
         budget=None, threads=None, noise=None, repeat=None, explain=False) -> Result
plan(circuit, request, *, budget=None) -> Explanation     # predict only, run nothing
```

Requests (plain frozen dataclasses):

| constructor | needs | result | payload |
|---|---|---|---|
| `statevector()` | unitary part | `StatevectorResult` | `.state`: complex `ndarray[2**n]` |
| `amplitudes(bitstrings)` | unitary part | `AmplitudesResult` | `.amplitudes`: complex128 `ndarray[k]` |
| `samples(shots)` | any circuit | `SamplesResult` | `.bits`: uint8 `ndarray[shots, m]`; `.counts()` |
| `expectation(paulis)` | unitary part | `ExpectationResult` | `.values`: float64 `ndarray[k]` |

* "unitary part": terminal measurements are ignored (the request is about the state before
  them); mid-circuit measurement, reset, conditionals and noise channels raise
  `UnsupportedOperationError` for these requests.
* `samples`: one column per measurement in program order (`SamplesResult.measured_qubits[k]`
  is the qubit of column k). A circuit without measurements is sampled on every qubit
  (column q = qubit q). Noise channels, resets, conditionals are simulated exactly.
* `expectation(paulis)`: a Pauli string or a list of them; real `<ψ|P|ψ>` per string.

Every result has: `engine` (str: the engine that ran, `"pipeline"` when components ran on
different engines), `components` (`list[(num_qubits, num_gates, engine)]`), `seed` (int, the
seed actually used), `precision` (`"f64"`/`"f32"`: what was actually used), `wall_time` (s),
`explanation` (`Explanation | None`).

**Engines.** `engine="auto"` sends the circuit through the compile pipeline (peephole, light
cone, connected components, classical suffix) and Planner v2 picks the cheapest exact engine
per component from fitted cost models. Overrides (whole circuit, no compile passes):
`"statevector"`, `"sparse"`, `"mps"` (exact, refuses to truncate), `"hsf"`, `"compressed"`
(samples/expectations only: no global phase), `"tableau"` (Clifford only), `"symphase"`
(noisy Clifford sampling, batched, the QEC workhorse). `ENGINES` lists them with what each
supports. An engine that cannot do the job raises; it never silently falls back.

**explain=True** attaches an `Explanation`: `engine` (the planner's choice for the whole
circuit), `ranked` (`[(engine, predicted_seconds)]`, best first), `plan_seconds`,
`cached`, `features` (dict: qubits, gates, t_count, clifford, predicted MPS bond, ...), and
`notes` (rule-based decisions, e.g. "Clifford: tableau"). With `engine="auto"` the pipeline
may split the circuit into components and plan each separately; `components` shows what
actually ran.

## 4. Errors

All exceptions derive from `qsimlab.QsimError`; each also derives from the closest builtin.

| exception | builtin base | raised for (Rust `SimError`) |
|---|---|---|
| `CircuitError` | `ValueError` | repeated qubit, bad gate name/arity/parameter, bad argument |
| `QubitIndexError` | `IndexError` | `QubitOutOfRange`, `ClassicalBitOutOfRange` |
| `UnsupportedOperationError` | `ValueError` | `Unsupported` (gate on engine), `MeasurementNotSupported`, `NotSupported` |
| `ResourceLimitError` | `MemoryError` | `TooLarge`, `TooManyTerms` (attrs `needed`, `limit` in bytes/terms) |
| `EngineAbortedError` | `ResourceLimitError` | a forced engine gave up (MPS would truncate, sparse too dense) |
| `ParseError` | `ValueError` | OpenQASM / Stim syntax or unsupported instruction (message has the line) |
| `MissingDependencyError` | `ImportError` | interop target (qiskit, cirq, stim) not installed |

Python-side argument checking raises `TypeError`/`ValueError` subclasses above as appropriate.
Rust panics are converted to `pyo3_runtime.PanicException` and are always bugs.

## 5. Conventions

* **Qubit order: little-endian everywhere.** Qubit q is bit q of a basis-state index:
  `|q2 q1 q0>` has index `4 q2 + 2 q1 + q0` (same as Qiskit). `statevector()[i]` is the
  amplitude of index i.
* **Bitstrings** (input to `amplitudes`, keys of `counts()`) are the binary representation of
  that index: the **rightmost character is qubit 0** (or measurement 0 for counts). Integers
  are accepted everywhere a bitstring is (up to 128 qubits).
* **Pauli strings**: dense `"XIZ"` has the rightmost character on qubit 0 (Qiskit order) and
  must have length `num_qubits`; sparse `"X0 Z2"` (also `"X0*Z2"`, `"Z2X0"`) names qubits
  explicitly and is preferred. Optional leading sign `+`/`-`. `""`/`"I"` = identity.
* **Two-qubit gates**: first argument is the control (`cx`, `cp`), matrices indexed by
  `2·bit(first) + bit(second)`.
* **Angles** in radians. `rx/ry/rz(θ) = exp(-iθP/2)`, `p(θ) = diag(1, e^{iθ})`,
  `u(θ, φ, λ)` = OpenQASM `u3`. `cp(θ) = diag(1,1,1,e^{iθ})`.
* **Global phase**: kept. Statevectors and amplitudes include `circuit.global_phase` and the
  exact phase of every gate. Interop keeps the source's global phase where the source has one.
* **Precision**: `precision="f64"` (default) or `"f32"`. f32 is honoured by the dense
  state-vector paths (statevector requests, `engine="statevector"`, noisy state-vector
  shots); every other engine computes in f64. `result.precision` reports what was used.
* **Seeds**: `seed=None` draws a fresh 64-bit seed from the OS and stores it in
  `result.seed`. Same seed + same engine + same qsimlab version ⇒ bit-identical samples. With
  `engine="auto"` the *distribution* is fixed but the specific samples can change if the
  planner picks a different engine (another budget, a different version, speculation).
* **Budget**: `budget=Budget(memory="4GiB")`, an int (bytes) or a string. Caps the largest
  register any engine allocates; default 32 GiB (the engine's hard cap), never above it.

## 6. Threading

* Every heavy call releases the GIL (`simulate`, `plan`, parsing, stats of big circuits), so
  Python threads can run simulations concurrently.
* `threads=None` uses the process default: `set_num_threads(n)` / `get_num_threads()`,
  initialised from `QSIMLAB_NUM_THREADS`, else `RAYON_NUM_THREADS`, else all cores. `threads=k`
  runs that one call on a dedicated k-thread rayon pool (pools are cached per size).
* `Circuit` objects are not safe to *mutate* from two threads at once (PyO3 borrow checks
  raise `RuntimeError` instead of corrupting); simulating the same circuit from many threads
  is fine.

## 7. Versioning and stability

* `qsimlab.__version__` follows the Rust crate (`0.y.z`). While `0.y`: anything marked
  **stable** above changes only with a minor bump (`0.y → 0.y+1`) after one minor release of
  `DeprecationWarning`; **provisional** modules may change in any release; `_native` and
  names starting with `_` are private.
* Numerical results are exact up to floating-point rounding (f64: ~1e-12 relative on
  amplitudes for circuits of ~10^4 gates); engine choice never changes a distribution.

## 8. Extension points (for phase-2 modules)

Each domain module owns exactly these files and needs no edits elsewhere:

| module | Rust (native) | Python (public) | tests |
|---|---|---|---|
| qec | `python/src/qec.rs` → `qsimlab._native.qec` | `python/qsimlab/qec.py` | `python/tests/test_qec.py` |
| shor | `python/src/shor.rs` → `qsimlab._native.shor` | `python/qsimlab/shor.py` | `python/tests/test_shor.py` |
| analysis | `python/src/analysis.rs` → `qsimlab._native.analysis` | `python/qsimlab/analysis.py` | `python/tests/test_analysis.py` |

* `pub fn register(m: &Bound<PyModule>)` in each Rust file is already called from
  `python/src/lib.rs` with the submodule; add `m.add_function(wrap_pyfunction!(f, m)?)?` etc.
* The Python files already import `from ._native import <name> as _native_<name>` and are
  imported by `qsimlab/__init__.py`; add public functions and list them in `__all__`.
  Add stubs for new native functions to `qsimlab/_native.pyi` (append a section).
* Shared Rust helpers (stable within phase 1, do not change their signatures):
  * circuits: take `circuit: &PyCircuit` (`crate::circuit::PyCircuit`) and call
    `circuit.snapshot()` → `Arc<CircuitData>` (fields `circuit: qsim_lab::Circuit`,
    `global_phase`, `readout_error`, `detectors`, `observables`, `has_repeats`;
    `noise_model()`, `without_terminal_measurements()`); return new circuits with
    `PyCircuit::from_data(CircuitData { .. })` and wrap them in Python with `Circuit._wrap(core)`.
    Python callers pass `circuit._core`.
  * GIL + threads: `crate::threads::heavy(py, threads, move || ...)` — never touch Python
    objects inside the closure.
  * errors: `crate::errors::map_sim_err(SimError)`, `qerr("ClassName", msg)`,
    `circuit_err`, `unsupported`, `value_err` (classes live in `qsimlab/errors.py`; add new
    ones there, derived from `QsimError` and a builtin).
  * conversions: `crate::convert::{parse_bitstrings, parse_pauli (→ PauliTerm), parse_memory,
    gate_from_parts, gate_parts, GATES}`; engine names `crate::sim::{engine_name, backend_name,
    parse_engine}`; whole requests `crate::sim::run_request(&CircuitData, &Req, &Opts)`.
  * numpy out: `numpy::PyArray1::from_vec(py, vec)` (`.reshape([r, c])` for 2-D).
* Results: reuse `qsimlab.sim.Result` (dataclass) as the base of new result types so every
  result carries `engine/components/seed/precision/wall_time/explanation`.

## 9. Known limitations (v0.1)

* Planned amplitudes need every connected component to have ≤ 63 qubits (engine indices are
  64-bit); larger components fall back to a dense state and hit the memory budget.
  Samples are indexed up to 128 qubits per component (Clifford and noisy-Clifford circuits:
  any size through the tableau / symphase samplers).
* Stim import supports one global `M(p)` readout probability (`readout_error`); use
  `X_ERROR(p)` before `M` for per-measurement flips.
* OpenQASM 2: measurement `k` in program order is record `k` regardless of the classical bit
  it writes; `if` works on one-bit registers only; the global phase is not representable.
* Predicted costs in `Explanation.ranked` come from models fitted on an Apple M1 Pro (single
  thread); use them to compare engines, not as wall-clock promises.
