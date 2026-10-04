# qsim-lab

Exact quantum circuit simulation, written from scratch in Rust, plus the
research built on top of it: gate-level Shor, quantum error-correction
tooling, a cost-model planner that picks the right exact engine for each
circuit, and a set of theorems with executable checks.

Everything here is **exact**: no truncation or approximation unless a result
says so explicitly. Every engine is differential-tested against an independent
reference state vector, and every headline result went through an independent
audit before merge (see [research/audit.md](research/audit.md)).

- **Results summary:** [RESULTS.md](RESULTS.md)
- **Architecture:** [research/ARCHITECTURE.md](research/ARCHITECTURE.md)
- **Lab notebooks:** [research/](research/), indexed [below](#research-index)
- **Python API:** `qsimlab` (PyO3 + maturin, abi3 wheels, numpy in/out): [quickstart](#python-quickstart), contract in [python/API.md](python/API.md)

## Highlights

| result | number | notebook |
|---|---|---|
| Gate-level Shor, generic semiprime | 31-bit N = 1 537 596 787 factored by exact simulation of the full X/CNOT/Toffoli circuit (132 qubits); 77 s on an M1 Pro with Ekerå–Håstad | [shor.md](research/shor.md), [ge-shor.md](research/ge-shor.md) |
| Shor circuit size at 31 bits | 1.70 M → 1.04 M gates (superoptimised oracle); Toffolis 528 k → 46–61 k with measurement-based uncompute and Gidney–Ekerå techniques (within ~4 % of GE19's own construction at this size) | [superopt.md](research/superopt.md), [mbu-shor.md](research/mbu-shor.md), [ge-shor.md](research/ge-shor.md) |
| Shor under noise | one random depolarizing fault is fatal with probability 0.716 ± 0.004; success halves at 5.5 × 10⁻⁷ per location at 24 bits | [shor-noise.md](research/shor-noise.md) |
| Detector sampling vs Stim 1.16 | 9–12× faster than natively built AVX2 Stim on x86 (≥ 10⁵ shots), identical output distribution; Stim still wins small jobs | [fast-sampler.md](research/fast-sampler.md), [fast-sampler-audit.md](research/fast-sampler-audit.md) |
| Colour-code syndrome schedules | certified d = 9 schedule with circuit distance 8 (Kishony–Fowler: 7); flagged boundary circuits reach full distance d at d = 5, 7, 9 | [colour-global.md](research/colour-global.md), [colour-flags.md](research/colour-flags.md) |
| Simulability transition | exact simulation cost in monitored Clifford+T circuits has a transition at p_c = 0.1598 — the Clifford measurement-induced transition seen through a dephasing field h = η/n; explains ν_eff ≈ 2.45 = ν·y_h | [magic-transition.md](research/magic-transition.md), [transition-theory.md](research/transition-theory.md) |
| Engine planner | picks the fastest exact engine for expectations, samples and amplitudes: held-out regret 1.07–1.11, end-to-end 1.13 | [planner-v2.md](research/planner-v2.md) |
| Apple-silicon GPU | Metal state-vector backend (f32): 1.7–2.2× QFT, 2.4–3.2× brickwork over the best CPU path on an M1 Pro | [metal.md](research/metal.md) |

Context and caveats for every row are in the linked notebooks. In particular,
the Shor results are exact simulation of a compilable circuit whose cost grows
with the multiplicative order r; they are not a factoring speed-up, and the
largest gate-level Shor simulation we know of (Willsch et al. 2023, 39-bit N
on a GPU supercomputer) is larger.

## Theorems (with proofs and executable checks)

| topic | statement (short) | notebook |
|---|---|---|
| Shor support law | exact branch count per semiclassical round, explicit cancellation criterion, P(deficient) ≤ 4/r_odd | [theory-shor.md](research/theory-shor.md) |
| Borrowed magic | nullity of a branch state = affine dimension of its support minus a symmetry term; two-branch states stay stabilizer iff equal weight and phase ∈ {±1, ±i} | [theory-shor.md](research/theory-shor.md) |
| Shor noise windows | faults in the last ν₂(r) rounds are free (sharp); early phase faults bounded; exact Z-fault formula | [theory-shor.md](research/theory-shor.md) |
| Colour-code corner and boundary lemmas | any bare single-auxiliary corner or boundary plaquette forces circuit distance ≤ d − 1, for every odd d | [theory-colour.md](research/theory-colour.md) |
| Coset arithmetic | output total-variation error is linear in the misplaced fraction (not √); 31-bit padding c = 12 suffices for ≤ 1 % | [theory-coset.md](research/theory-coset.md) |

Each theorem has a test in `tests/` that fails if the statement is false.

## Engines

All engines share one circuit representation (`Circuit`, OpenQASM 2 and
`.stim` import/export) and are reached through one planner-routed entry point
(`pipeline::simulate`).

| engine | module | best for |
|---|---|---|
| cache-blocked state vector (AVX2 / NEON FMA, diagonal batching, dense 2-qubit fusion) | `statevector`, `blocked` | any circuit up to RAM |
| out-of-core state vector | `ooc`, `ooc_window` | states larger than RAM |
| Metal GPU state vector (f32, macOS, `--features metal`) | `metal_sv` | 20–29 qubits on Apple silicon |
| stabilizer tableau, SymPhase and FastSampler | `stabilizer` | Clifford circuits, QEC sampling |
| rotation frame / compressed state / magic recycling | `adaptive`, `pauli_frame`, `magic_atlas` | Clifford+T with low active dimension |
| sparse state vector | `sparse` | low-superposition circuits |
| bit-sliced reversible branches (with exact measurement-based uncompute) | `shor` | gate-level Shor and arithmetic |
| matrix product state | `mps` | low entanglement |
| Hybrid Schrödinger–Feynman | `hsf` | wide, shallow circuits |
| monitored Clifford+T | `monitored` | circuits with mid-circuit measurement |

Supporting tools: DAG compiler passes (peephole, light cone, components,
repeat-block fast paths, phase folding) in `compile` and `dag`; QEC circuits,
detector error models, an exact circuit-distance solver and a BP+OSD decoder in
`qec`; and the cost-model planner in `planner` and `mps_cost`.

## Building and running

```sh
cargo build --release
cargo test --release            # full suite; some audit tests take minutes

# examples
cargo run --release --example ghz
cargo run --release --example shor
cargo run --release --example surface_threshold

# CLI
cargo run --release -- run ghz --qubits 6 --backend stab
cargo run --release -- bench qft

# 31-bit gate-level Shor (~90 s and ~4.3 GB on an M1 Pro)
cargo run --release -- run shor --modulus 1537596787 --semiclassical --sliced \
    --window 4 --oracle windowed-mbu-lookup --f32 --seed 2 --tries 1
```

On macOS, `cargo build --release --features metal` adds the GPU backend.

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings` and the test suite. Merges to
`main` additionally require the full release suite to pass locally.

## Python quickstart

The `qsimlab` package (in [`python/`](python/)) exposes the engines to Python:
a typed `Circuit` builder, one `simulate()` entry point routed through the
planner, numpy results, and converters for Qiskit, Cirq, Stim and OpenQASM 2.
The full contract (conventions, errors, threading, stability) is
[python/API.md](python/API.md).

```sh
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy pytest
cd python && maturin develop --release     # or: pip install -e python
pytest                                     # tests + doctests
```

```python
import qsimlab as qs

c = qs.Circuit(3).h(0).cx(0, 1).cx(1, 2)             # GHZ; qubit 0 = least significant bit
qs.simulate(c, qs.statevector()).state               # complex128 ndarray, global phase kept
qs.simulate(c, qs.amplitudes(["000", "111"]))        # bitstrings: rightmost char = qubit 0
qs.simulate(c, qs.expectation(["Z0 Z2", "X0 X1 X2"])).values      # -> array([1., 1.])

r = qs.simulate(c.copy().measure_all(), qs.samples(10_000), seed=1, explain=True)
r.counts()          # {'000': ..., '111': ...}
r.engine            # engine the planner ran; r.explanation lists predicted costs
qs.plan(c, qs.samples(10_000))                       # predict only

# noisy Clifford circuits (e.g. from Stim) go to the batched SymPhase sampler
surface = qs.interop.from_stim(stim_circuit)         # detectors/observables kept
r = qs.simulate(surface, qs.samples(100_000))
r.parity(surface.detectors[0])

qs.simulate(c, qs.statevector(), engine="mps")        # force an engine
qs.interop.from_qiskit(qc); qs.interop.from_cirq(cc)  # optional dependencies
```

Wheels for Linux x86_64/aarch64 and macOS arm64 are built by
`.github/workflows/python-wheels.yml`.

## Research index

- **Shor:** [shor.md](research/shor.md) · [shor-r4-audit.md](research/shor-r4-audit.md) · [superopt.md](research/superopt.md) · [mbu-shor.md](research/mbu-shor.md) · [ge-shor.md](research/ge-shor.md) · [shor-noise.md](research/shor-noise.md) · [theory-shor.md](research/theory-shor.md) · [theory-coset.md](research/theory-coset.md)
- **QEC:** [qec.md](research/qec.md) · [qec-r4.md](research/qec-r4.md) · [schedules.md](research/schedules.md) · [colour-global.md](research/colour-global.md) · [colour-flags.md](research/colour-flags.md) · [theory-colour.md](research/theory-colour.md) · [fast-sampler.md](research/fast-sampler.md) · [fast-sampler-audit.md](research/fast-sampler-audit.md) · [stab.md](research/stab.md)
- **Simulability, magic and physics:** [simulability.md](research/simulability.md) · [planner.md](research/planner.md) · [planner-v2.md](research/planner-v2.md) · [magic-atlas.md](research/magic-atlas.md) · [magic-transition.md](research/magic-transition.md) · [transition-theory.md](research/transition-theory.md) · [adaptive.md](research/adaptive.md) · [pauli.md](research/pauli.md)
- **Performance:** [sv.md](research/sv.md) · [mac-m1.md](research/mac-m1.md) · [metal.md](research/metal.md) · [ooc.md](research/ooc.md) · [hsf.md](research/hsf.md) · [mps.md](research/mps.md) · [sv-monomial.md](research/sv-monomial.md) · [dense-fusion.md](research/dense-fusion.md)
- **Compiler:** [compiler.md](research/compiler.md) · [dag.md](research/dag.md) · [pipeline.md](research/pipeline.md) · [repeat.md](research/repeat.md) · [phasepoly.md](research/phasepoly.md)
- **Process:** [audit.md](research/audit.md) · [literature.md](research/literature.md) · [ARCHITECTURE.md](research/ARCHITECTURE.md) · [ARCHIVE.md](research/ARCHIVE.md) (archived branches)

## Corrections we have published

We keep wrong claims visible rather than quietly editing them away:

- "SymPhase is 4–6.5× faster than Stim" was wrong: Stim had been timed through its slow numpy path. On identical circuits it was parity; the later FastSampler is genuinely faster ([audit.md](research/audit.md) §13, [fast-sampler-audit.md](research/fast-sampler-audit.md)).
- A reported T-count of 340 omitted 250 equivalent phase gates; the true count was 590 ([audit.md](research/audit.md) §15).
- The "13× gap" to Gidney–Ekerå 2019 was mostly an artefact of comparing against their 2048-bit asymptotic formula ([ge-shor.md](research/ge-shor.md)).
- Several smaller corrections are logged in [audit.md](research/audit.md) §16.

## Licence

MIT, see [LICENSE](LICENSE).
