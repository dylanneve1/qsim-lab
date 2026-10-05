# `qsimlab.shor` — Shor's algorithm at gate level

**Status: provisional** (phase 2; see [`../API.md`](../API.md) §1, §8 and the `qsimlab.shor`
section). Every heavy call runs in Rust with the GIL released. The results are checked against
independent references in `python/tests/test_shor.py`: a numpy QPE distribution, a pure-Python
evaluator for the reversible oracles, factors and orders checked classically, and the numbers
recorded in the research notebooks.

| what | function | engine underneath |
|---|---|---|
| factor N (random or given bases, retries) | `factor(N, oracle=, window=, base=, precision=, seed=, tries=, budget=)` → `FactorResult` | semiclassical order finding (`qsim_lab::shor`), bit-sliced branch engine (`shor::sliced`), fused dense/sparse, GE windowed engine (`shor_ge`) |
| whole-run gate counts, no simulation | `resource_counts(N, oracle, base=, per_round=)` → `ResourceCounts` | the oracle builders (`shor_window`, `shor_superopt`, `shor_mbu`, `shor_ge`) |
| one controlled-`U_a` block as a `Circuit` | `oracle_circuit(N, a, oracle)` → `OracleCircuit` | same |
| the whole semiclassical circuit as a `Circuit` | `shor_circuit(N, a, oracle)` | same (runs through `qsimlab.simulate`) |
| support law, peak memory, work | `predict_support(N, a, oracle)`, `support_bounds(r, t)` | [theory-shor T1](../../research/theory/theory-shor.md#t1-the-support-law) |
| exact distribution of the measured integer | `exact_distribution(N, a, oracle)` | whole measurement tree (n ≤ 8 bits) |
| success under circuit noise | `noisy_success(N, a, p, noise=, trajectories=, faults=)` → `NoisySuccess` | exact Pauli-noise trajectories (`shor::noisy`) |
| classical helpers | `multiplicative_order(a, N)`, `carmichael(N)` | Pollard–Brent + Carmichael function |

The circuit is the one of [research/shor/shor.md](../../research/shor/shor.md): one recycled
control qubit (qubit 0), the work register on qubits `1 … n` prepared in `|1⟩`, and `t = 2n`
rounds of `H`, controlled `U^(2^(t−1−i))`, a phase correction conditioned on the bits measured so
far, `H`, measure. Measurement `i` is bit `i` of the measured integer `y`; its distribution is
exactly that of the textbook circuit with a `2n`-qubit counting register.

## Tutorial: factor a 20-bit N

```python
>>> import qsimlab as qs
>>> import qsimlab.shor as shor
>>> r = shor.factor(1_005_973, seed=1)
>>> r.factors, r.base, r.order
((997, 1009), 980062, 41832)
>>> r.oracle, r.engine, r.qubits, r.toffoli_gates
('windowed-opt(w=4)', 'sliced', 88, 70720)

```

That simulated every gate of an 88-qubit circuit with 271 522 gates (`r.total_gates`). The
measured integer is a 40-bit `y` with `y / 2^40 ≈ s / r`; the continued-fraction post-processing
recovered `r = 41 832`, and `gcd(a^(r/2) ± 1, N)` gave the factors. A failed run (odd `r`, or
`a^(r/2) ≡ −1`, or an unlucky `y`) is retried with a new base, up to `tries` runs:

```python
>>> run = r.runs[0]
>>> run.measured, run.measured.bit_length() <= 40
(787573303086, True)
>>> pow(run.base, run.order, 1_005_973), run.order == shor.multiplicative_order(980_062, 1_005_973)
(1, True)

```

With the same `seed`, `factor` draws the same bases and measures the same integers as the Rust
CLI `qsim run shor --modulus N --semiclassical --oracle windowed-opt --sliced --seed s`, and the
measured integer does not depend on the oracle or engine (they implement the same unitary; tested
on eight engine/oracle pairs). For example the round-4 runs of `research/shor/shor.md` are
reproduced exactly:

```python
>>> shor.factor(1_005_973, oracle="ripple", base=980_062, seed=1, tries=1).measured
475634978396
>>> shor.factor(1_005_973, oracle="windowed", base=980_062, seed=1, tries=1).measured
475634978396

```

## The support law and the memory guard

The cost of a run is set by the multiplicative order `r` of the base, not by `N`. Before round
`i` the state holds at most `B_i = min(2^i, r / gcd(r, 2^(t−i)))` basis branches
([theory-shor T1](../../research/theory/theory-shor.md#t1-the-support-law)); equality holds
except with probability ≤ 4/r_odd per round. Each run records its support trace next to the
prediction:

```python
>>> run.support_trace[:12].tolist()
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
>>> bool((run.support_trace == run.predicted_support).all()), run.peak_support
(True, 20916)
>>> p = shor.predict_support(1_005_973, 980_062)
>>> p.order, p.nu, p.peak, p.sum, p.predicted_bytes
(41832, 3, 20916, 170290, 1066716)

```

The peak support is `max(r_odd, r/2)` for even `r` (`r` for odd `r`), and the sliced engine's
work is about `2 · Ḡ · Σ_i B_i` gate·branch steps for `Ḡ` gates per round. `factor` evaluates
this **before** allocating anything and refuses a run whose predicted memory exceeds `budget`
(default: half the physical memory, at most 32 GiB). The 31-bit record instance of
`research/shor/shor.md` needs 6.5 GB in f64 (4.2 GB in f32, measured 4.28 GB):

```python
>>> try:
...     shor.factor(1_537_596_787, seed=2, budget="2GB")
... except qs.ResourceLimitError as e:
...     print(e.needed, e.limit)
...     print(str(e).split(", so")[0])
6534438750 2000000000
refusing to run: N = 1537596787 with base a = 457167243 has multiplicative order r = 256252500 (ν₂(r) = 2)

```

Bytes per stored branch are the measured peak RSS per peak support element of the research runs
(51 B in f64, 33 B in f32); the dense paths are predicted from their register size and the GE
engine from `2 · 2^{w_e} · r` branch slots. The order used by the guard is computed classically
(Pollard–Brent factoring and the Carmichael function); the simulated algorithm never sees it. For
a generic semiprime and a random base `r ≈ λ(N)/small`, so the work grows like `N · n³`: this is
exact simulation, not a factoring speed-up.

## Oracles and resource counts

`resource_counts` builds every controlled-`U` block of a run and counts its operations without
simulating anything. The numbers match `research/data/mbu-shor/counts.txt` exactly:

```python
>>> for o in ["ripple", "windowed", "windowed-opt", "windowed-mbu-lookup", "windowed-mbu", "ge", "eh"]:
...     c = shor.resource_counts(1_005_973, o, base=980_062)
...     print(f"{o:<20} {c.qubits:4} {c.rounds:3} {c.oracle_gates:9} {c.toffoli:8} {c.measurements:6}")
ripple                 64  40   1148298   415498      0
windowed               88  40    501808   176800      0
windowed-opt           88  40    271380    70720      0
windowed-mbu-lookup    88  40    224622    55169  14089
windowed-mbu          107  40    294892    31399  36839
ge                     90  40    191498    44239  13799
eh                     90  30    143641    33094  10344

```

* `ripple`: Cuccaro ripple-carry adders, `3n + 4` qubits. `windowed`: Gidney's windowed
  table-lookup multiplier ([shor.md, round 4](../../research/shor/shor.md)), `4n + 4 + w` qubits.
  `windowed-opt` (default): the same layout with the superoptimised blocks of
  [superopt.md](../../research/shor/superopt.md).
* `windowed-mbu-lookup` / `windowed-mbu`: measurement-based uncomputation
  ([mbu-shor.md](../../research/shor/mbu-shor.md)): X-basis measurements with Z/CZ fix-ups,
  fewer Toffolis. Their counts depend on the measurement outcomes; the engine draws them from a
  fixed stream seeded by `(N, multiplier)`, and so does `resource_counts`.
* `ge` / `eh`: Gidney–Ekerå exponent windowing (`exponent_window=w_e`, `window=w_m`, default 2
  and 3) for Shor's order finding, or the Ekerå–Håstad schedule
  ([ge-shor.md](../../research/shor/ge-shor.md)): `1.5n` exponent bits and lattice
  post-processing of the pair `(k, j)`, for balanced semiprimes (every prime factor below
  `2^⌈n/2⌉`).
* `beauregard`: QFT arithmetic, `2n + 3` qubits on a dense state (about 10-bit N).
  `permutation`: a classical lookup table instead of a circuit (`n + 1` qubits; measurement
  statistics only).

```python
>>> e = shor.factor(1_005_973, oracle="eh", seed=3)
>>> e.factors, e.measured, e.qubits
((997, 1009), (191, 146312), 90)

```

(`measured` is `(k, j)`: the `m`-bit register, which the engine runs first, then the `2m`-bit
one.)

One block of a unitary oracle is available as an ordinary circuit, with its control and work
qubits; every other qubit is an ancilla that starts and ends in `|0⟩`:

```python
>>> o = shor.oracle_circuit(21, 2, "windowed-opt", window=2)
>>> o.num_qubits, o.work, o.gates, o.toffolis
(26, [1, 2, 3, 4, 5], 556, 193)
>>> o.circuit.stats()["is_unitary"]
True

```

`shor_circuit(N, a, oracle)` returns the whole semiclassical circuit (measurements and classically
controlled corrections included), so it can be sampled, exported to OpenQASM or converted to
Qiskit. `qsimlab.simulate` runs it as a generic circuit (dense shots), which is only practical
for N ≈ 15; `factor` is the scalable path. For small N the exact distribution of the measured
integer is available directly:

```python
>>> import numpy as np
>>> p = shor.exact_distribution(21, 2, "windowed-opt")       # r = 6, t = 10
>>> len(p), round(float(p.sum()), 12)
(1024, 1.0)
>>> [int(y) for y in np.flatnonzero(p > 0.02)]                # peaks at y ≈ s · 1024/6
[0, 170, 171, 341, 342, 512, 682, 683, 853, 854]

```

## Noise

`noisy_success` runs exact Monte-Carlo trajectories of the gate-level circuit under circuit-level
Pauli noise ([shor-noise.md](../../research/shor/shor-noise.md)): a fault at rate `p` after
every oracle gate on each of its qubits, and on the control at preparation, after each `H` and
the phase correction, and as a readout flip. `faults=k` conditions every trajectory on exactly
`k` faults (the stratified estimate of the notebook):

```python
>>> s = shor.noisy_success(143, 5, 1e-4, trajectories=300, seed=1)
>>> s.order, s.locations, round(s.mean_faults, 1), round(s.peak_rate, 3)
(20, 102572, 10.1, 0.087)
>>> s0 = shor.noisy_success(143, 5, 0.0, trajectories=300, seed=1)
>>> s0.peak_rate
1.0

```

`peak_rate` is the fraction of "good" outcomes, `|y/2^t − s/r| < 1/(2r²)`, the metric of the
notebook. `success` (a factor was found) is uninformative for orders `r ≤ 256`: the
post-processing then recovers `r` from almost any `y` by its small-multiples search (here
`s.success` is 1.0 with ten faults per run).

## Performance

Timings are those of the Rust engine (the binding adds only the conversion of the result dict).
Measured on the Mac (M1 Pro, 8 threads unless stated) in the research notebooks:

| N (bits) | oracle | qubits | gates | time | peak RSS | source |
|---|---|---|---|---|---|---|
| 1 005 973 (20) | windowed, w = 4 | 88 | 0.50 M | 0.050 s | 9 MB | shor.md, round 4 |
| 221 643 407 (28) | windowed, w = 4 | 120 | 1.26 M | 16.1 s | 2.82 GB | shor.md, round 4 |
| 1 537 596 787 (31) | windowed-opt, f32 | 132 | 1.04 M | 97.6 s | 4.3 GB | superopt.md |
| 1 537 596 787 (31) | windowed-mbu-lookup, f32 | 132 | 0.87 M | 89.3 s | 4.28 GB | mbu-shor.md |
| 1 537 596 787 (31) | eh (2, 3), odd-order base | 134 | 0.55 M | 76.9 s | 4.79 GB | ge-shor.md |

Through the binding, `factor(1_005_973, seed=1)` (windowed-opt, f64, trace on) took 0.16 s on the
same Mac with 2 threads at load ≈ 8–19 from other jobs (not a benchmark).

## Limitations

* `N < 2^62`, odd, composite and not a prime power (`ValueError` otherwise). `eh` additionally
  needs a balanced N.
* The memory guard predicts the sliced engine from measured bytes per branch; the GE engine's
  envelope is conservative. The guard does not bound *time*: `predict_support(...).sum` times
  the gates per round estimates the work.
* `oracle_circuit` / `shor_circuit` cannot export the MBU oracles (their fix-ups depend on the
  parity of several measurement outcomes, which a `Circuit`'s single-bit `c_if` cannot express)
  or the GE schedules; `resource_counts` covers them.
* `exact_distribution` walks the full measurement tree: `n ≤ 8` bits (10 for the permutation
  oracle, 6 for Beauregard).
* `noisy_success` supports the reversible oracles (`windowed`, `windowed-opt`, `ripple`) up to
  129 qubits (n ≤ 30 for w = 4); a trajectory whose support exceeds `cap` is abandoned and
  counted as a failure (`capped`).
* `precision="f32"` is honoured by the sliced, fused-dense, dense and GE engines; the sparse
  engines (`engine="sparse"`, the permutation oracle's fused sparse state) compute in f64.
