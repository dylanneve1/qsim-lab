# `qsimlab.analysis` — magic, stabilizer rank, simulability, monitored circuits

**Status: provisional** (phase 2; see [`../API.md`](../API.md) §1, §8 and the
`qsimlab.analysis` section). These functions explain *why* a circuit is easy or hard to simulate
exactly, with the invariants the engines themselves use. Most of them never build a state vector.
Everything is checked against independent references in `python/tests/test_analysis.py`: brute
force over all Pauli strings, the active dimension recomputed in pure Python, dense numpy states,
and Born probabilities of a numpy simulation forced to the same outcomes.

| what | function | engine underneath |
|---|---|---|
| active dimension `d_k`, factored dimension `f_k`, skeleton entanglement, support bound | `magic_profile(c)` → `MagicProfile` | `qsim_lab::magic_atlas::profile` ([magic-atlas.md](../../research/simulability/magic-atlas.md)) |
| stabilizer nullity, stabilizer 2-Rényi entropy (`n ≤ 13`) | `state_magic(c_or_state)` → `StateMagic` | `magic_atlas::state_magic` (all `4^n` Pauli expectations, one Walsh–Hadamard transform per `x`) |
| number of stabilizer terms after every gate | `branching_rank(c)` → `BranchingRank` | `qsim_lab::stab_rank` ([theory-rank.md](../../research/theory/theory-rank.md)) |
| per-engine work estimates + the planner's explanation | `simulability(c, request)` → `Simulability` | `simulability::features`, Planner v2 |
| Clifford+T with mid-circuit measurements: `d(t)`, Born probabilities, cut entropies | `monitored(c)` → `MonitoredResult` | `qsim_lab::monitored` ([magic-transition.md](../../research/simulability/magic-transition.md)) |

## The rotation frame in one paragraph

Lower a unitary circuit to Clifford gates and Z rotations and push every rotation to the front
through the Clifford part: `U = C · R_m ⋯ R_1` with `R_j = exp(−iθ_j Q_j/2)` and Pauli axes `Q_j`.
Acting on `|0^n⟩`, only the x-parts of the axes create superposition, so
`U|0^n⟩ = C (|φ⟩ ⊗ |0⟩)` where `|φ⟩` lives on `d = dim span{x(Q_j)}` qubits: the **active
dimension**, the exact register size of the compressed-state engine (`2^d` amplitudes). The
rotations only couple the coordinates they touch, so `|φ⟩` factorises; `f` is the largest
factor. The stabilizer nullity of the final state is at most `d`.

## Tutorial: the magic profile of QFT vs Grover

A 12-qubit QFT applied to a basis state, and Grover search for `|1…1⟩` on 8 qubits with a
Toffoli-ladder oracle (6 ancillas), 2 iterations:

```python
>>> import math
>>> import qsimlab as qs
>>> import qsimlab.analysis as an
>>> def qft(n, x):
...     c = qs.Circuit(n)
...     for q in range(n):
...         if x >> q & 1:
...             c.x(q)
...     for j in reversed(range(n)):
...         c.h(j)
...         for k in reversed(range(j)):
...             c.cp(k, j, math.pi / 2 ** (j - k))
...     return c
>>> def grover(n, iterations):
...     s, anc = list(range(n)), list(range(n, 2 * n - 2))
...     c = qs.Circuit(2 * n - 2)
...     def mcz():                     # Z on |1...1> of s, via a Toffoli ladder into anc
...         c.ccx(s[0], s[1], anc[0])
...         for i in range(2, n - 1):
...             c.ccx(s[i], anc[i - 2], anc[i - 1])
...         c.cz(anc[n - 3], s[-1])
...         for i in reversed(range(2, n - 1)):
...             c.ccx(s[i], anc[i - 2], anc[i - 1])
...         c.ccx(s[0], s[1], anc[0])
...     for q in s:
...         c.h(q)
...     for _ in range(iterations):
...         mcz()                      # oracle
...         for q in s:
...             c.h(q).x(q)
...         mcz()                      # diffusion
...         for q in s:
...             c.x(q).h(q)
...     return c
>>> for name, c in [("qft", qft(12, 0b101100111)), ("grover", grover(8, 2))]:
...     p = an.magic_profile(c)
...     print(f"{name:<7} n={p.num_qubits} T={p.t_count:3} rot={p.rotations:3} d={p.d:2} f={p.f:2} "
...           f"log2_work={p.log2_work:5.2f} factored={p.log2_work_factored:5.2f}")
qft     n=12 T= 33 rot=198 d=11 f= 1 log2_work=14.45 factored= 8.04
grover  n=14 T=336 rot=336 d=14 f=14 log2_work=22.22 factored=22.22

```

Both circuits reach almost full active dimension, but the QFT of a basis state factorises
completely (`f = 1`: its output is a product state, a 2-amplitude register per qubit), while
Grover's state is one 14-qubit block from the first oracle on. The same profile is what the
compressed-state engine pays: `2^14.45` amplitude updates for the QFT in the unfactored frame and
`2^8.04` factored, against `2^22.22` for Grover.

The profile is a trace, so you can see *when* the magic arrives: Grover's register is full
(`d = 14`) by the first checkpoint, after 31 gates, in the first oracle's Toffoli ladder:

```python
>>> p = an.magic_profile(grover(8, 2), checkpoints=4)
>>> [(gate, d, f) for gate, rot, t, d, f, e in p.checkpoints]
[(30, 14, 14), (61, 14, 14), (92, 14, 14), (123, 14, 14)]
>>> int(p.d_profile[0]), int(p.d_profile[15]), int(p.rotation_gate[15]), int(p.d_profile[-1])
(1, 7, 10, 14)

```

State magic needs the dense state (`n ≤ 13`). For the QFT of a basis state the nullity is
`n − 2` ([magic-atlas.md](../../research/simulability/magic-atlas.md), finding 4: nullity is not
cost) even though the state is a product of single-qubit states:

```python
>>> m = an.state_magic(qft(8, 0b1011001))
>>> m.nullity, round(m.m2, 4)
(6.0, 1.5728)
>>> an.branching_rank(qft(8, 0b1011001)).rank
64

```

## Branching rank

`branching_rank` runs the exact low-rank simulator and records how many stabilizer terms the
state needs after every gate. A non-Clifford gate splits a term only when it cannot be applied as
a Clifford on that term, and equal rays merge again:

```python
>>> c = qs.Circuit(3).h(0).t(0).cx(0, 1)
>>> an.branching_rank(c).trace.tolist()
[1, 2, 2]
>>> an.branching_rank(c.copy().t(1)).trace.tolist()[-1]   # (|00> + i|11>)/√2 is a stabilizer state
1
>>> b = an.branching_rank(grover(5, 2))
>>> b.max_rank, b.rank, b.overflow
(5, 2, False)

```

## Simulability and the planner

`simulability` returns the planner's features (one log2 work estimate per engine) and its
explanation (the engines ranked by predicted seconds, from cost models fitted on an M1 Pro):

```python
>>> s = an.simulability(qft(30, 12345))
>>> s.features["d"], s.features["chi_bits"], s.explanation.engine
(29, 15, 'mps')
>>> [e for e, _ in s.log2_costs][:3]
['pauli-frame', 'hsf', 'compressed']

```

## Monitored Clifford+T circuits

`monitored` simulates circuits with mid-circuit `Z` measurements exactly in the rotation frame.
A measurement is either random in the frame (probability 1/2, no amplitude work), determined, or
Born-sampled on the active register, which then shrinks by one qubit; `d(t)` does not depend on
the outcomes:

```python
>>> import numpy as np
>>> rng = np.random.default_rng(1)
>>> c = qs.Circuit(8)
>>> for layer in range(12):
...     for q in range(layer % 2, 7, 2):
...         _ = c.h(q).cx(q, q + 1).t(q + 1)
...     for q in range(8):
...         if rng.random() < 0.3:
...             _ = c.measure(q)
>>> r = an.monitored(c, seed=7, cuts=[[0, 1, 2, 3]], entropy_every=len(c) // 4)
>>> r.final_d, r.max_d, len(r.outcomes), r.stats["t_activating"], r.stats["meas_register"]
(3, 7, 29, 30, 27)
>>> [(k, round(lo, 3), round(hi, 3), round(s2, 3)) for k, [(lo, hi, s2)] in r.entropies]
[(37, 0.0, 0.0, 0.0), (75, 0.0, 3.0, 0.0), (113, 0.0, 3.0, 0.0), (151, 0.0, 3.0, 0.415), (154, 0.0, 2.0, 0.0)]
>>> bool((an.monitored(c, seed=8).d == r.d).all())        # same d(t) for other outcomes
True

```

`exact=False` keeps only the tableau: `d(t)` and entropy bounds for any `n`, without amplitudes.

## Limitations

* `magic_profile`, `branching_rank` and `simulability` take unitary circuits (terminal
  measurements are dropped; mid-circuit measurements, resets, conditionals and noise raise
  `UnsupportedOperationError`); `monitored` takes them all, but not `readout_error`.
* `state_magic` enumerates all `4^n` Pauli expectations: `n ≤ 13`.
* `branching_rank` stops at `max_terms` (`overflow=True`); its rank is an upper bound on the
  stabilizer rank, not the rank itself. `state=True` builds a dense vector (`n ≤ 20`).
* `monitored` returns the state up to a global phase (it does not track it); the exact Rényi-2
  entropy `s2` is computed only when the cost `2^(d + ...)` stays under `2^max_cost_log2`
  (else `None`, with the bounds still exact). One call is one trajectory; with `exact=False`,
  register outcomes are drawn 50/50, so `c_if` on them is not physical.
* Planner predictions are relative (fitted on one machine, single thread), see `API.md` §9.
