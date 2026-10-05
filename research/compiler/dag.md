# Circuit DAG IR (`src/dag.rs`)

The owner asked for this on 1 Oct: compile circuits into a graph that can be
optimised. The agent hit its time limit before writing this notebook; the
parent wrote it from the agent's code, tests and raw data
(`research/data/dag/`).

## What it is
- Nodes are ops: gates, measurements, resets, classically conditioned gates
  and noise channels.
- **Wire edges** run from the previous op on each qubit.
- **Classical edges** run from a measurement to every conditional op that
  reads its bit.
- **Record edges** keep measurement order, so the output bit indices stay the
  same.
- `Dag::from_circuit` / `to_circuit` round-trips exactly.
- Queries: wire predecessors and successors, front layer, layers,
  reverse-reachability light cone (follows classical edges), components, and
  exact commutation of neighbouring nodes (cached).
- Rewrites: remove (splice wires), replace, merge, and slide past a commuting
  neighbour.
- Commutation-aware peephole: a worklist that runs to a fixpoint and
  re-examines wire predecessors after each rewrite, so nested cancellations go
  in one sweep, with an optional look-back.

## Verification (`tests/dag.rs`)
- Round trip and topological validity on random circuits with every op type.
- The DAG light cone, components and peephole are A/B'd against `src/compile`
  (`light_cone`, `components`, `optimize`) and against the state vector:
  amplitudes are checked up to tracked global phase, and outcome distributions
  exactly.
- `U·U†` cancels to nothing. A completeness test covers the named gates.

## Results: gate counts (raw: `research/data/dag/counts.txt`)

| circuit | gates | `Circuit::optimize()` (adjacent) | DAG adjacent-only | DAG commutation-aware |
|---|---|---|---|---|
| random Clifford+T n=8 d=50 | 600 | −1.0% | −12.8% | **−19.3%** (T 87 → 48) |
| random Clifford+T n=24 d=200 | 7,200 | −0.1% | −13.4% | **−20.8%** (T 1451 → 691) |
| random Clifford+T n=50 d=400 | 30,000 | 0% | −13.1% | **−19.7%** (T 5033 → 2620) |
| QFT lowered to Clifford+T, n=32 | 2,528 | 0% | 0% | **−18.4%** |
| Grover (Clifford+T), n=8 | 1,512 | 0% | −6.7% | **−9.4%** |
| QFT·QFT† | 1,088 | −100% | −100% | −100% |

- Commutation-aware cancellation removes about **20% of the gates, and about
  half of the T gates**, on random Clifford+T circuits, where the adjacent-only
  `optimize()` removes ~0–1%. Halving the T count matters most for the
  Pauli-path engine, whose cost is exponential in T.
- **Honest comparison:** the DAG peephole gives **gate-for-gate identical
  results** to `compile::optimize` on every benchmark. Both are
  commutation-aware. The DAG version is the shared infrastructure; it doesn't
  find more yet.

## Speed (ns per op, min of 5, `research/data/dag/timing_v1.txt`, load ~8)
At 192k ops:
- `from_circuit`: 25 ns
- `to_circuit`: 17 ns
- DAG light cone: 123 ns
- DAG peephole: 596 ns
- `compile::optimize`: 158 ns

The DAG peephole is about 3–4× slower than the flat compile pass for the same
result. That's the cost of a general graph, and it's the first optimisation
target.

## Next
- Port `src/compile` passes to share the DAG so dependencies are computed
  once, and make the DAG peephole competitive.
- Add rewrites the flat pass can't express: commuting non-adjacent rotations
  into phase polynomials, and template matching.
- Schedule backend switches (adaptive) and out-of-core global/local qubit
  swaps off the DAG's front layer.
