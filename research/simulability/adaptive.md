# Adaptive representation switching (`src/adaptive.rs`)

The agent hit its time limit before writing this notebook; the parent wrote
it from the code, the tests, the raw data in `research/data/adaptive/` and the
agent's log.

## What it does
An exact simulator for Clifford+T circuits that changes representation as it
runs:
1. **Tableau** while the circuit is Clifford.
2. At the first non-Clifford gate, switch to the **rotation frame**: the
   Cliffords are absorbed, leaving Pauli rotations `exp(-iθQ/2)`.
3. The rotations act on an **active register** of dimension
   `d_k = dim span{x(Q_1..Q_k)} ≤ k`. This is the compression argument in
   research/performance/pauli.md. The frame's symplectic map gives an *exact* change of
   basis onto `d_k` qubits, and the remaining rotations are applied to a
   **dense state vector over only those qubits** (2^{d_k} amplitudes instead of
   2^n).
4. **Policies:** `frame` (never switch), `dense` (switch at the first
   rotation), and `auto`. `auto` uses a live growth-rate meter on the term
   count and switches when the projected term count exceeds the dense cost.
5. **Sampler:** samples of all n qubits come from the compressed state plus
   the Clifford frame. The Pauli-path engine can't do this.
6. A meet-in-the-middle mode for expectation values.

## Verification
`tests/adaptive.rs` checks amplitudes and expectation values against the
state vector at small n on random Clifford+T circuits, for every policy, plus
sampler distributions. Values in the benchmark tables agree across `legacy`,
`frame`, `dense`, `auto` and `sv` to ~1e-12.

## Results (shared VM, load 14–17; raw tables in research/data/adaptive/)

**Expectation values, random Clifford+T, n=30:**

| t | legacy Pauli paths | frame (pruned) | dense from the start | auto |
|---|---|---|---|---|
| 30 | 0.110 s (19k terms) | 0.0034 s | 0.76 s (26 dense qubits) | 0.0033 s |
| 40 | 6.76 s (1.19M terms) | 0.044 s | 1.13 s | **0.0042 s** (switches at k=5, 4 dense qubits) |
| 50 | aborted (>4M terms) | 0.018 s | 20.3 s | similar to frame |

**Against the full state vector, n=22:**

| t | full SV | adaptive (auto) |
|---|---|---|
| 16 | 9.98 s, 64 MiB | 0.0024 s, ~0 MiB |
| 40 | 24.2 s | 0.0052 s |

**Sampling all qubits:**
- n=50, t=25: 23 active qubits, 128 MiB, **5.7×10^5 shots/s**.
- n=64, t=25: 24 active qubits, 256 MiB, **5.3×10^5 shots/s**.

A state vector for 50–64 qubits is impossible.

**Where it doesn't pay:** "dense from the start" is the wrong policy when
pruning keeps the term count tiny (t=50: 20 s dense vs 0.018 s frame). That's
why `auto` exists. When `d_k` approaches t (circuits with no structure), the
dense register is 2^t and the method is no better than its parts.

## Novelty: honest assessment
The brief claimed nothing switches representation adaptively at runtime. The
agent's literature search **found close prior art**:
- **Clifft: Fast Exact Simulation of Near-Clifford Quantum Circuits**
  (arXiv:2604.27058, 2026). It factors the state into an offline Clifford
  frame, an online Pauli frame and a **dynamically sized active state
  vector**. That's essentially steps 1–3 above.
- Related: CAMPS (arXiv:2412.17209); STABSim (arXiv:2507.03092); the
  compute-and-compress results of Jozsa–Van den Nest and
  Yoganathan–Jozsa–Strelchuk.

What looks less covered is `auto`: switching between the *pruned* Pauli-path
representation and the compressed dense register, driven by a live term
growth-rate meter. That's a policy contribution, not a new method. It should
be benchmarked against Clifft directly before claiming anything stronger.

## Next
- A head-to-head against Clifft's published numbers or code.
- Wire it into `src/compile` plans as a backend option.
- Drive switch points from the DAG.
