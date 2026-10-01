# Lab notebook: surface-code syndrome-extraction schedule search

Machine: shared 4-vCPU AMD EPYC-Rome VPS (AVX2), 7.7 GB RAM, other agents
compiling concurrently. All timed runs go through `qsim-swarm/bench.sh`
(flock). Rust stable, release profile (`debug=0`). Branch `exp/schedules`.

---

## §0  Prior art and the exact question

### §0.1  Hook errors and the origin of schedule constraints

The rotated planar surface code (Kitaev 2003; Fowler et al., *Phys. Rev. A*
86, 032324, 2012; Horsman et al., *New J. Phys.* 14, 123011, 2012) extracts
X- and Z-type stabilizers in each syndrome round using a CNOT schedule.
Dennis et al. (*J. Math. Phys.* 43, 4452, 2002) identified *hook errors*: a
single fault on a syndrome ancilla after the k-th CNOT of a weight-4 check
spreads to the last (4 − k) data qubits. When that pair of data qubits lies
along the *same* direction as the logical operator, the hook fault is a
single-shot logical error; the effective code distance drops to roughly
(d + 1) / 2.

Tomita & Svore (*Phys. Rev. A* 90, 062320, 2014) systematically analysed
CNOT ordering for the rotated surface code under circuit-level depolarizing
noise and showed that the "Z" or "N" orderings (named after the path traced
across the plaquette) avoid hook errors for both X- and Z-type checks
simultaneously, giving true circuit distance d. The key constraint is that
the hook pair of each check must be perpendicular to *its own* logical
direction: horizontal hook pairs are safe for Z-checks (which correct X
errors along the vertical direction); vertical hook pairs are safe for
X-checks (which correct Z errors along the horizontal direction) — but in a
Z-basis memory experiment only the X-check hook matters.

The current codebase (`src/qec/surface.rs`, comments in
`generate_stabilizers`) implements exactly this: Z-checks use NW, NE, SW, SE
(hook pair is vertical SW–SE, perpendicular to Z_L = column 0 for a
Z-memory experiment); X-checks use NW, SW, NE, SE (hook pair is vertical
NE–SE, perpendicular to the X_L direction). The qec.md notebook (§3)
verified that the graph-like circuit distance equals d for d = 3, 5, 7, 9.

Google's surface-code experiment papers (Acharya et al. 2022, 2023) use the
same "N-type" ordering principle validated at large scale under realistic
hardware noise.

### §0.2  XZZX code and bias-tailored schedules

Bonilla Ataides et al. (*Nat. Commun.* 12, 2172, 2021) showed that the XZZX
surface code — a local Clifford conjugation of the standard surface code that
turns all plaquettes from pure X or pure Z to mixed XZZX type — performs
dramatically better under Z-biased noise. Under pure Z-bias (η → ∞) the XZZX
code behaves like a repetition code along one axis, and the threshold under
biased noise *grows* as η increases. Crucially, the schedule for the XZZX
code is different from the rotated surface code: the Hadamard layers are
rearranged and the CNOT targets alternate between X and Z roles within each
plaquette.

Dua et al. (*PRX Quantum* 3, 040308, 2022) and Tuckett et al. (*Phys. Rev.
Lett.* 120, 050505, 2018; *Phys. Rev. X* 9, 041031, 2019) studied biased
noise decoders and schedules for other code families under Z-biased noise.

**The standard rotated surface code with standard (N/Z) schedules was not
specifically optimised for biased noise.** There is no published systematic
search over the 4! = 24 CNOT orderings per check type for the rotated
surface code under biased circuit-level noise. Prior art considers either
(a) the standard code with unbiased noise or (b) the XZZX code, which is a
*different* code with a different Clifford frame. The question of whether a
*different CNOT ordering of the standard rotated surface code* — without
changing stabilizer types or logical frame — can improve the Z-memory logical
error rate under biased noise is unanswered.

### §0.3  Automated schedule synthesis (2022–2026)

Baireuther et al. (*npj Quantum Inf.* 4, 48, 2018) and Chamberland et al.
(*Quantum* 4, 291, 2020) used reinforcement learning and circuit synthesis to
find better syndrome circuits for specific hardware, but targeted gate depth
and connectivity constraints, not noise-model-dependent reordering of a fixed
connectivity set.

Battistel et al. (*PRX Quantum* 4, 030314, 2023) used MCTS to optimise QEC
circuits for biased noise, but on different code families and small codes.
No published work has exhaustively searched the CNOT orderings of the
standard rotated surface code at d = 3 under biased circuit-level noise and
measured logical error rate with held-out validation.

### §0.4  The exact question this notebook addresses

**Can any of the 24 × 24 = 576 combinations of per-check-type CNOT orderings
for the rotated surface code syndrome extraction beat the standard hook-safe
order (NW,NE,SW,SE for Z-checks; NW,SW,NE,SE for X-checks) on logical error
rate per round under circuit-level Z-biased noise (bias η = p_Z/p_XY from 1
to 100), at equal circuit depth and code distance d = 3?**

Two subsidiary questions:
1. How many of the 576 orderings preserve the full circuit distance d = 3?
2. Among those, does logical error rate under high bias differ significantly
   from the standard order?

A negative result (no schedule beats the standard) is acceptable and
informative. A positive result must pass held-out validation.

---

## §1  Parameterisation and schedule space

For the rotated surface code, each weight-4 plaquette has 4 data-qubit
neighbours (NW, NE, SW, SE in the dual-grid notation). A *schedule* assigns a
permutation of these 4 neighbours as the CNOT order. There are 4! = 24
permutations per check type. Boundary plaquettes have only 2 or 3
neighbours; their orderings are a subset.

Implementation: `src/qec/schedules.rs` — see §2.

Constraints:
- Within a given schedule, all Z-checks use the *same* permutation and all
  X-checks use the *same* permutation (schedule classes; more general
  per-plaquette orderings exist but are left for future work).
- The full search is 24 × 24 = 576 combinations.
- Filter: only keep orderings where the circuit-derived graph distance = d
  (`DecodingGraph::min_logical_weight`).

Naming: a permutation is written as a 4-tuple of the compass labels in
application order, e.g., `[NW, NE, SW, SE]`. Index encoding: NW=0, NE=1,
SW=2, SE=3.

---

## §2  Distance filter results

(Filled in after running the search binary.)

---

## §3  Logical error rate search (d = 3)

### §3.1  Unbiased noise (η = 1)

### §3.2  Z-biased noise (η = 10)

### §3.3  Z-biased noise (η = 100)

---

## §4  Validation

### §4.1  Fault-injection distance check on winners

### §4.2  Tableau cross-check at d = 3

---

## §5  Results and novelty statement

(Filled in after completing runs.)

---

## Appendix: exact commands

```
# Build
CARGO_BUILD_JOBS=2 cargo build --release --example schedule_search 2>&1

# Run distance filter + full search
./target/release/examples/schedule_search d=3 rounds=3 shots=200000 seed=42

# Run held-out validation
./target/release/examples/schedule_search d=3 rounds=3 shots=200000 seed=99 --validate
```
