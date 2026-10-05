# Code discovery: exhaustive search of weight-6 two-block (BB / GB / coprime-BB) codes, n ≤ 300

Branch `exp/code-discovery`.
- Code:
  - `src/qec/bicycle.rs`: two-block codes over `Z_l × Z_m`, GF(2) rank for k, exact distance, ISD upper bound.
  - `src/qec/bb_search.rs`: enumeration up to equivalence.
  - `src/qec/bb_circuit.rs`: depth-7 syndrome circuits and schedule validity.
  - `examples/bb_codes.rs`: CLI with `params`, `search`, `schedules`, `schedsearch`, `cdist`, `ler`.
- Tests: `tests/bicycle_codes.rs` pins published parameters. There are also unit tests in all three modules, including a brute-force check that the enumeration is complete.
- Data and scripts: `research/data/code-discovery/`. `literature.md` is a 341-entry table of published BB/GB/2BGA codes from 21 papers. It was compiled by a sub-agent, with every polynomial copied verbatim.
- Compute: all runs were on the Mac (M1 Pro) with ≤ 2 workers and no GPU. The Mac was shared, with load 8–16.

**Question.** Can a search over code families, scored with our fast exact tools, find quantum LDPC codes that beat known ones per physical qubit under circuit-level noise? The baselines are IBM's bivariate-bicycle codes and the 2024–26 code tables.

**Answer.**
- **Code capacity: no code in this space beats the literature's best k·d²/n at any n.** The space is every weight-6 two-block group-algebra code over every abelian group of rank ≤ 2, with n ≤ 300 — 147,107 inequivalent classes with k > 0.
  - Wherever our search decided the frontier, it matches the maximum k·d²/n of the published tables. The published frontier is mostly Liang et al. 2503.03827, IBM, and Lin–Pryadko.
  - It also reproduces every published entry in the space that it decided.
  - Two Pareto points are not in any table we found and are not matched by direct sums of published codes: **[[168,14,10]]** and **[[300,16,14]]** (polynomials below). Both have lower k·d²/n than their published neighbours.
  - A third code, **[[112,12,8]]**, has the same parameters as two copies of the published [[56,6,8]].
- **Circuit level: the same parameters do not mean the same performance.**
  - Setup: depth-7 CNOT schedules (IBM's generalised), uniform circuit noise, BP+OSD-CS(10), Z memory.
  - The connected **[[112,12,8]] has 4–6× lower logical error per logical qubit than the published [[56,6,8]]** (that is, than two copies of it, which is the same qubit overhead n/k = 9.3 and the same d):
    - p = 0.3%: 2.7e-3 vs 1.1e-2 block failures per round;
    - p = 0.2%: 4.2e-4 vs 2.5e-3.
  - **No depth-7 schedule of this shape keeps d_circ = 8 for [[56,6,8]].** 75 of the 936 valid schedules do for [[112,12,8]].
  - [[168,14,10]] has the same n/k = 12 as the gross code. At p = 0.3% it has 5.0e-5 failures per logical qubit per round against the gross code's 6.6e-5. The 95% intervals barely overlap. Its d is 10 against 12, so the gross code must win at lower p; we could not afford the shot counts to show the crossover.
- **Tools.**
  - Exact distance via connected-cluster branch and bound, with the group's translation symmetry used to fix the root. It takes 21 ms for the gross code [[144,12,12]], 4.7 s for [[288,12,18]] and 32 s for [[294,10,20]].
  - Exact circuit distance uses the same search on the circuit DEM with translation orbits as roots. It proves d_circ = 6 for IBM's [[72,12,6]] circuit in 0.1 s (IBM gives ≤ 6) and d_circ ≥ 9 for the gross-code circuit (IBM gives ≤ 10).

__DETAILS__
