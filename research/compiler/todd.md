# T-count optimisation: TODD on Hadamard-delimited slots, with exact verification (`src/compile/todd/`)

Branch `exp/todd`, base `main` 6b21728, 5 Oct 2026. Machine: Intel Xeon Gold 6548Y+
(Hyper-V VM, 16 vCPU = 8 cores × 2 HT, 31 GB), shared with six other agents and another
user (1-minute load 6–55 during this work); nothing below is a timing claim except the
rough wall times, which are labelled as such.

## Headline

RESULTS_PLACEHOLDER

## Literature check (best published T-counts, October 2026)

Two rule sets are in use and must not be mixed:

* **Ancilla-free** (unitary, no extra qubits, no measurement). This is what this branch
  produces.
* **Hadamard gadgets** (one `|+>` ancilla per internal Hadamard, X-measurement and
  classically controlled Clifford corrections), used by TODD's main table, FastTODD's
  Table 1, AlphaTensor-Quantum (both of its columns), Polytof and VarTODD. AlphaTensor-
  Quantum's "with gadgets" column additionally charges Toffoli/CS gadgets as 2 T each and
  is a cost model, not a T-count.

Sources (all read in full; tables transcribed in `research/data/todd/published_best.csv`):

| source | what it reports |
|---|---|
| Amy, Maslov & Mosca, TCAD 2014 (arXiv:1303.2042) | T-par; origin of the benchmark suite |
| Heyfron & Campbell, QST 4 015004 (2018) (arXiv:1712.01557) | TODD (gadgets), TODD-part (ancilla-free Hadamard-bounded partitions) |
| Kissinger & van de Wetering, PRA 102 022406 (2020) (arXiv:1903.10477) | PyZX full_reduce + TODD, ancilla-free |
| de Beaudrap, Bian & Wang, TQC 2020 (arXiv:2004.05164) | STOMP/PHAGE, GF(2^n) |
| Ruiz et al., Nat. Mach. Intell. 7 374 (2025) (arXiv:2402.14396) | AlphaTensor-Quantum (gadgets) |
| Zen, Nägele & Marquardt, Nat. Mach. Intell. 8 113 (2026) (arXiv:2511.09951) | reusability report: small cases reproduced, nothing improved |
| Vandaele, Quantum 9 1860 (2025) (arXiv:2407.08695v2) | FastTODD, TOHPE: Table 1 (gadgets), Table 2 (ancilla-free partitions), Table 3 (GF) |
| Amy & Lunderville, POPL 2025 (arXiv:2410.23493) | non-linear phase folding (barenco records, ancilla-free) |
| Khoruzhii, Gelß & Pokutta, arXiv:2602.15285 (2026) | Polytof (gadgets; GF ancilla-free) |
| Fisher et al., arXiv:2603.29894v2 (2026) | VarTODD + LLM-guided search (GF ancilla-free; gadgets) |

LITERATURE_TABLE_PLACEHOLDER

Notes from the check that matter for comparisons:
* AlphaTensor-Quantum's "without gadgets" column still uses Hadamard gadgets (only
  Toffoli/CS gadgets are excluded); it is not ancilla-free.
* The GF(2^n) circuits have no internal Hadamard after Hadamard minimisation, so the
  GF numbers of AlphaTensor-Quantum, Polytof and VarTODD are ancilla-free.
* Feynman's own phase folding assumes non-input qubits start in `|0>`; a few Feynman
  numbers (grover_5 → 0, mod_adder_1024 923, ham15-high 985, ham15-med 210, qft_4 65)
  depend on that and are not unitary equivalences. This branch treats every qubit as
  an arbitrary input (full unitary equivalence).
* Heyfron & Campbell's Grover5 row is an older circuit (T 52); Feynman's current
  `grover_5.qc` has T 336.
