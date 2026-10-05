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

## Method

Code: `src/compile/todd/` (`mod.rs` slot model and Hadamard rewrites, `tensor.rs` TODD,
`pauli.rs` Pauli-frame mode, `verify.rs` checkers, `gf2.rs`), `src/io/qc.rs` (`.qc` reader and
writer), driver `examples/todd_bench.rs`, tests `tests/compiler/todd.rs`.

**Input.** The `.qc` circuits are read as written (`Z a b c` is a CCZ, `tof a b c` a Toffoli,
i.e. `H·CCZ·H` on the target). Every qubit is treated as an arbitrary input: all results
below are **full unitary equivalences, ancilla-free and measurement-free**; the `.i`
declaration of `|0>`-initialised ancillas is never used.

**Phase polynomial and slots (slot mode).** Every Hadamard stays in place and starts a
new path variable, so the circuit is a sum over paths whose phase is a sum of terms
`k·(p·v)` mod 8 (`ω = e^{iπ/4}`; `p` an affine parity of the inputs and path
variables). `scan` merges equal parities (phase folding, Amy–Maslov–Mosca) and computes
for each term the stretch of Hadamard-delimited *slots* in which its parity is a XOR of
wire values. That stretch is an interval (the part of the wire span over old variables
can only shrink at a Hadamard), so the odd terms (one T each) are assigned to slots by
greedy interval stabbing, then by a local search that moves a term to another slot in
its interval when deterministic TODD on the two slots involved gets cheaper. In each
slot the terms are columns `p_j ∈ F_2^n` in the coordinates of the wire values there.

**TODD** (`tensor.rs`; Heyfron & Campbell 2018). Columns with the same signature
tensor `Σ_j p_j⊗p_j⊗p_j (mod 2)` give the same unitary up to a diagonal Clifford. A step
`A → A + z yᵀ` keeps the tensor when `A y = 0`, `|y|` is even and
`A·diag(y)·Aᵀ = z wᵀ + w zᵀ` for some `w`; with `z = p_a + p_b` and `y_a ≠ y_b`, columns
`a` and `b` become equal and cancel (one comes back as `z` if `|y|` is odd). The system
in `(y, w)` is row-reduced once per step with the `w`-coefficients of every unit `z`
carried along, so each candidate `z` costs one `n`-column elimination (FastTODD's
idea). The exact diagonal-Clifford correction (S, Z and CZ on parities) is computed
from the linear and quadratic parts of the two term lists and emitted with the new
terms. `restarts > 0` adds randomised runs (random pair order and kernel vector) and
keeps the best.

**Hadamard rewrites** (`PhaseCircuit::reduce_hadamards`, slot mode, `--hred`): adjacent
`H·H` pairs cancel; `H(t)·[CNOT(c_i, t), X(t)]·H(t) = [CZ(c_i, t), Z(t)]` when every gate
between the two Hadamards that touches `t` is a CNOT targeting `t` or an X on `t`; and
`[H_S] B [H_T] = [H_{W∖S}] B' [H_{W∖T}]` for a CNOT/SWAP block `B` on wires `W` (`B'`
has every CNOT reversed) when it removes Hadamards (`|S| + |T| > |W|`). Fewer
Hadamards means longer slots.

**Pauli-frame mode** (`pauli.rs`, `--pauli`). The circuit is rewritten as
`U = ω^g · C · Π_j ω^{k_j (I - P_j)/2}`: every odd phase becomes a rotation whose axis
`P_j` is the Pauli string `C_{<j}† Z C_{<j}` in the *input* frame (inverse tableau
tracked gate by gate), and `C` is the circuit's Clifford skeleton. Rotations move past
rotations whose axes commute; equal axes merge (Zhang & Chen, arXiv:1903.12456);
Clifford rotations created by merging are pushed to the end, conjugating the
rotations they pass, and merging is repeated. Rotations are then grouped (greedy, both
"as early" and "as late as possible", the cheaper kept) into sets of pairwise
commuting axes that can be brought together; a local search moves a rotation to any
group strictly between the groups of its earlier and later anticommuting partners
(which keeps every anticommuting pair in order). Each group is diagonalised by a
Clifford `D` (CNOT/S/H elimination on the X-parts), TODD runs on the diagonal phase
polynomial, and the group is emitted as `D · (CNOT/T network) · D†`; the Clifford
skeleton follows. This mode sees through any Hadamard structure (its merged counts
equal PyZX's `full_reduce` on most circuits) but changes the Hadamards, so the
path-sum identity no longer applies and outputs are checked semantically (below).

## Verification

Every number in the results table is for an output circuit that was checked against
the input. Three independent checks exist; which ones apply is stated per row.

1. **Path-sum identity (slot mode, any size, exact).** `verify::path_sum` writes a
   circuit as `U|x> = 2^{-h/2} Σ_y ω^{P(x,y)} |f(x,y)>` with `P` in canonical
   multilinear form over `Z_8` (monomials of degree ≤ 3) and `f` affine. Equal `h`, `f`
   and `P` (up to the constant term, i.e. the global phase) prove the two circuits
   equal as unitaries on all `2^n` inputs. The checker shares no code with the
   optimiser's term bookkeeping (it expands every gate, including the CCZs and
   Hadamards of the input, from scratch). It is sufficient, not necessary, which is
   why it only applies when the Hadamards are kept.
2. **Exact basis-state simulation (both modes).** `verify::simulate_basis` runs a
   computational-basis input through a circuit with amplitudes in `Z[ω]` and one
   common power of `1/√2` (no floating point); two circuits agree on an input when the
   output states are equal up to `ω^j`, and `basis_equivalent` requires the *same*
   `j` for every input, so agreement on all `2^n` inputs is unitary equality. It is
   run on all `2^n` inputs for `n ≤ 16` and on 4096 random inputs above.
3. **Independent tools.** The repository's floating-point state-vector engine on
   random entangled input states (tests, `n ≤ 12`, tolerance 1e-9 including the global
   phase), and PyZX 0.10.0's dense tensor contraction (`compare_tensors`, for the
   circuits small enough) via `research/data/todd/pyzx_check.py`. PyZX's
   `verify_equality` (ZX rewriting of `U·V†`) is *not* usable here: it returned False
   for TODD outputs whose tensors PyZX itself finds equal (mod_mult_55, 9 qubits), so it
   is incomplete on these circuits, and a False from it means nothing.

The Hadamard rewrites of slot mode change the path variables, so `--hred` runs are
checked in two steps: the rewritten circuit against the original by exact basis-state
simulation, then the output against the rewritten circuit by the path-sum identity.

Corruption tests (`tests/compiler/todd.rs`): every single-gate mutation of a verified
output (T↔T†, S↔S†, dropped Z or X, Hadamard moved, CNOT reversed) that changes the
unitary is rejected by both exact checks, and an added global phase is reported, not
hidden.

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

CAVEATS_PLACEHOLDER

## Reproduction

```sh
export CARGO_TARGET_DIR=...            # any
cargo build --release --example todd_bench
cd research/data/todd/circuits
../fetch_circuits.sh                   # optional: hwb6, hwb8, gf2^16, gf2^32 (not redistributable)
B=$CARGO_TARGET_DIR/release/examples/todd_bench
# slot mode (Hadamards kept, path-sum verified), Hadamard rewrites, restarts, slot search
$B --hred --restarts 32 --passes 4 --seconds 60 --seed 1 --out out_slot --csv slot.csv *.qc
# Pauli-frame mode (exact basis-state verification: all inputs for n <= 20)
$B --pauli --restarts 16 --passes 3 --seconds 60 --lns 40 --seed 1 --out out_pauli --csv pauli.csv *.qc
# PyZX baseline and independent tensor check (pyzx 0.10.0)
python3 ../pyzx_check.py baseline *.qc
python3 ../pyzx_check.py verify . out_pauli qft_4      # verify_equality (incomplete, see Verification)
cargo test --release --test todd                       # exactness, corruption and regression tests
```

Raw results: `research/data/todd/results.csv` (per circuit and mode: T-counts, qubits,
Hadamards, CNOTs, verification method), `research/data/todd/pyzx_baseline.csv`,
`research/data/todd/published_best.csv` (literature table with sources), and the record
output circuits in `research/data/todd/outputs/` (`.qc`, readable by Feynman and PyZX).
