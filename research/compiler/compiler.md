# Compiler: exact circuit-level passes and plans (`src/compile/`)

Do less work before any backend runs. Every pass is exact: the optimised
circuit gives the same amplitudes (up to a tracked global phase) or the same
exact outcome distribution.

## Passes
- **Peephole**: cancels inverse pairs and merges rotations across gates that
  commute with them (axis rules in `compile/mod.rs`), not just adjacent ones.
- **SWAP elimination**: SWAPs become wire relabellings.
- **State propagation**: forward-tracks a product or stabilizer description
  of the state from |0…0⟩ and drops gates whose effect is known classically.
  For example, a QFT on |0⟩ becomes a layer of H gates, and
  Bernstein–Vazirani collapses entirely.
- **Light cone**: keeps only ops that can affect the measured or observed
  outputs, following classical edges from a measurement to the conditional ops
  that read it.
- **Independent components**: qubits that never interact are simulated
  separately, so the cost is 2^a + 2^b instead of 2^(a+b).
- **Monomial suffix**: a trailing network of X, CNOT, SWAP and phase gates
  before measurement maps basis states to basis states. It's applied
  classically to the samples instead of simulated.
- **Clifford-prefix absorption** (`stabsv.rs`): the leading Clifford segment
  runs on a tableau and converts to a state vector only when needed.
- **Plans**: `compile_sampling`, `compile_unitary` and `expectation_z_product`
  pick a backend per component (tableau, Pauli path or state vector).

## Integration with main's new ops and gates
- Reset, noise channels (X/Y/ZFlip, Depolarize1q/2q) and classically
  conditioned gates are **barriers** to the peephole pass on their qubits.
- The light cone keeps them, and it follows the **classical** dependency from
  a measurement to every `ClassicControlled` op that reads its bit.
- Plans either handle these ops exactly or fall back to the plain path
  (commits 3699605 and 9daa08d).
- New gates: I, Sx, Sxdg, U, ISwap and ISwapdg. The axis and monomial
  classification is exact, and Sx, Sxdg, ISwap and ISwapdg are treated as
  Clifford.

## Verification
- `tests/compile.rs` checks every pass and plan with proptests (256 and 192
  cases), against the state vector to 1e-12 or against exact outcome
  distributions.
  - Circuit families include mid-circuit measurement, reset, noise, classical
    control and the new gates.
  - Noisy circuits are checked by enumerating Pauli trajectories exactly.
- The independent audit of 49facf8 (before the port) passed. Worst amplitude
  error was 3.9e-14, plus chi-square tests of the samplers over ~900 plans.
  Its interleaved reproduction gave BV-23 745×, GHZ-24 148× and random
  Clifford+T 22q 6.2×.

## Results after the port (re-measured, load ~7.4–8.3; raw data in research/data/compiler/)

| workload | gates | always-SV (s) | compiled (s) | speedup | what removed the work |
|---|---|---|---|---|---|
| BV-23 | 61 | 0.315 | 0.0011 | 298× | 61→41 (peephole) →13 (state prop) →0 (cone) |
| GHZ-24 + measure | 24 | 0.112 | 0.0012 | 90× | cone→1 gate + classical CNOT suffix |
| QFT-20 + measure | 220 | 0.052 | 0.0023 | 22× | state propagation: QFT|0⟩ is a product state (220→20 gates) |
| rand Clifford+T n22 d30 | 990 | 0.918 | 0.096 | 9.5× | 990→849→592→556 gates |
| rand Clifford+T n20 d20 | 600 | 0.162 | 0.049 | 3.3× | 600→256 gates |
| rand Clifford+T n40, 3 measured | 360 | impossible | 0.0009 | — | cone → 8 gates in independent 1–2-gate tableau components |
| repetition code d7, 3 rounds + noise | 58 | 11.98 | 0.020 | 604× | Clifford components → tableau |
| teleport chain, 6 hops (feed-forward) | 26 | 0.255 | 0.220 | 1.2× | little to remove |
| Grover-11 (+8 ancillas) | 3441 | 0.515 | 0.543 | 0.9× | nothing removable (negative result) |

Honest negatives:
- Grover and the teleport chain gain nothing; the passes are overhead there.
- The baseline here is the plain gate-by-gate state vector. Against the
  blocked state-vector executor now on main (3–10× faster), the smaller ratios
  shrink accordingly. The structural wins (BV, GHZ, QFT+measure, 40-qubit,
  noisy repetition) don't, because they remove the work entirely.

## Next
- Rebuild these passes on the circuit DAG (exp/dag) so dependencies are
  computed once.
- Use the blocked executor as the state-vector backend inside plans.
