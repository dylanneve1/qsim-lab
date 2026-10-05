# Exact simulation of the 2025 RSA factoring circuit's approximate modular exponentiation (exp/approx-modexp)

Branch `exp/approx-modexp`, based on main `6b21728`; October 2026. Machine:
the shared 16-vCPU Xeon Gold 6548Y+ VM (31 GB; see `/dev/shm/qsim/MACHINE.md`),
builds and runs under the swarm's build throttle, `RAYON_NUM_THREADS ≤ 8`.
No timing claims are made here (the 1-minute load from other users was
5–30 during the runs); wall-clock seconds are quoted only to say what is
feasible.

Code: `src/shor/approx.rs` (precomputation port, resolved quint-level
program, all-branch evaluator, exact output distributions, gate-level loop4
step), `examples/approx_modexp.rs` (driver), `tests/shor/approx_modexp.rs`
(tests that fail if the statements below break). Data and scripts:
`research/data/approx-modexp/`.

**Source studied.** C. Gidney, *How to factor 2048 bit RSA integers with
less than a million noisy qubits*, arXiv:2505.15917 (2025), which builds on
the approximate residue arithmetic of Chevignard–Fouque–Schrottenloher
(ePrint 2024/222), Ekerå–Håstad period finding, superposition masking and
measurement-based uncomputation. The paper's code and assets are released
under **CC-BY-4.0** on Zenodo, doi:10.5281/zenodo.15347487 (`code.zip`,
sha256 `e627abdeb91e880ec8500a3015ab59eb09c3171e1c8f8d9c5eab96728064c94d`,
not committed here). We use its `facto/algorithm/_detailed_example_code.py`
(the reference `approx_modexp`: loop1–loop4, unloop3, unloop2, phase vents),
`facto/algorithm/prep/` (precomputation), `facto/algorithm/sim/main1_sample_masked_success_rates.py`
(success-rate model) and `scatter_script/` (its trajectory simulator).

## Headline

(filled in below)

## 1. What the paper verified, and what is new here

The paper's Appendix A.1: *"the simulator works by tracking the value and
phase of a few randomly sampled classical trajectories. This isn't
sufficient to verify interference effects, or to verify that information
wasn't incorrectly revealed (e.g. it can't verify that masking was done
correctly), but it fuzzes that the classical output is correct and that
phase kickback from measurement based uncomputation is being fixed."*
Its success-rate study (`main1_sample_masked_success_rates.py`, Figure 5)
samples from an idealised model of the output (exact `g^e mod N`, a mask,
a QFT modulo the period), not from the circuit.

Here the circuit is simulated on **every** branch: all `2^m` exponent
values times all `2^{mask}` mask values, every register value and every
sign, for given measurement outcomes, followed by the exact
frequency-basis measurement (QFT of the exponent register) of the
resulting state.

## 2. Method

### 2.1 The circuit, as the paper's code writes it

`approx_modexp` (paper, Appendix A.1 and `_detailed_example_code.py`) acts
on the exponent register `e` (`m` qubits, uniform superposition, never
modified) and an output accumulator of `f + 1` qubits initialised to the
mask `s ∈ [0, 2^{mask})`. For each prime `p_i` of the residue system `P`:

* **loop1** adds windowed discrete-log differences `D_i − D_{i−1}` (lookups
  addressed by `w1`-qubit windows of `e`) into a `ℓ + bitlen(m)`-qubit
  discrete-log accumulator; every lookup output is uncomputed by an X-basis
  measurement whose phase correction is XOR-ed into a per-window "vent"
  table that is applied once, at the very end, by a phaseup on `e`;
* **loop2** compresses the discrete log modulo `p_i − 1` by binary long
  division (constant subtractions and one-qubit GHZ lookups);
* **loop3** computes the residue `g_i^{dlog} mod p_i` by windowed modular
  multiplications into a helper register: subtraction of a lookup
  (addressed by `w3a` discrete-log qubits and `w3b` residue qubits), a
  GHZ-lookup add of `p_i` on underflow, an X-measurement of the underflow
  ("wrap") qubit whose phase correction is **deferred to unloop3**, and,
  after each window, an X-measurement of the **whole previous residue
  register** whose phase is also deferred;
* **loop4** adds the residue's truncated contribution
  `C = ((u_i k 2^{j w4} mod L mod N) >> t)` into the accumulator modulo
  `T = N >> t` with the same subtract/underflow/measure pattern; the wrap
  qubit's phase is fixed immediately by a comparison when its outcome is 1;
* **unloop3** recomputes every intermediate residue, fixes the deferred
  phases (CZ with the recorded measurement mask, comparisons conditioned
  on `phase_wrap ⊕ not_phase_wrap`, phaseups of vents shared between
  computation and uncomputation) and finally X-measures the residue;
* **unloop2** restores the discrete log for the next prime's loop1.

The last loop1 subtracts the last discrete log, the accumulator is
measured (`V`), and the exponent register is measured in the frequency
basis. For Shor-style period finding the exponent is one register with base
`g` (the paper's code); for Ekerå–Håstad the windows of register `a` use
base `g` and those of `b` use `y^{-1}`, `y = g^{(N−1)/2}` (the repo's
convention, `src/shor/ge.rs`; the paper describes EH but its code computes
`g^e`).

### 2.2 Precomputation (`ApproxConfig::new`)

A line-by-line port of `facto/algorithm/prep`: window multipliers, the
paper's prime-length estimate, its prime-set search (largest primes plus a
scanned pair, primes dividing the deviation pruned; the paper's parallel
search takes whichever worker reports first, ours takes the first pair in
order), smallest primitive roots `≥ 3`, discrete-log difference tables
(`uint32` wrap), loop3 tables (`table3a/b/c`) and loop4 tables (`table4`,
computed with the CRT identity `u_i c mod L = (L/p_i)((c·inv_i) mod p_i)`
so no big integers are needed). `xcheck_tables.py` runs the paper's own
table code on our prime sets and its own `_verify_rns_solution` on them.

### 2.3 Resolved program and all-branch evaluation

`plan` walks the program once with an outcome source (SplitMix64 seeds,
all-0, all-1, or a replayed log) and emits ~3–7 k resolved operations per
prime: lookups with their measurement outcome (per-branch kickback
`(−1)^{mx·T[addr]}`), constant and GHZ additions, X-measurements with
reset, phaseups with the final vent tables, comparisons, CZ with recorded
masks, global sign flips (`qpu.z(not_phase_wrap)`), and checks. The random
draws happen in exactly the order of the paper's code, so a log recorded
by the paper's code replays here.

`evaluate` runs the program on every `e < 2^m` (chunks of 1024, rayon) and,
for the accumulator operations, on every `s < 2^{mask}` at once. Per branch
it tracks five registers and a sign. It counts: non-zero ancillas at every
`del_by_equal_to(0)` and at the end, residue registers different from the
exact residue `(∏_j M_j) mod p_i` after loop3, final signs `−1`, and
accumulators different from `(s + F̃(e)) mod T`; it also checks `F̃(e)`
against the table formula (the windowed Eq. 20) and records the deviation
`δ(e) = F̃(e) − ⌊f(e)/2^t⌋ mod T` and the paper's modular deviation
`Δ_N(f(e) − F̃(e) 2^t)` for every `e`.

The X-basis measurements are exact in this branch representation because
each measured register is a function of the remaining registers (the
exponent and the accumulator determine the branch), so no branches merge
and every outcome has probability `2^{−len}`. The Python backend (§2.5)
checks this without assuming it.

### 2.4 Exact output distribution

With all signs `+1` and `acc(e, s) = (s + F̃(e)) mod T`, the state before
the output measurement is `(2^m W)^{−1/2} Σ_{e,s} |e⟩|(s + F̃(e)) mod T⟩`.
Measuring `V` and then the exponent in the frequency basis gives
`P(V, j) = |Σ_{e: (V − F̃(e)) mod T < W} e^{−2πi e·j/2^m}|² / (2^{2m} W)`
(two-dimensional for Ekerå–Håstad). `distribution` computes it with one
complex FFT per pair of `V` values (two real inputs share an FFT; the
marginal over `V` is symmetrised), `O(T 2^m m)`. The same routine gives the
**ideal masked** distribution (exact arithmetic, `F = ⌊f/2^t⌋ mod T`, same
mask) and the **unmasked** one (`F = f`, `W = 1`, `T = N`). `overlap` gives
the exact fidelity of the pre-measurement states, and `cond_fidelity` the
average fidelity of the post-measurement exponent state with its ideal
counterpart.

Post-processing: Shor-style outcomes are scored with the success test of
the paper's own model (`C.success_mask`: continued-fraction denominator `d`
of `j/2^m` limited to `N`, success iff `1 < gcd(g^{⌊d/2⌋} + 1, N) < N`)
and with the repo's order-finding post-processing (`shor::postprocess`,
which also tries multiples); Ekerå–Håstad (`s = 1`) outcomes with the
repo's lattice post-processing `ge::eh_postprocess` (every candidate
verified by `g^d = y` and `pq = N`). For `s > 1` one run cannot be scored
alone; we report the exact distribution of `α = {dj + 2^m k}_{2^{m+ℓ}}`
(the quantity the lattice post-processing needs small).

### 2.5 Independent checks

* **The paper's own code on a genuinely quantum backend**
  (`qbackend.py`). A drop-in `QPU` for the paper's `scatter_script`
  whose branches are the whole superposition (every exponent value × every
  mask value, complex amplitudes) and whose `mx_rz`/`del_measure_x` is a
  real projective X-basis measurement: branches that agree on every other
  live register are summed before taking `|·|²`, the outcome probability is
  computed, the state renormalised; merges are recorded. The paper's
  `approx_modexp` runs unchanged on it, with the paper's own tables for our
  prime set. Its outcome log is replayed by the Rust simulator and every
  branch compared; its final state's `P(V, j)` (numpy FFT) is compared with
  ours.
* **Tables.** `xcheck_tables.py`: every table entry of the Rust port equals
  the paper's code's output.
* **Gate level.** One loop4 window step, and one loop3 step together with
  its unloop3 counterpart (the deferred wrap-phase correction), are
  compiled to X/CNOT/Toffoli gates plus X-basis measurements and Z/CZ
  fix-ups with the repo's measurement-based blocks (`shor::mbu`: unary
  iteration lookups with measured ANDs, Gidney adders, phase comparator,
  phase lookups) and checked against the quint-level semantics on every
  input; the loop4 step also on a superposition with random complex
  amplitudes under real `H` + projection (`engines::sparse`).
* **The paper's success model** (`paper_model.py`): `main1`'s model with
  the paper's own `C` class, averaged exactly over every measured value
  instead of sampled; `model_vs_paper_csv.py` checks it against the
  paper's released Monte-Carlo data.

## 3. Results

### 3.0 The paper's success model, evaluated exactly

`paper_model.py` evaluates the paper's `main1` model with the paper's own
`C` class (signal, success mask) exactly over every measured value.
Averaged exactly over all bases `g` (as the paper samples them), it matches
the paper's released 200 000-shot Monte-Carlo data
(`assets/masked_success_stats.csv`) in all 24 (N, mask proportion) cells
checked (N = 15, 21, 35, 77, 143, 323; p = 0, 0.01, 0.1, 0.5): largest
difference 0.0009, binomial standard error of the paper's estimates ≈ 0.001
(`out/model_vs_paper_csv.txt`). So the model numbers quoted below are the
paper's model, not a re-interpretation of it.

### 2.6 Instances

Balanced semiprimes, base `g = 2` unless stated (the paper's Figure 4
instance uses `g = 3122`):

| n | N | factors | g | ord(g) | d = (p+q−2)/2 |
|---|---|---|---|---|---|
| 8 | 143 | 11 · 13 | 2 | 60 | 11 |
| 9 | 323 | 17 · 19 | 3 | 144 | 17 |
| 10 | 899 | 29 · 31 | 2 | 140 | 29 |
| 12 | 3127 | 53 · 59 | 3122 (Figure 4), 2 | 1508 | 55 |
| 14 | 11663 | 107 · 109 | 2 | 1908 | 107 |
| 20 | 1022117 | 1009 · 1013 | 2 | 11592 | 1010 |
| 24 | 16016003 | 4001 · 4003 | 2 | 2001000 | 4001 |

### 3.1 Independent checks of the simulator

* **Tables = the paper's code.** For five configurations (n = 8, 10, 12,
  14 Shor-style with windows (2,2,2,2), (4,2,3,4), (5,1,2,4), and n = 16
  Ekerå–Håstad) every entry of every table (discrete-log differences
  mod 2^32, `table3a/b/c`, `table4`) and every generator equals the output
  of the paper's own precomputation code for the same prime set; the
  paper's `_verify_rns_solution` accepts each of our prime sets; the
  paper's own (randomised) search succeeds at the same sizes with the same
  number of primes (`out/xcheck_tables.txt`).
* **The paper's own `approx_modexp` on a genuinely quantum backend**
  (`qbackend.py`, N = 143, m = 10, f = 6, mask 2^2, |P| = 6): 4 096 branches
  (2^10 exponent values × 4 mask values), 1 671 real X-basis measurements;
  **no two branches ever merged**, every outcome probability equals
  2^{−len} to machine precision, every register other than the exponent and
  the accumulator ends at 0, and all 4 096 final amplitudes are equal (+1,
  global phase included). The Rust simulator replaying the same 1 671
  outcomes agrees on every branch (0 of 4 096 accumulator values differ;
  its scalar per-branch evaluator agrees with the vectorised one), and the
  exact distribution `P(j)` of the frequency measurement agrees to
  **7·10⁻¹⁸** (`out/xcheck_qbackend_n8.txt`; this comparison is the test
  `distribution_matches_quantum_python_backend`).
* **Distributions = numpy.** `distribution` (1-D and 2-D paths) against
  numpy FFTs of the exact-arithmetic masked state: 9·10⁻¹⁸ (Shor, N = 899,
  m = 14) and 6·10⁻¹⁸ (Ekerå–Håstad, N = 899, 10 + 5 exponent qubits)
  (`out/xcheck_dist_numpy.txt`).
* **Gate level.** The loop4 window step compiled to gates (unary-iteration
  lookup with measured ANDs, Gidney subtractor on f + 1 bits,
  measurement-based unlookup, controlled load + adder for the GHZ add,
  X-measurement of the wrap qubit, comparison phase fix-up with its own
  lookup/unlookup, vent phaseup) equals `acc → (acc − T[k]) mod T_mod` with
  clean ancillas and one common sign on **every input** for 64/32/16 random
  tables and outcome streams at (w4, f, T_mod) = (2, 6, 61), (3, 8, 211),
  (4, 10, 781): 15 616 + 54 016 + 199 936 inputs, 0 mismatches
  (`out/gate.txt`; 59 Toffolis, 78 X-measurements per step at w4 = 4,
  f = 10). The loop3 inner step followed by its unloop3 counterpart — the
  pair whose wrap-qubit phase correction the paper defers from one
  subroutine to the other — compiles to the identity with clean ancillas on
  every input (tests `gate_level_loop3_pair_is_identity`). Both steps also
  run on superpositions with random complex amplitudes under real `H` +
  projective measurement, on the repo's sparse engine and on its
  independent dense reference state vector (`tests/audit_common`, 16 and
  18 qubits): the output is the expected permutation of the input up to one
  global phase, every measurement probability is 1/2.

### 3.2 The construction is correct on every branch

`approx_modexp verify` resolves the program for the all-0 and all-1
outcome streams and 8 random ones, and evaluates each on every branch
(`out/verify.txt`, table generated by `summarize.py`):

VERIFY_TABLE

On **every branch of every configuration and every outcome stream**:
every ancilla (discrete-log accumulator, both residue/helper registers, the
accumulator's wrap qubit) is 0 at each of the paper's `del_by_equal_to(0)`
points and at the end; every residue register equals the exact residue
`(∏_j M_j) mod p_i` after loop3; every branch ends with sign **+1** — all
phase kickback from the ~1.5–4 k X-basis measurements per run is cancelled
exactly, including the corrections deferred from loop3 to unloop3 (wrap
qubits, whole-register measurements of intermediate residues) and the
shared loop1 vent applied once at the very end; the accumulator equals
`(s + F̃(e)) mod T` for every mask value `s`, with the same `F̃` for every
outcome stream, and `F̃` equals the table formula. This is what the paper's
trajectory simulator fuzzes on a few random branches (its own verifier on
the n = 12 instance: 32 trajectories, 0.2 s, `out/paper_trajectories_n12.txt`);
here it holds on all 2^m · 2^{mask} branches (up to 1.07·10⁹ per stream).

**The deviation is a bias plus a random walk, far below the worst case.**
`δ(e) = F̃(e) − ⌊f(e)/2^t⌋` (accumulator units) is concentrated around a
configuration-dependent constant: each truncated addition rounds down, each
modulo-`T` wrap drops `N mod 2^t`, each modulo-`L` wrap adds `L mod N`. For
N = 899 (72 additions) `δ ∈ [−1, 21]` with mean 10.4 and an approximately
Gaussian spread (σ ≈ 2.7), against a worst case of order
`additions × (2·2^{−f} + 2^{−gap})·N` = 72 × 3 × 899/256 ≈ 760 (≈ 190 accumulator units of 2^t = 4) implied by the paper's deviation
model. Only the `e`-dependent part matters for period finding (a constant
shift of the output only relabels the measured value).

### 3.3 Interference: exact output distributions against the ideal ones

TODO_INTERFERENCE

### 3.4 Masking hides what it should

TODO_MASKING

### 3.5 Against the paper's success model

TODO_MODEL

### 3.6 Approximation error against parameters and the paper's bounds

TODO_SWEEPS

### 3.7 Moonshot: 20- and 24-bit moduli in the paper's error regime

TODO_MOON
