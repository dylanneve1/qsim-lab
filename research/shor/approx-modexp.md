# Exact simulation of the 2025 RSA factoring circuit's approximate modular exponentiation (exp/approx-modexp)

Branch `exp/approx-modexp`, based on main `6b21728`; October 2026. Machine:
the shared 16-vCPU Xeon Gold 6548Y+ VM (31 GB; see `/dev/shm/qsim/MACHINE.md`),
builds and runs under the swarm's build throttle, `RAYON_NUM_THREADS ≤ 8`.
No timing claims are made here (the 1-minute load from other users was
5–55 during the runs); wall-clock seconds are quoted only to say what is
feasible.

Code: `src/shor/approx.rs` (precomputation port, resolved quint-level
program, all-branch evaluator, exact output distributions, gate-level loop4
step and loop3/unloop3 step pair), `examples/approx_modexp.rs` (driver), `tests/shor/approx_modexp.rs`
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

HEADLINE_TEXT

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
all-0, all-1, or a replayed log) and emits ~300–600 resolved operations per
prime (1.5–6.5 k per run): lookups with their measurement outcome (per-branch kickback
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

| N | mode | m | f | mask | ℓ | \|P\| | additions | branches / stream | streams | bad sign | dirty | bad residue | bad acc | max\|δ\| | mean\|δ\| | δ range | max Δ_N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 143 | Shor | 10 | 6 | 2 | 11 | 6 | 36 | 4,096 | 10 | 0 | 0 | 0 | 0 | 7 | 3.0996 | [-1, 7] | 1.9580e-1 |
| 899 | Shor | 20 | 8 | 7 | 11 | 12 | 72 | 134,217,728 | 10 | 0 | 0 | 0 | 0 | 21 | 10.4224 | [-1, 21] | 9.1212e-2 |
| 3127 | Shor | 22 | 10 | 8 | 11 | 8 | 24 | 1,073,741,824 | 10 | 0 | 0 | 0 | 0 | 10 | 2.3783 | [-6, 10] | 1.2472e-2 |
| 3127 | EH | 18 | 10 | 8 | 11 | 8 | 24 | 67,108,864 | 10 | 0 | 0 | 0 | 0 | 9 | 1.8935 | [-6, 9] | 1.0873e-2 |
| 11663 | EH | 21 | 12 | 9 | 11 | 8 | 24 | 1,073,741,824 | 5 | 0 | 0 | 0 | 0 | 10 | 2.8848 | [-5, 10] | 3.4296e-3 |

(Streams: all-0, all-1 and random seeds 1–8; the N = 11663 run was stopped
after 5 streams to free the machine. δ: `F̃(e) − ⌊f(e)/2^t⌋` in accumulator
units; Δ_N: the paper's modular deviation `Δ_N(f(e) − F̃(e)·2^t)`, which
includes the bias.)

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

Three distributions of the frequency-basis outcome are computed exactly
for each instance (`out/dist.txt`): the **approximate circuit** (this
simulation), **exact arithmetic with the same mask and truncation**
(`F = ⌊f/2^t⌋ mod T`), and the **textbook unmasked** distribution
(`f` measured exactly: Shor's / Ekerå–Håstad's ideal). TV(approx., exact
arith.) isolates the residue-arithmetic approximation, TV(exact arith.,
unmasked) the mask plus truncation, TV(approx., unmasked) the total.

| instance | mask (S) | 1 − F (best shift) | TV(approx., exact arith.) | TV(exact arith., unmasked) | TV(approx., unmasked) | P(success): approx. / exact arith. / unmasked | P(peak 0): approx. / exact arith. / Eq. 42 (w/P) |
|---|---|---|---|---|---|---|---|
| N = 899, Shor, m = 20, f = 8 | 2^7 (0.57) | 0.030 | **0.021** | 0.565 | 0.584 | 0.179 / 0.186 / 0.421 | 0.5717 / 0.5717 / 0.5695 |
| N = 3127, Shor (Fig. 4 instance), m = 22, f = 10 | 2^8 (0.33) | 0.010 | **0.0070** | TVEU3127 | TVAU3127 | 0.406 / 0.410 / 0.605 | PEAK3127 |
INTERFERENCE_ROWS

(Masks from the paper's rule `S = √ε` (capped at f − 1 bits); success with
the paper's model's test; for Ekerå–Håstad (s = 1) with the repo's lattice
post-processing. The repo's order-finding post-processing, which also tries
multiples of the convergent denominators, succeeds on every outcome at these
sizes and is not informative.)

What the numbers say:

* **The approximation is invisible at the level of the frequency peaks.**
  With the paper's mask, the approximate circuit's distribution is within
  TV INTERF_TV_RANGE of exact arithmetic with the same mask — one to two
  orders of magnitude below the rigorous trace-distance bound
  `√(1 − F)` — and its success probability is within INTERF_SUCC_RANGE of
  exact arithmetic's. The per-peak structure (probability of each
  frequency peak k ≈ j r/2^m) matches exact arithmetic to TV ≤ 0.002, and
  the paper's randomised-remainder prediction for the zero peak (Eq. 42:
  `E|β_0|² = w/P`, with `w/P ≈ S`) holds to within 0.4 % (N = 899: 0.5717
  measured, 0.5695 predicted); the other peaks fluctuate between 0.04× and
  1.4× their mean, the "dips" visible in the paper's Figure 4.
* **What costs success is the mask itself**, as the paper's Assumption 2
  says: masked and unmasked distributions are far apart (TV ≈ 0.56 at
  S = 0.57) and the success factor is ≈ 1 − S (N = 899: 0.186 / 0.421 =
  0.44 at 1 − S = 0.43).
* **No information leaks through the measurements.** No branches merge in
  any X-basis measurement (§3.1), so every outcome is uniform and
  independent of the exponent; the exponent register's post-measurement
  state given the measured output has average fidelity INTERF_CF_RANGE
  with the exact-arithmetic one (after the constant output shift), and
  every measured output value of the approximate circuit is also a possible
  output of exact arithmetic (P(V outside the ideal support) = 0).

### 3.4 Masking hides what it should

The paper's simulator cannot see this; the exact one can. N = 899
(29·31), Shor-style, m = 14, f = 8 (t = 2, T = 224), |P| = 9, 54
accumulator additions; `δ` has mean 7.3, σ = 1.8, range [1, 14]
(`out/sweep_mask_n10_shor_m14.csv`; test `masking_restores_interference`):

| mask bits | W | 1 − F (best shift) | √(1 − F) (trace-distance bound) | TV(approx., exact arithmetic) | P(success) approx. | P(success) exact arith. |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.953 | 0.976 | **0.778** | 0.0506 | 0.1304 |
| 1 | 2 | 0.835 | 0.914 | 0.641 | 0.0638 | 0.1283 |
| 2 | 4 | 0.580 | 0.761 | 0.429 | 0.0841 | 0.1274 |
| 3 | 8 | 0.324 | 0.569 | 0.227 | 0.1029 | 0.1253 |
| 4 | 16 | 0.170 | 0.412 | 0.113 | 0.1118 | 0.1230 |
| 5 | 32 | 0.087 | 0.295 | 0.057 | **0.1125** | 0.1183 |
| 6 | 64 | 0.044 | 0.210 | 0.029 | 0.1002 | 0.1032 |

(F: the exact overlap of the pre-measurement states, maximised over a
constant output shift, here 7; success with the paper's model's test; m = 14
is below 2n, so absolute success values are low.)

* **Without masking the approximate arithmetic destroys most of the
  interference**: the frequency distribution is at TV 0.78 from the
  exact-arithmetic one and the success probability drops 2.6×, although
  every branch individually is computed correctly (§3.2). Measuring the
  unmasked approximate output reveals `δ(e)`, which is not periodic in
  `e`: exactly the "information incorrectly revealed" that the paper's
  simulator cannot check.
* **With the mask the damage is linear in 1/W**: TV halves per mask bit and
  equals ≈ 0.65 (1 − F), i.e. it tracks the infidelity itself, not the
  trace-distance bound √(1 − F) (the same linear behaviour the repo found
  for the coset representation, `research/theory/theory-coset.md`).
  1 − F ≈ 2 E|δ − c| / W.
* **The paper's trade-off is real but sits elsewhere**: the success
  probability of the approximate circuit peaks at W = 32 (S = W 2^t / N =
  0.14), where it is 0.95 of exact arithmetic with the same mask; larger
  masks lose to the mask itself. The paper's rule (S = √ε with its
  worst-case ε = 0.63 here) asks for the largest mask allowed (2^7),
  because its ε is the worst case (≈ 142 accumulator units against an
  actual max |δ| of 14 and spread σ = 1.8).

The same sweep for **Ekerå–Håstad** (s = 1), N = 3127, g = 3122, f = 10,
windows (3, 2, 3, 4), 12 + 6 exponent qubits, A = 24 additions
(`out/sweep_mask_f10.csv`):

| mask bits | S = W·2^t/N | paper ε/S | 1 − F (no shift) | 1 − F (best shift) | TV(approx., exact arith.) | P(success) approx. / exact arith. |
|---|---|---|---|---|---|---|
| 0 | 0.001 | 72.00 | 0.978 | 0.9444 | 0.7200 | 0.211 / 0.907 |
| 1 | 0.003 | 36.00 | 0.914 | 0.8116 | 0.5809 | 0.346 / 0.907 |
| 2 | 0.005 | 18.00 | 0.710 | 0.5434 | 0.3752 | 0.545 / 0.908 |
| 3 | 0.010 | 9.00 | 0.417 | 0.2994 | 0.1947 | 0.720 / 0.908 |
| 4 | 0.020 | 4.50 | 0.223 | 0.1564 | 0.0974 | 0.815 / 0.909 |
| 5 | 0.041 | 2.25 | 0.115 | 0.0798 | 0.0488 | 0.864 / 0.911 |
| 6 | 0.082 | 1.12 | 0.058 | 0.0403 | 0.0245 | 0.891 / 0.915 |
| 7 | 0.164 | 0.56 | 0.029 | 0.0203 | 0.0123 | 0.911 / 0.922 |
| 8 | 0.327 | 0.28 | 0.015 | 0.0102 | 0.0062 | 0.932 / 0.937 |
| 9 | 0.655 | 0.14 | 0.007 | 0.0051 | 0.0031 | 0.965 / 0.968 |

TV again halves per mask bit (≈ 0.61 (1 − F)), from 0.72 without a mask to
0.006 at the paper's mask (2^8). The success column uses the repo's EH
lattice post-processing with its default candidate budget (4096); at these
toy sizes that budget can enumerate every `d < 2^m` for outcomes near
`j = 0`, which is why exact-arithmetic success *rises* with the mask (the
mask enhances the zero peak, which fails in Shor-style post-processing and
in EH at real sizes). The scale-free comparison is the TV column and the
α statistics of §3.3; the approximate circuit's success approaches exact
arithmetic's (0.932 vs 0.937 at 2^8) as TV → 0.

MASK_SHOR_TEXT

### 3.5 Against the paper's success model

The paper's Figure 5 / Assumption 2 rest on `main1`'s model: exact
`g^e mod N`, a mask of width `W` in units of N, and a QFT taken modulo the
period. The table compares it, evaluated exactly (§3.0), with the exact
circuit at the same mask width (`W = 2^{mask}·2^t` in units of N; the
circuit's frequency measurement is the full `2^m`-point QFT with m = 2n − 2
= 22 ≥ 2·log2(r)), for the paper's own Figure 4 instance N = 3127,
g = 3122, f = 10, windows (4, 2, 3, 4) (`model_compare.py`,
`out/sweep_mask_f10_shor.csv`, `out/model_n12.txt`):

MODEL_TABLE

At the paper's own mask choice for this instance (2^8, S = 0.33):

| | P(success), unmasked | P(success), masked | suppression | 1 − S |
|---|---|---|---|---|
| paper's model (mod-P QFT, exact `f`) | 0.7029 | 0.4782 | 0.680 | 0.673 |
| circuit, exact arithmetic (2^22-point QFT) | 0.6048 | 0.4098 | 0.678 | |
| circuit, approximate (this simulation) | — | 0.4057 | 0.671 | |

The **suppression factor caused by masking agrees with the paper's model to
0.4 % (exact arithmetic) and 1.4 % (approximate arithmetic)**, and both
circuit values are within 1 % of the paper's Assumption 2 (`1 − S`). The absolute values differ
by the model's idealisation, not by the approximation: the model's QFT is
taken modulo the period, the circuit's over 2^22 exponent values
(2^22 ≈ 1.8 r²), whose finite resolution costs 14 % of the unmasked success
under the paper's success test (0.605 against 0.703).

MODEL_SWEEP_TEXT

### 3.6 Approximation error against parameters and the paper's bounds

**The paper's Eq. 28 omits a factor 2.** Eq. 28 states
`1 − |⟨ψ1|ψ̃1⟩|² ≤ ε/S`. Conditioned on `e`, the two states are uniform
superpositions over windows of width `SN` offset by at most `εN`, so
`|⟨ψ1|ψ̃1⟩| ≥ 1 − ε/S` and hence `1 − |⟨ψ1|ψ̃1⟩|² ≤ 2ε/S − (ε/S)²`; a
uniform offset attains it (dense numpy check: offsets of 1, 5, 10, 25 on
width-100 windows give infidelities 0.0199, 0.0975, 0.190, 0.4375 against
the stated bound 0.01, 0.05, 0.10, 0.25; `eq28_check.py`). Carried through
Eq. 43–45 this gives `P_deviant ≤ S + 2ε/S`, `S = √(2ε)`,
`P_deviant ≤ 2√(2ε)`; at the paper's n = 2048 point (P_deviant = 1.25 %)
the expected number of shots would move from 9.21 to 9.25–9.27 — negligible
for the paper's conclusions. The simulated circuits do not come near the
bound: their deviations are neither uniform nor worst-case (next
paragraphs), so the exact infidelity is far below `ε/S` with the paper's
ε (at the paper's own mask choice: 1 − |⟨ψ1|ψ̃1⟩|² = 0.156 against
ε/S = 1.69 for N = 899; 0.018 against 0.28 for N = 3127 Shor-style; 0.015
against 0.28 for N = 3127 Ekerå–Håstad).

**The deviation: a constant bias plus a random walk of rounding errors.**
From the every-branch runs (§3.2), with `A` = number of accumulator
additions (`|P|·⌈ℓ/w4⌉`):

| instance | A | mean δ (bias) | σ(δ) | σ/√A | E\|δ − c\| | δ range | paper's worst case (units) |
|---|---|---|---|---|---|---|---|
| N = 143, Shor, f = 6 | 36 | 3.07 | 1.58 | 0.26 | 1.22 | [−1, 7] | ≈ 60 |
| N = 899, Shor, f = 8 | 72 | 10.42 | 2.39 | 0.28 | 1.91 | [−1, 21] | ≈ 190 |
| N = 3127, Shor, f = 10 | 24 | 2.26 | 1.67 | 0.34 | 1.31 | [−6, 10] | ≈ 55 |
| N = 3127, EH, f = 10 | 24 | 1.65 | 1.64 | 0.34 | 1.30 | [−6, 9] | ≈ 55 |
| N = 11663, EH, f = 12 | 24 | 2.85 | 1.58 | 0.32 | 1.22 | [−5, 10] | ≈ 51 |
MOON_SIGMA_ROWS
(units of `2^t`; c = the rounded mean; worst case = `A·(2·2^{−f} + 2^{−gap})·N/2^t`.)

The spread is σ ≈ 0.26–0.34 √A, i.e. that of a sum of `A` independent
uniform rounding errors (√(A/12) = 0.29 √A); the mean is the net of the
truncations' rounding down, the `N mod 2^t` dropped at each modulo-`T`
wrap and the `L mod N` added at each modulo-`L` wrap. The quantity that
matters for interference, `E|δ − c|`, is 1.2–1.9 units, against a paper
worst case of 55–190 units at these sizes. *If* the rounding errors stay
this independent at scale, the paper's worst-case ε (∝ 3A) overstates the
relevant deviation by a factor growing like √A — consistent with the
paper's own remark that "analyzing the distribution of deviations rather
than focusing on the worst case deviation" could improve its bound. We
have not checked this beyond n = 24 and make no claim about the 2048-bit
regime.

SWEEP_DETAILS

### 3.7 Moonshot: 20- and 24-bit moduli in the paper's error regime

The paper's design point has P_deviant of a few percent (1.25 % at
n = 2048). At n ≤ 14 the accumulator cannot be long enough for that
(`f < n` is needed for truncation to happen at all), so the moonshot uses
larger moduli with the paper's own structure and parameter rules scaled
down: Ekerå–Håstad with the tradeoff parameter `s` chosen so that the
exponent register has m = 20 qubits (the paper uses s = 8 at 2048 bits),
windows (w1, w3a, w3b, w4) = (4–5, 2, 3, 4) (the paper: 6, 3, 3, 5),
`gap = f`, and the paper's mask rule `S = √ε`:

| instance | s | exponent qubits | f | t | T | mask | S | ℓ | \|P\| | A | paper ε | paper P_deviant bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n = 20, N = 1 022 117 = 1009·1013 | 2 | 15 + 5 | 14 | 6 | 15 970 | 10 | 1/16 | 11 | 9 | 27 | 4.9·10⁻³ | 14 % |
| n = 24, N = 16 016 003 = 4001·4003 | 3 | 16 + 4 | 16 | 8 | 62 562 | 11 | 1/32 | 11 | 13 | 39 | 1.8·10⁻³ | 8.8 % |

MOON_RESULTS
## 4. Caveats and negative results

* **Not the paper's regime.** The paper's design point is n = 2048,
  f = 33, ℓ = 21, |P| ≈ 2·10⁴ primes, ~10⁵ accumulator additions,
  P_deviant ≈ 1.25 %. Here n ≤ 24, f ≤ 18, ℓ = 11–14, |P| ≈ 5–15,
  30–90 additions. The paper's prime search refuses fewer than 100
  candidate primes of length ℓ, which forces ℓ ≥ 11 even for 8-bit N, so
  residues are as long as N itself at the smallest sizes and the number of
  additions per prime is larger relative to n than at 2048 bits. The
  approximation is therefore *coarser* here than in the paper's regime
  (ε ≈ 10⁻³–10⁻¹ instead of 10⁻⁵): a construction error would be easier,
  not harder, to see; but numbers like success probabilities do not
  extrapolate.
* **A corner case of the paper's prime search (toy sizes only).** The
  search prunes a chosen prime `q` that divides the candidate `L mod N`,
  assuming `(L/q) mod N = (L mod N)/q`; that needs `gcd(q, N) = 1`. When a
  factor of N is itself an ℓ-bit prime (possible only for n ≤ 2ℓ, i.e. toy
  instances), the pruned set can violate the deviation constraint: for
  N = 4001·4003 with 12-bit primes the pair (2213, 4003) gives a pruned
  candidate 55 < N >> 18 = 61 while the true deviation of the pruned set is
  3 024 811, reproduced with the paper's own `prune`
  (`rns_prune_check.py`). The paper's code then stops at its final
  `_verify_rns_solution` assertion; our first port lacked that final check
  and returned the invalid set (caught here because the reported `L mod N`
  violated the constraint). The port now re-verifies the pruned set and
  excludes primes dividing N (which would also put a factor of N into the
  tables). Irrelevant at RSA sizes.
* **Prime sets differ.** The paper's prime search is parallel and keeps the
  first worker's answer; ours scans pairs in order. The tables are
  identical to the paper's code *for the same prime set*
  (`xcheck_tables.py`), and the paper's own search succeeds at the same
  sizes, but its primes and hence `F̃` differ in detail.
* **Ekerå–Håstad.** The paper describes EH but its code computes `g^e`; we
  build `g^a y^{-b}` by changing the window multipliers (`y = g^{(N−1)/2}`,
  the repo's convention, `d = (p + q − 2)/2`) — the circuit itself is the
  paper's. Single-run success is scored only for `s = 1` (the repo's
  lattice post-processing); for `s > 1` we report the exact `α`
  distribution, not a multi-run success probability.
* **Frequency-basis measurement.** Simulated as an exact QFT. The paper's
  semiclassical implementation with a truncated phase-gradient state adds a
  rounding error (≤ nπ/2^g) and a state-preparation infidelity (< 10⁻⁵) that
  are not simulated.
* **Outcome streams.** Every branch is checked for the outcome streams run
  (all-0, all-1 and 8 random seeds per configuration; 2 seeds in sweeps and
  the moonshot). A correction that is wrong only for a specific
  combination of outcomes would be missed; a wrong correction term flips the
  sign of a branch for half of all streams, so each random stream misses a
  given error with probability ≤ 1/2.
* **Gate level.** Only two subroutine steps are compiled to gates (one loop4
  window step; one loop3 step with its unloop3 counterpart), with the
  repo's unary-iteration lookups and Gidney adders rather than the paper's
  power-product lookups/phaseups and lattice-surgery GHZ lookups (different
  costs, same function). The rest of the circuit is simulated at the quint
  level, i.e. as the paper's code writes it.
* **Measurement-based uncomputation.** The Rust evaluator relies on each
  measured register being a function of the remaining registers (argued in
  §2.3 for every measurement of the program). The genuinely quantum Python
  backend checks it without assuming it, but only at the cross-check size.
* **No noise.**

## 5. Reproduction

```sh
export CARGO_TARGET_DIR=...                       # optional
cargo build --release --example approx_modexp
cargo test --release --test approx_modexp         # the headline statements (≈ 1 min)
# the paper's release (CC-BY-4.0, doi:10.5281/zenodo.15347487), unzipped:
export GIDNEY_SRC=/path/to/release/src            # numpy, sympy needed
bash research/data/approx-modexp/run_experiments.sh xcheck   # tables, quantum backend, numpy, model
bash research/data/approx-modexp/run_experiments.sh verify   # every branch, 10 outcome streams
bash research/data/approx-modexp/run_experiments.sh dist     # exact distributions, success
bash research/data/approx-modexp/run_experiments.sh sweep    # mask / f / window sweeps (CSV)
bash research/data/approx-modexp/run_experiments.sh gate     # gate-level loop4 step
bash research/data/approx-modexp/run_experiments.sh model    # the paper's model, exactly
bash research/data/approx-modexp/run_experiments.sh moon     # n = 20, 24
python3 research/data/approx-modexp/summarize.py             # tables of §3.2 / §3.7
python3 research/data/approx-modexp/model_compare.py         # table of §3.5
python3 research/data/approx-modexp/plot_results.py          # figures
```

Single runs: `approx_modexp verify|dist|sweep|gate|config|dump|replay key=value...`
(see the header of `examples/approx_modexp.rs`). Seeds: outcome streams are
SplitMix64 from `seed=` (default 1; `verify` uses all-0, all-1 and
`seed..seed+seeds`). Every run reported here uses `RAYON_NUM_THREADS=8`
and < 2 GB of memory.
