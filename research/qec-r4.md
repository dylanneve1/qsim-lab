# QEC round 4: identical-circuit Stim comparison, and colour-code schedule search

Branch `exp/qec-r4`. Raw data and scripts: `research/data/qec-r4/`.

Machines:
- **VPS**: 4 vCPU AMD EPYC-Rome (AVX2), shared with other agents. Timings are only quoted when the 1-minute load was ≤ 4, and the load is recorded per row.
- **Mac**: Apple M1 Pro, 8 cores, NEON. Every timing run held `/tmp/qsim-mac-bench.lock` and waited for 1-minute load < 3; the starting load is recorded.

Everything is single-threaded unless stated. Stim is 1.16.0 throughout.

---

## Part 1. Is the SymPhase detector sampler really faster than Stim?

### 1.1 What was wrong with the earlier comparison

`RESULTS.md` and `research/audit.md` §13 say SymPhase beats Stim by **4.0–6.5×** on rotated surface-code memory. Two problems:

1. **The .stim file was a hand-written reconstruction.** `examples/stim_export.rs` re-implements the circuit line by line instead of serialising the `Circuit` object that is actually sampled. It happened to match, but nothing enforced that.
2. **Stim was timed through its slowest output path.** Stim was timed with `sampler.sample(shots, bit_packed=True)`, which builds a shot-major numpy array and transposes it. qsim-lab was timed writing 64-shot words into a scratch buffer that was immediately discarded. Stim's own streaming path, `sample_write(..., format="ptb64")`, emits exactly qsim-lab's layout (64-shot words per detector). On the VPS it is **2–4× faster** than `sample()`. Most of the claimed lead came from comparing different output paths.

### 1.2 New tooling: `src/stim_io.rs`

- **`to_stim(circuit, noise, detectors, observables)`** walks the real `Op` list and makes every implicit noise location of the `NoiseModel` explicit, in the order `run_noisy` and SymPhase use:
  - `DEPOLARIZE1`/`DEPOLARIZE2` after each gate;
  - `M(p)` for readout flips;
  - `X_ERROR` after `R`.

  Ops on disjoint qubits are merged into one line (`CX 0 1 2 3`). That is the same circuit, and it is fair to Stim.
- **`parse_stim(text)`** reads Stim's own format into a `Circuit` with explicit noise ops. It supports:
  - `REPEAT`;
  - `R/RX/M/MX/MR/MRX`;
  - `X/Y/Z_ERROR`, `DEPOLARIZE1/2`;
  - `DETECTOR`/`OBSERVABLE_INCLUDE` with `rec[-k]`;
  - coordinates and `TICK` (ignored).

  Any other instruction is an error, never silently dropped.

Tests (`tests/stim_io.rs`, plus unit tests):
- **Exact round trip.** For the surface code (d = 3, 5) and for 200 random noisy Clifford circuits, `parse(to_stim(c))` compiles to a SymPhase sampler with the **same multiset of variable groups and the same per-outcome detector signatures** as `c` itself. That is an exact equality of sampling distributions, not a statistical test. The parsed program also re-serialises to identical text.
- **Stim's generated circuit.** `stim.Circuit.generated("surface_code:rotated_memory_z", d=3)` parses into 24 detectors. All of them are deterministic: no random "coin" variable reaches a detector, and the noiseless version gives all-zero shots.

### 1.3 Equivalence test (both directions, 10⁶ shots per side)

- **Direction A** (ours → Stim): qsim-lab's native `SurfaceCode::new(d, d)` circuit under `NoiseModel::circuit_level(0.003, 0.003)` is serialised by `to_stim`. qsim-lab samples its in-memory circuit object; Stim samples the file.
- **Direction B** (Stim → ours): Stim's generated `rotated_memory_z` with rounds = d and all four noise knobs at p = 0.003. Stim samples it; qsim-lab parses the same file and samples the result.

**Test.** Per direction and per d, at a family-wise error rate of 1% (Bonferroni over every test in the cell):

- **T0, support (exact):** the set of distinct single-fault signatures (detectors + observable) of our compiled sampler equals the set of error-target sets in Stim's DEM.
- **T1, marginals:** a two-proportion z-test per detector and for the raw observable flip.
- **T2, pairs:** a two-proportion z-test on P(Dᵢ ∧ Dⱼ) for every detector pair that co-occurs in a Stim DEM error. This checks correlations, not just marginals.
- **T3, event counts:** the mean and variance of the number of detection events per shot.

`research/data/qec-r4/stim_equivalence.py` → `stim_equivalence.json`:

| direction | d | detectors | DEM support equal | max abs z (marginals) | max abs z (pairs, #pairs) | z (events/shot mean, var) | observable rate ours / Stim | Bonferroni z* | rejections | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A: ours→Stim | 3 | 16 | yes (55) | 1.67 | 2.16 (39) | +0.35, +0.58 | 0.0409 / 0.0411 | 3.76 | 0/58 | PASS |
| A: ours→Stim | 7 | 192 | yes (883) | 3.43 | 3.89 (819) | -0.04, -0.22 | 0.1875 / 0.1871 | 4.42 | 0/1014 | PASS |
| A: ours→Stim | 11 | 720 | yes (3631) | 3.21 | 3.77 (3487) | -0.27, +1.15 | 0.3433 / 0.3427 | 4.72 | 0/4210 | PASS |
| A: ours→Stim | 15 | 1792 | yes (9451) | 3.95 | 4.40 (9195) | +0.65, -1.41 | 0.4421 / 0.4413 | 4.91 | 0/10990 | PASS |
| B: Stim→ours | 3 | 24 | yes (219) | 1.92 | 3.08 (94) | -0.50, -0.06 | 0.0649 / 0.0646 | 3.94 | 0/121 | PASS |
| B: Stim→ours | 7 | 336 | yes (5471) | 3.07 | 3.72 (2638) | -1.01, +1.24 | 0.2536 / 0.2539 | 4.65 | 0/2977 | PASS |
| B: Stim→ours | 11 | 1320 | yes (24483) | 3.06 | 3.49 (12158) | -2.68, -1.19 | 0.4103 / 0.4093 | 4.95 | 0/13481 | PASS |
| B: Stim→ours | 15 | 3360 | yes (66087) | 3.73 | 4.10 (33262) | -1.56, -0.55 | 0.4786 / 0.4786 | 5.14 | 0/36625 | PASS |

The mean of z² over the marginals is 0.64–1.32 in every cell, as expected under H₀.

**Power check** (`stim_equivalence_negative_control.py`, d = 7, direction B). Our side samples a deliberately wrong circuit; the same test must reject:

| perturbation | max abs z | rejections |
|---|---|---|
| every `DEPOLARIZE2(0.003)` → `0.0033` (+10% on one channel) | 17.1 | 1525 |
| 4 of the 7 `X_ERROR` lines removed | 60.4 | 1720 |

**Verdict: the two samplers produce the same distribution on identical circuits, in both directions.**

### 1.4 Timing on identical circuits

`stim_timing.py`. Both simulators read the **same .stim file**, single-threaded, interleaved, min of 3. Stim's sampler is compiled once (its compile time is reported separately). Columns:

- **Stim write:** `sample_write(shots, os.devnull, "ptb64", append_observables=True)`. This is the same output layout as ours.
- **Stim numpy:** `sample(shots, append_observables=True, bit_packed=True)`. This is what the old comparison timed.
- **Stim native:** Stim 1.16.0 built from source for the host (`-DSIMD_WIDTH=256`, i.e. AVX2, on the VPS; `-O3 -mcpu=native` on the Mac), `stim detect --out_format ptb64`. This includes process start-up and parsing, about 1–5 ms.
- **ours:** `stim_compare bench`, which parses and compiles once, then streams ptb64 bytes to /dev/null.
- **Ratio:** Stim time / our best time (> 1 means we are faster).

**VPS (EPYC, AVX2).** Note that pip's Stim wheel runs its SSE2 build (AVX2 is disabled in the wheel; Stim issue #432).

(VPS table: see §1.5 for the final run.)

**Mac (M1 Pro).** pip's Stim on arm64 runs `_stim_polyfill`, i.e. **64-bit words with no NEON**. A native `-mcpu=native` build is no faster, because Stim has no NEON SIMD backend. The Mac ratios therefore flatter us and should not be generalised.

(Mac table: see §1.5.)

### 1.5 Results

FILLED BELOW

### 1.6 Where our time goes

`stim_compare profile` splits a 100k-shot run into drawing the random variables (`sample_vars`) and the GF(2) evaluation (`eval`). VPS:

| circuit | variables | variable groups | nnz(A) | detector rows | sample_vars | eval |
|---|---|---|---|---|---|---|
| A, d = 15 | 53,873 | 16,073 | 53,116 | 1,793 | 0.194 s (82%) | 0.042 s |
| B (Stim gen.), d = 15 | 71,040 | 26,505 | 121,032 | 3,361 | 0.319 s (72%) | 0.122 s |

**Drawing the noise variables dominates, not the matrix product.** At p = 0.3% Stim's d = 15 circuit has about 80 faults per shot. Each fault costs about 45 ns: a ChaCha12 `f64`, a `ln` for the geometric skip, `random_range` for the Pauli, and a scattered OR. On top of that, the dense path zeroes all `num_vars` words (570 KB) every 64 shots.

Stim pays per gate per 128- or 256-shot word but only a few ns per fault. On its compact, layered circuit B the two costs come out even. On our sequential circuit A it pays per-instruction overhead for about 2,000 one-gate lines, and we win 1.6–2×.

**Fix tried here.** `SymPhaseSampler::sample_vars_sparse` + `eval_sparse`:
- clear only the touched words;
- evaluate column-wise over the faults that fired.

It is **bit-identical** to the dense path for the same RNG stream (`tests/symphase.rs::sparse_sampling_path_is_identical_to_dense`). Together with `SmallRng` (Xoshiro256++) it gains 5–20% on large circuits. It does not change the picture.

**The next lever, not done here:** cheaper fault draws. Options are batching geometric skips with a precomputed `1/ln(1-p)`, drawing the Pauli index from spare bits of the same random word, or a wider batch (256 shots) so that coins and per-group overhead amortise.

Our compile step is also slower than Stim's: 35–45 ms against about 1 ms at d = 15. It is amortised after about 10⁴ shots.

---

## Part 2. Colour-code syndrome-extraction schedules

### 2.1 Literature: where is the open question?

I looked at three candidates:

- **Bivariate-bicycle codes such as [[72,12,6]].** Crowded. There is IBM's depth-7 schedule, the morphing circuits (arXiv:2407.16336), reinforcement-learning schedule synthesis (AlphaSyndrome, arXiv:2609.12020) and PropHunt (MaxSAT-guided circuit edits). Our decoder story is also weakest there.
- **Rotated surface code.** Settled. Our own `research/schedules.md` (exhaustive over 576 orders), and recent "off-the-hook" and "no more hooks" papers (arXiv:2602.09099, 2603.01628).
- **4.8.8 / 6.6.6 colour code.** Mostly settled at the uniform-schedule level:
  - Beverland et al. and Lee et al. ("tri-optimal") optimised *spatially uniform* single-auxiliary circuits. These are distance-halving.
  - Gidney–Jones (superdense, middle-out).
  - **Kishony & Fowler, "Color code off-the-hook" (arXiv:2603.28852, 2026)** gave a *colour-dependent* single-auxiliary schedule that keeps the full circuit distance in the bulk at minimal depth (6 CNOT layers per Pauli type, collision-free).

  Kishony–Fowler leave one thing open at the boundary. Boundary trapezoids reuse the bulk schedule. The resulting "fractional hook errors" give d_circ = d − ⌊(d+3)/6⌋, which they verified to d = 13. Their words: *"multiple valid schedules remain, of which we select one arbitrarily. It may be interesting to explore the performance of the alternatives in future work"*. They suggest flag qubits at the boundary as the fix. Their companion code (`Classiq/classiq-library`, `syndrome_extraction_optimization/`) searched only **colour-uniform** schedules: 864 zero-collision ones, all with d_circ = 6 at d = 7.

**The open question taken here.** Within *exactly their design space*, can non-uniform per-plaquette schedules (in particular at and near the boundary) do better than their arbitrary pick? Their design space is: one auxiliary per plaquette, the same 6-step schedule for the X and Z halves, every plaquette collision-free. The objective is lexicographic: (1) circuit distance; (2) the number of minimum-weight logical fault sets, which sets the leading-order logical error rate p_L ≈ N_min·p^d_circ at low p; (3) measured logical error per round at p = 0.1–0.3%.

### 2.2 Tools built for this (all new on this branch)

- **`src/qec/color.rs`: triangular 6.6.6 colour-code memory circuits.**
  - Kishony–Fowler's exact layout (colour-code-stim coordinates and offsets) and their round structure: CX data→anc at steps 1–6, `M`, `RX`, CX anc→data at steps 1–6, `MX`, `R`.
  - One auxiliary per plaquette and any per-plaquette schedule.
  - Two noise models, written as explicit ops so the sampled circuit is exactly the exported one: noisy-CNOT (`DEPOLARIZE2(p)` after every CNOT) and uniform depolarizing (adds idle `DEPOLARIZE1(p)`, readout flips, reset errors).
  - `KF_SCHEDULE` is their published pick (`zero_collision_schedules.csv`, row 1, the one `benchmark_circuits.py` loads).

  Validation, `tests/color_code.rs`:
  - n = (3d²+1)/4 data qubits and (n−1)/2 plaquettes for d = 3–11.
  - All X/Z stabilizer pairs overlap evenly, and the logical (the y = 0 row, weight d) commutes with every stabilizer.
  - Code distance is d by brute force (d = 3, 5).
  - KF is collision-free for d = 3–13.
  - **Every detector is deterministic** for KF, tri-optimal and random per-plaquette schedules (d = 3, 5, 7; 1–3 rounds; both noise models): no random variable reaches a detector, and noiseless shots are all zero.
- **`circuit_dem`**: the circuit-derived DEM read off the SymPhase sampler (each noise outcome is one fault). Cross-checked against Stim's DEM of the exported circuit: same mechanism counts (288 at d = 3, 2221 at d = 5), same ILP distance.
- **`src/qec/distance.rs`: exact minimum-weight logical by branch and bound**, with an exact count of minimum-weight logicals.
  - The root branches on the lowest-index observable-flipping mechanism. Each node branches on the mechanisms touching its most constrained fired detector, and excludes a mechanism after its branch. The branches therefore *partition* the solution set, so every minimum-weight logical is found exactly once.
  - Pruning: ⌈|F|/maxdeg⌉ plus a greedy independent-set bound over fired detectors.
  - Verified against brute force on 300 random DEMs, weights *and* counts.
  - It runs on the Z sector (Z detectors + observable). That is a lower bound on the full-DEM distance, and it is *certified* exact when every mechanism of the found logical has a twin with an empty X-sector signature. Certification held in every case reported here.
  - Speed: KF d = 9 rounds = 1 takes 2 s (pysat RC2 MaxSAT: 381 s); d = 7 rounds = 7 takes 5 s (RC2: 367 s; HiGHS ILP: > 10 min).
- **`research/data/qec-r4/color_exhaustive.py`, `color_lns.py`: schedule search.**
  - For the Z-memory noisy-CNOT (and uniform) DEM, a plaquette's schedule matters only through (a) the order of its own CNOTs, which sets the X-half hooks, and (b) for each of its data qubits, the order in which that qubit meets its 2–3 plaquettes in the Z half (a data X error at step t reaches exactly the ancillas that touch the qubit later).
  - Options with equal (a, b) are DEM-equivalent, so the search enumerates one representative per class. This cuts a weight-4 corner plaquette from 165 options to 33.
  - The LNS repeatedly re-optimises, *exhaustively over all classes*, each single plaquette and each adjacent pair of plaquettes that the current minimum-weight logicals touch. It moves to the best improvement and stops at a schedule no such move improves.
- **Decoders for LER.**
  - Our BP+OSD-CS (`src/qec/bposd.rs`): min-sum, 50 iterations, combination sweep of order λ, decoding the Z sector.
  - Tesseract (K–F's decoder and settings) fed with **our** samples and **our** DEM, used as a cross-check.
  - On the KF d = 3 circuit at p = 0.5%, Tesseract on our samples gives p_L = 1.98(+0.11/−0.11)% per shot against K–F's published 1.91% (378k shots): **our circuit reproduces theirs**.
  - Our BP+OSD is 2.5–3× weaker than Tesseract on this code (d = 5, p = 0.3%: 5.4×10⁻³ at λ = 100 against 1.9×10⁻³). LER comparisons between schedules therefore use the *same* decoder on both arms and are decoder-relative.

### 2.3 Reproducing Kishony–Fowler

PART2_RESULTS
