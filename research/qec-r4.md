# QEC round 4: identical-circuit Stim comparison, and colour-code schedule search

Branch `exp/qec-r4`. Raw data and scripts: `research/data/qec-r4/`.

Machines:
- **VPS**: 4 vCPU AMD EPYC-Rome (AVX2), shared with other agents. Timings are only quoted when the 1-minute load was ≤ 4, and the load is recorded per row.
- **Mac**: Apple M1 Pro, 8 cores, NEON. Every timing run held `/tmp/qsim-mac-bench.lock` and waited for 1-minute load < 3; the starting load is recorded.

Everything is single-threaded unless stated. Stim is 1.16.0 throughout.

**Headlines.**
- **Part 1: SymPhase is at parity with Stim, not 4–6.5× ahead.**
  - The equivalence holds both ways: identical circuits in both directions, 10⁶ shots, 0 of 69,476 tests rejected.
  - Single-thread, same output format, x86:
    - on Stim's own circuit, qsim-lab / Stim = 0.89–1.01× against AVX2-native Stim and 1.03–1.19× against pip Stim;
    - on qsim-lab's sequential circuit, 1.6–1.8×.
  - M1: 3.0–5.6×, but Stim has no NEON backend there.
  - The old claim came from timing Stim's numpy output path.
- **Part 2: in Kishony–Fowler's single-auxiliary colour-code design space, per-plaquette boundary schedules found by exact search keep their circuit distance but halve the minimum-weight logicals.**
  - The minimum-weight logical count drops 388 → 197 (d = 5), 12,901 → 6,627 (d = 7) and 492 → 255 (d = 9), all over d rounds.
  - Logical error per round falls by **22–31% at d = 5 and 7, p = 0.2–0.3%, under noisy-CNOT noise**: d = 5, 0.69–0.70× with BP+OSD (CI ±4%) and 0.69× [0.56, 0.84] / 0.78× [0.64, 0.96] with Tesseract; d = 7, 0.74–0.76× [0.66, 0.84].
  - **At d = 9 there is no measurable gain** (1.02× [0.94, 1.11] at p = 0.3%) despite the halved N_min. The boundary-local benefit fades with distance at these error rates.
  - X-memory improves too, although only Z-memory was optimised.
  - No change tried raised the circuit distance above K–F's d − ⌊(d+3)/6⌋, which we reproduce exactly up to d = 9 over 9 rounds (and at d = 11 over 1 round).

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

**Mac (M1 Pro).** pip's Stim on arm64 runs `_stim_polyfill`, i.e. **64-bit words with no NEON**. A native `-mcpu=native` build is no faster, because Stim has no NEON SIMD backend. The Mac ratios therefore flatter us and should not be generalised.

### 1.5 Results

**VPS (EPYC-Rome, AVX2), 1-min load 3.5–4.2; ours = dense path, StdRng**

| circuit | d | shots | Stim sample_write ptb64 (Mshot/s) | Stim sample() numpy (Mshot/s) | Stim native CLI ptb64 (Mshot/s) | ours dense (Mshot/s) | ours/Stim (write) | ours/Stim (numpy) | ours/Stim native | load |
|---|---|---|---|---|---|---|---|---|---|---|
| A (ours) | 3 | 2000000 | 30.57 | 10.48 | 26.59 | 48.79 | 1.60× | 4.65× | 1.84× | 3.5 |
| B (Stim gen.) | 3 | 2000000 | 22.26 | 8.00 | 26.38 | 26.54 | 1.19× | 3.32× | 1.01× | 3.5 |
| A (ours) | 7 | 499968 | 2.46 | 1.12 | 2.75 | 4.40 | 1.78× | 3.91× | 1.60× | 3.6 |
| B (Stim gen.) | 7 | 499968 | 2.28 | 0.73 | 2.55 | 2.50 | 1.09× | 3.41× | 0.98× | 3.8 |
| A (ours) | 11 | 200000 | 0.59 | 0.29 | 0.70 | 1.17 | 1.96× | 4.01× | 1.66× | 3.9 |
| B (Stim gen.) | 11 | 200000 | 0.58 | 0.18 | 0.69 | 0.62 | 1.06× | 3.47× | 0.89× | 4.0 |
| A (ours) | 15 | 99968 | 0.24 | 0.12 | 0.27 | 0.44 | 1.83× | 3.66× | 1.66× | 4.0 |
| B (Stim gen.) | 15 | 99968 | 0.23 | 0.05 | 0.26 | 0.24 | 1.03× | 4.51× | 0.89× | 4.2 |
**Mac M1 Pro, load 2.8–3.1, under the bench lock; ratios use our best variant**

| circuit | d | shots | Stim sample_write ptb64 (Mshot/s) | Stim sample() numpy (Mshot/s) | Stim native CLI ptb64 (Mshot/s) | ours dense (Mshot/s) | ours sparse+SmallRng (Mshot/s) | ours/Stim (write) | ours/Stim (numpy) | ours/Stim native | load |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (ours) | 3 | 2000000 | 17.25 | 15.15 | 16.77 | 59.71 | 77.32 | 4.48× | 5.10× | 4.61× | 2.9 |
| B (Stim gen.) | 3 | 2000000 | 14.29 | 11.61 | 13.88 | 32.29 | 42.87 | 3.00× | 3.69× | 3.09× | 2.9 |
| A (ours) | 7 | 499968 | 1.27 | 1.23 | 1.28 | 5.01 | 7.00 | 5.53× | 5.71× | 5.48× | 2.9 |
| B (Stim gen.) | 7 | 499968 | 1.15 | 0.93 | 1.16 | 2.82 | 4.32 | 3.75× | 4.65× | 3.73× | 2.8 |
| A (ours) | 11 | 200000 | 0.32 | 0.31 | 0.32 | 1.27 | 1.75 | 5.51× | 5.59× | 5.45× | 3.1 |
| B (Stim gen.) | 11 | 200000 | 0.30 | 0.24 | 0.30 | 0.75 | 1.16 | 3.89× | 4.86× | 3.87× | 3.0 |
| A (ours) | 15 | 99968 | 0.12 | 0.12 | 0.12 | 0.50 | 0.69 | 5.62× | 5.67× | 5.62× | 3.1 |
| B (Stim gen.) | 15 | 99968 | 0.12 | 0.09 | 0.12 | 0.30 | 0.47 | 4.07× | 5.37× | 4.02× | 3.1 |
The pip "write" path and the native build agree on the Mac (`stim_timing_mac_m1_v2_nativecli.jsonl`), as expected with no SIMD backend there.

**Honest ratios.**
- **x86, Stim's own circuit (B): 0.89–1.01× against AVX2-native Stim, 1.03–1.19× against pip Stim.** This is parity; at d ≥ 11 Stim is about 11% faster.
- **x86, qsim-lab's sequential circuit (A): 1.6–1.8× faster than either Stim.**
- **M1: 3.0–5.6× faster**, an artefact of Stim having no NEON path.
- Against Stim's numpy path, which is what the old claim used, we are 3.3–4.7× faster on x86. That is where "4–6.5×" came from.

A later VPS run at load 15, with the sparse path and SmallRng, is indicative only: sparse + SmallRng is 0.97× / 0.75× of the dense time on A / B at d = 15.

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

It is **bit-identical** to the dense path for the same RNG stream (`tests/symphase.rs::sparse_sampling_path_is_identical_to_dense`). Together with `SmallRng` (Xoshiro256++) it cuts sampling time by 4–25% on x86 (indicative, load 15) and 23–36% on the M1. It does not change the x86 picture.

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

Exact Z-memory circuit distance under the noisy-CNOT model. N_min is the number of distinct minimum-weight logical fault sets in the merged Z-sector DEM. Every value is certified, i.e. it lifts to the full DEM. Times are single-core on the VPS.

| schedule | d | rounds | d_circ (ours) | K–F formula d − ⌊(d+3)/6⌋ | N_min | solver time | independent cross-check |
|---|---|---|---|---|---|---|---|
| KF | 3 | 3 | 2 | 2 | 9 | < 1 ms | full-DEM ILP (HiGHS) on our DEM and on Stim's DEM of the exported circuit: 2 |
| KF | 5 | 1 | 4 | 4 | 55 | 1 ms | Z-sector MaxSAT (RC2): 4 |
| KF | 5 | 5 | 4 | 4 | 388 | 10 ms | full-DEM ILP on our DEM and on Stim's DEM: 4; full-DEM MaxSAT: 4 |
| KF | 7 | 1 | 6 | 6 | 883 | 0.1 s | RC2: 6 |
| KF | 7 | 7 | 6 | 6 | 12,901 | 4.8 s | RC2 (Z sector): 6 in 367 s |
| KF | 9 | 1 | 7 | 7 | 36 | 2 s | RC2: 7 in 381 s |
| KF | 9 | 9 | 7 | 7 | 492 | 156 s | — |
| KF | 11 | 1 | 9 | 9 | 612 | 257 s | — |
| tri-optimal (uniform) | 3 / 5 / 7 / 9 | 1 | 2 / 3 / 4 / 5 | — | 2 / 4 / 7 / 12 | ≤ 0.03 s | distance-halving, as K–F state |

- **The formula is reproduced exactly, including at the full d = 9, 9-round circuit.** That is independent confirmation by a different generator, DEM builder and solver.
- Under the uniform depolarizing model (idle, readout and reset noise added), d_circ and N_min are identical. The extra fault locations only duplicate existing Z-sector signatures.
- The count is *per distinct merged mechanism set*, not weighted by how many physical faults produce each mechanism.
- Rounds = 1 is a valid upper bound for screening: any logical in the 1-round circuit maps onto the last round of a longer one. It equalled the full-rounds value in every case checked (d = 5, 7, 9).

**Where K–F's minimum-weight logicals live (d = 9).** All 36 weight-7 logicals (1 round) run from the observable boundary (y = 0, the red boundary) to the opposite corner. Every one uses two mechanisms on the apex plaquettes 27/28/29. Most also use a weight-3 middle hook or a boundary weight-2 hook near the bottom. So the deficit is corner-anchored, not only boundary-hugging (`DUMP_ALL=1 color_search distance 9 1 kf`).

### 2.4 Schedule search: results

**Search space.**
- One auxiliary per plaquette.
- The same 6-step schedule for the Z half (CX data→anc) and the X half (CX anc→data).
- Every plaquette's steps are a permutation of 1..6 restricted to its present positions.
- The whole circuit is collision-free: no data qubit is in two CNOTs at the same step.

So depth and qubit count are identical to K–F's. "Free" plaquettes are those touching a data qubit that lies on the boundary; at d = 5 that is all 9. Plaquettes not free keep K–F's colour schedule. Objective: (d_circ, −N_min), lexicographic.

**Method.** LNS from K–F (§2.2): exhaustive over every DEM-equivalence class of every single involved plaquette, then every adjacent involved pair. It moves to the best class of the first neighbourhood that improves, and repeats until no neighbourhood improves (a local optimum). Distances and counts are exact (§2.2).

| d | rounds in objective | K–F (d_circ, N_min) | LNS result (d_circ, N_min) | status | moves | file |
|---|---|---|---|---|---|---|
| 5 | 5 | (4, 388) | **(4, 197)** | local optimum: no single (9) or pair (15) re-schedule improves | 20 | `schedules/d5_lns_r5.sched`, `lns_d5_r5.jsonl` |
| 5 | 1 | (4, 55) | (4, 36) | local optimum (9 singles, 10 pairs) | 7 | `lns_d5_r1.jsonl` |
| 7 | 1 | (6, 883) | (6, 505) at the snapshot used for the LER runs; **(6, 434)** when stopped | singles exhausted; still improving through pair moves when stopped | 34 | `schedules/d7_lns_r1_snapshot.sched`, `schedules/d7_lns_r1_final.sched`, `lns_d7_r1.jsonl` |
| 7 | 7 (check) | Z: (6, 12,901); X: (6, 13,120) | snapshot Z: (6, 7,509); **final Z: (6, 6,627), final X: (6, 6,856)** | exact, certified | | `d7_snapshot_r7_distance.txt` |
| 9 | 1 | (7, 36) | **(7, 18)** after singles, then (7, 15) after a pair move | singles exhausted at 18; 4 of about 60 pairs done when stopped (about 17 min per pair on the Mac) | 4 | `schedules/d9_lns_r1_singles.sched` (N_min 18), `schedules/d9_lns_r1_15.sched` (N_min 15), `lns_d9_r1_part1/2.jsonl` |
| 9 | 9 (check) | (7, 492) | **(7, 255)** for the count-18 schedule | exact, certified (408 s) | | `d9_lns_r9_distance.txt` |

Earlier random local search (`color_schedule_search.py`, 6 seeds × about 60 evaluations at d = 9) and exhaustive enumeration of the apex pair {28, 29} at d = 9 (140 of 1003 classes before it was stopped) found nothing above d_circ = 7. Exhaustive per-colour (uniform) schedules are K–F's own search (864 zero-collision, all d_circ = 6 at d = 7).

**Circuit distance.**
- **No schedule in this space beat K–F's d_circ.** At d = 5, where every plaquette is free, no single or pair move reaches 5. This is consistent with K–F's observation that every hook on a corner plaquette is malign, so d_circ ≤ d − 1 for any single-auxiliary schedule.
- At d = 9, a gain over K–F would mean d_circ = 8, i.e. eliminating every weight-7 logical. The LNS reduced their number but has not removed them; see the table.
- This is a negative result for the distance question within single and pair moves. It is not a proof over the whole space.

**Entropy.** Non-uniform boundary schedules roughly **halve the number of minimum-weight logicals at the same d_circ and depth** (d = 5: 388 → 197 over 5 rounds; d = 7 and d = 9 in the table). K–F's arbitrary pick among colour-uniform schedules is far from optimal on this second-order metric.

### 2.5 Does it matter? Logical error rate per round, K–F vs LNS (d = 5, rounds = 5)

Independent samples, same decoder on both arms. 1–4×10⁶ shots per arm for BP+OSD (order 100, Z sector); 128k shots per arm for Tesseract (K–F's decoder and settings) on our samples and DEM. Per-round p_L = (1 − (1 − 2P)^{1/r})/2. Ratio CIs use the log-ratio normal approximation (`ler_compare.py`; `ler_d5_bposd.jsonl`, `tess_d5.jsonl`).

| noise | p | decoder | K–F p_L/round | LNS p_L/round | ratio LNS/K–F [95% CI] |
|---|---|---|---|---|---|
| noisy CNOT | 0.3% | BP+OSD | 1.125×10⁻³ (5600 fails) | 7.83×10⁻⁴ (3903) | **0.697 [0.669, 0.726]** |
| noisy CNOT | 0.3% | Tesseract | 3.79×10⁻⁴ (242) | 2.60×10⁻⁴ (166) | **0.686 [0.563, 0.836]** |
| noisy CNOT | 0.2% | BP+OSD | 4.62×10⁻⁴ (4606) | 3.17×10⁻⁴ (3162) | **0.686 [0.656, 0.718]** |
| noisy CNOT | 0.2% | Tesseract | 1.33×10⁻⁴ (212) | 1.04×10⁻⁴ (166) | **0.783 [0.639, 0.959]** |
| noisy CNOT | 0.1% | BP+OSD | 1.09×10⁻⁴ (2184) | 8.27×10⁻⁵ (1653) | **0.757 [0.710, 0.807]** |
| uniform depolarizing | 0.3% | BP+OSD | 1.89×10⁻² (87661) | 1.79×10⁻² (83221) | 0.949 [0.940, 0.958] (raw JSON lost in a rebase; numbers from `ler_d5_bposd.log`) |
| uniform depolarizing | 0.1% | BP+OSD | 1.358×10⁻³ (27017) | 1.189×10⁻³ (23662) | **0.876 [0.861, 0.891]** |

**The same comparison at other settings.**

| memory | d | noise | p | decoder | K–F p_L/round | LNS p_L/round | ratio [95% CI] |
|---|---|---|---|---|---|---|---|
| **X basis** (the schedule was optimised for Z only) | 5 | noisy CNOT | 0.3% | BP+OSD | 1.118×10⁻³ (5566) | 8.08×10⁻⁴ (4029) | **0.724 [0.695, 0.754]** |
| X basis | 5 | noisy CNOT | 0.1% | BP+OSD | 1.024×10⁻⁴ (512) | 8.20×10⁻⁵ (410) | **0.801 [0.703, 0.912]** |
| Z basis | 7 (snapshot) | noisy CNOT | 0.3% | BP+OSD | 3.527×10⁻⁴ (2464) | 2.694×10⁻⁴ (1883) | **0.764 [0.720, 0.811]** |
| Z basis | 7 (snapshot) | noisy CNOT | 0.2% | BP+OSD | 9.11×10⁻⁵ (637) | 6.75×10⁻⁵ (472) | **0.741 [0.658, 0.835]** |
| Z basis | 9 (N_min-18 schedule; 255 over 9 rounds) | noisy CNOT | 0.3% | BP+OSD | 1.882×10⁻⁴ (1015) | 1.923×10⁻⁴ (1037) | **1.022 [0.937, 1.114]**: no gain |

X-memory exact distances for the d = 5 schedule: K–F (4, 399) → LNS (4, 201). The gain transfers to the basis that was not optimised, as self-duality suggests.

- Under noisy-CNOT noise, K–F's own headline model where hooks matter most, the re-scheduled circuit has **about 30% lower logical error per round at p = 0.1–0.3%**. Two different decoders agree: BP+OSD 0.70 [0.67, 0.73] and Tesseract 0.69 [0.56, 0.84] at p = 0.3%.
- Under uniform depolarizing noise at p = 0.3% this d = 5 BP+OSD point is far from the low-p regime: about 9% failure per shot, roughly at threshold for this decoder. The gain shrinks to 5%, as expected once higher-weight failures dominate.

### 2.6 What is known vs what is new

**Known.**
- Uniform single-auxiliary colour-code circuits halve the circuit distance (Beverland et al.; Lee et al.). Reproduced here: tri-optimal gives d_circ = (d+1)/2.
- K–F's colour-dependent schedule keeps the bulk distance. Their boundary loses ⌊(d+3)/6⌋, verified to d = 13 by ILP. Corner hooks are always malign.
- Their search covered the 864 colour-uniform zero-collision schedules.

**New here.**
1. **Independent reproduction** of K–F's d_circ at d = 3, 5, 7, 9 (all rounds) and d = 11 (1 round). It uses a different generator (`src/qec/color.rs`), DEM builder (SymPhase) and solver (an exact branch and bound that is about 100–200× faster than MaxSAT/ILP on these instances).
2. **Exact minimum-weight-logical counts N_min**, which K–F do not report: 388 (d = 5), 12,901 (d = 7) and 492 (d = 9) over d rounds. The d = 9 minimum-weight logicals are corner-anchored strings from the observable boundary to the opposite corner.
3. **Per-plaquette (non-uniform) boundary scheduling inside K–F's exact design space** (same qubits, same 6+6 CNOT layers, collision-free):
   - it **does not raise d_circ** under any single- or pair-plaquette change we tried (d = 5–9);
   - it **roughly halves N_min**: 388 → 197 (d = 5), 12,901 → 6,627 (d = 7), 492 → 255 (d = 9), all over d rounds;
   - it lowers the logical error per round under noisy-CNOT noise by 22–31% at p = 0.2–0.3% (d = 5 and 7; two decoders agree at d = 5) and by 24% at p = 0.1% (d = 5);
   - **at d = 9 it shows no LER gain** at p = 0.3% (1.02× [0.94, 1.11], BP+OSD) despite N_min 492 → 255. At this p, failures at d = 9 are dominated by paths above minimum weight and by the bulk, where the schedule is K–F's. This matches K–F's expectation that boundary effects fade with distance, and limits the practical value to small codes (d ≤ 7) or very low p, where N_min dominates;
   - it lowers it by 12% under uniform depolarizing noise at p = 0.1%;
   - the gain carries over to X-memory (N_min 399 → 201 at d = 5 and 13,120 → 6,856 at d = 7; LER 0.72× at d = 5, p = 0.3%) although only Z-memory was optimised.

   K–F flagged "the alternatives" as future work; this measures them. The improvement is free (no extra qubits or depth), measurable at d = 5–7, and not detectable at d = 9, p = 0.3%.

**Caveats.**
- The search is local (single and pair moves). "No distance gain" is not a proof over the whole space.
- N_min counts merged-DEM mechanism sets, not weighted by fault multiplicity.
- Only Z-memory is optimised. X-memory was checked at d = 5 (distance and LER) and d = 7 (distance), where it also improves. Individualised boundary plaquettes break the 3-fold rotation symmetry, so check each basis before use.
- Our BP+OSD is 2.5–3× weaker than Tesseract in absolute terms. The schedule ratios agree between the two decoders where both were run (d = 5, p = 0.2% and 0.3%).
- The d = 7 and d = 9 schedules are snapshots of searches that were still improving when stopped (d = 9 reached N_min = 15 over 1 round after a pair move).
- SI1000 noise was not run.

### 2.7 Reproduce

```bash
cargo build --release --example stim_compare --example color_search --example color_ler
# Part 1
python research/data/qec-r4/stim_equivalence.py 1000000 3,7,11,15          # needs stim
python research/data/qec-r4/stim_equivalence_negative_control.py
STIM_CLI=/path/to/native/stim python research/data/qec-r4/stim_timing.py target/release/examples/stim_compare /tmp/w 15 100000 3
# Part 2: exact distance and min-weight count (K-F schedule, or a schedule file)
target/release/examples/color_search distance 9 9 kf
target/release/examples/color_search distance 5 5 research/data/qec-r4/schedules/d5_lns_r5.sched 10000000 18446744073709551615 cnot x   # X basis
# search
python research/data/qec-r4/color_lns.py target/release/examples/color_search 5 5 2 out.jsonl --pairs 20
# logical error rate, same decoder on both arms
python research/data/qec-r4/ler_compare.py target/release/examples/color_ler 5 5 cnot 0.003 1000000 2 100 out.jsonl kf=kf lns=research/data/qec-r4/schedules/d5_lns_r5.sched
python research/data/qec-r4/tesseract_ler.py target/release/examples/color_search target/release/examples/stim_compare 5 5 cnot 0.003 kf 128000 77 2   # needs tesseract-decoder
```

Tests: `tests/stim_io.rs` (exact round trips), `tests/color_code.rs` (generator validation, K–F distance formula and counts, the d = 5 LNS schedule, X basis), `tests/symphase.rs::sparse_sampling_path_is_identical_to_dense`, `src/qec/distance.rs` (brute-force check of weights and counts), `src/qec/bposd.rs`.
