# qsim-lab results (5 October 2026)

What this repository has shown so far, with the caveats that go with each number. Every engine is
**exact** (no truncation) and differential-tested against an independent reference state vector
(1e-12 in f64, 1e-5 in f32) or against exact outcome distributions. Each merged branch was audited by a
separate agent that tried to break its headline claim; the audits, including the claims that were
corrected or withdrawn, are in `research/process/audit.md` (§16 is the latest), `research/shor/shor-r4-audit.md` and
`research/qec/qec-r4.md` Part 1. Raw data and methods are in `research/`.

> **Not yet audited (5 Oct 2026).** The branches merged on 5 October 2026 (`exp/shor-xl`, `exp/stabrank5`,
> `exp/qldpc-x`, `exp/approx-modexp`, `exp/todd`, `exp/tn`, `exp/sampler-x`, `exp/avx512-sv`) were stopped
> before their independent audits, and some were stopped mid-experiment (their last commit is a snapshot
> of in-progress work). Every number they added here and in their notebooks is **provisional** until an
> audit in `research/process/audit.md` confirms it. Treat their "new" / "record" claims with that caveat.

Machines:
- **VPS**: 4 vCPU AMD EPYC-Rome (AVX2), 7.7 GB, shared with other agents. Timings are only quoted when
  the 1-minute load was ≤ 4.
- **Xeon VM** (exp/shor-xl): Intel Xeon Gold 6548Y+ (8 cores × 2 hyper-threads, AVX-512), 31 GB,
  shared; the 1-minute load from other users was 15–55 during that work, so its timings are marked loaded.
- **Mac**: Apple M1 Pro (8 cores, NEON), 16 GB, the owner's laptop. Timing runs hold a swarm-wide lock
  so only one benchmark runs at a time; the load is recorded and was often 3–10 from other agents'
  builds.

"Record" below always means *this repo's* record. None of this is a classical factoring or decoding
speed-up; the Shor runs are exact simulations whose cost tracks the classical difficulty of the
instance.

## 1. Shor's algorithm, gate by gate

**The circuit.** Semiclassical order finding with one recycled control qubit; each round's
controlled-`U_{a^(2^k)}` is Gidney's windowed modular multiplier (w = 4) written out as X / CNOT /
Toffoli gates, 4n + 8 qubits. Because that block is a permutation, the state is a list of basis-state
branches; a bit-sliced evaluator runs every gate on 64·L branches at once. The audit re-checked the
oracle with a from-scratch gate interpreter (all 2^nq inputs at N = 15; random valid inputs at the
record's size) and the outcome distributions against the reference simulators.

**Generic N with a random base (Mac, `research/shor/shor.md` round 4; this repo's record until
exp/shor-xl, below):** N = 1,537,596,787 (31 bits, generic: the first balanced semiprime of a seeded
generator, random base), **132 qubits, 1.70 M gates per run,
r = 256,252,500, factored in 134 s (f32), 4.28 GB**. Cost is linear in the order r, so it is exponential
in the bit length for generic N: 24–29-bit N take 0.2–44 s depending on r. Structured N = p(2p − 1)
(classically trivial, λ ≈ √(2N)) go to 52 bits (216 qubits, 7.4 M gates, 60 s), which only shows that
r, not N, sets the cost. The old N ≈ 10⁶ record circuit (1.15 M gates) now takes 0.05 s instead of 10.8 s.

**Larger N whose support is small (exp/shor-xl, Xeon VM, `research/shor/shor-xl.md`).** Ekerå–Håstad
(EH) with an odd-order base `g = h^(2^n)` (h from a seeded rule; no factors used in the run) keeps
the support at `ord(g)`, a divisor of the odd part of λ(N), so the cost depends on λ(N), not on N:
- **N_W = 549,755,813,701 = 712,321 × 771,781** (39 bits), the largest N of Willsch et al. 2023
  (arXiv:2308.05047; a 40-qubit state vector on up to 2048 GPUs of the JUWELS Booster, modular
  multiplication applied as a permutation, cost independent of the order): factored by exact
  simulation of every gate of a 165-qubit EH circuit (60 exponent bits, 1.48 M operations, 352 k
  Toffolis) on every branch, one run, **99.6 s on 16 threads at a 1-min load of 21–27 (loaded),
  2.26 GB** — cheap here only because `ord(g) = λ_odd(N_W) = 71,582,595` is small
  (`p − 1 = 2^7·3·5·7·53`, which also makes N_W fall to Pollard's p − 1 at once). A 166-qubit
  `w_e = 2` circuit measured the same pair; Shor's order finding on the same g plus Miller's
  reduction also factors it. Plain Shor with the random base itself would need ≈ 147 GB.
- **43-bit N = 4,911,456,443,897 = 1,456,057 × 3,373,121** (the seeded generator's first balanced
  43-bit semiprime): 181 qubits, 66 exponent bits, 1.89 M operations (470 k Toffolis), one run,
  **190 s at load ≈ 30, 3.62 GB** — the largest N not constructed to be easy (the 52-bit runs above
  use N = p(2p − 1)) that we know to have been factored by gate-level simulation of a Shor-type
  circuit, **with its cost set by the support `ord(g) = 115,574,445 = λ_odd/83`** of the seed-1 base
  (a random base is this lucky with probability ≈ 5 %; full odd order would need ≈ 307 GB). The instance was chosen, from supports computed with the factors beforehand, as the only
  generator N of 34–63 bits whose seed-1 run fits in RAM; it is classically trivial (`p − 1`, `q − 1`
  are 243-smooth).
- Every generator N of 22–33 bits factors with its seed-1 odd-order base (≤ 3.6 GB, ≤ 75 s); the
  34-bit one would need 53 GB, so the contiguous generic frontier on this 31 GB machine is 33 bits.
- Engine: an AVX-512 (`VPTERNLOGQ`) tier of the bit-sliced evaluator (kernel 1.5× AVX2 per thread;
  gate evaluation 1.18–1.32× inside whole runs; 4–9 % end to end, where sorting dominates) and an
  in-place window finish (peak RSS 16 B per branch in f32, was up to 33 B). The 31-bit record command
  takes 76.4 s on this VM (load 23–29).
- All timings above are from a shared machine whose load exceeded its 16 vCPUs throughout; checks:
  an independent interpreter verified 11 full-size N_W window blocks on ≈ 2,000 random inputs each,
  and the small-N distributions equal the textbook EH / Shor distributions to 1e-12.

**Under circuit noise (`research/shor/shor-noise.md`).** Exact Pauli-noise trajectories of the same circuit,
10–24-bit N (up to 104 qubits, 0.82 M gates), depolarizing faults after every gate on every qubit it
touches (L ≈ 2.3 × gates locations; no idle noise, ideal classically controlled phases):
- each fault is fatal with probability **d = 0.716 ± 0.004** (0.67–0.77 per instance), so
  P_succ ≈ S₀·exp(−d·p·L) and the success probability halves at **p½ = 5.5·10⁻⁷** per location at
  n = 24 (about one expected fault per run at every size);
- phase-flip noise is 2.4× less damaging than bit-flip (Z faults are harmless in the spare low-bit
  rounds); an ideal ancilla reset between rounds lowers d to 0.47;
- the exponential form is a good description up to pL ≈ 2 but not exact: with three faults the
  success rate is about 2× the independent-fault prediction (a few-percent floor). The effect on p½
  is < 1 %.
- Audit (§16): an independent re-implementation at n = 10 (own bit-sliced evaluator, location model,
  control algebra and success test) agrees with the engine: phase-flip S₁ = 0.629 ± 0.006 vs
  0.631 ± 0.005; depolarizing per-Pauli outcomes within 1–2σ. The new `unsafe` in the noisy engine is
  sound (every index is asserted ≤ nq + 1 when the program is built, and the slice buffer has nq + 2
  words).

## 2. Surface-code sampling vs Stim 1.16 (corrected 4 Oct 2026; FastSampler 4 Oct 2026)

**Correction (still stands).** The earlier claim "4.0–6.5× faster than Stim" was withdrawn: it timed
Stim's slow numpy output path against qsim-lab without output, on a hand-written copy of the circuit.
With identical circuits and identical output, the old SymPhase sampler was at parity with AVX2 Stim on
Stim's own circuit (0.89–1.01×, `research/qec/qec-r4.md` §1.5).

**New sampler (`research/qec/fast-sampler.md`).** `FastSampler` keeps SymPhase's compiled fault → detector
map but draws faults as Poisson "hits":
- one uniform random word picks the location and the Pauli of a fault;
- a precomputed (location, Pauli) → detector table is XOR-ed branch-free into the 64-shot output words;
- hits are generated block by block so table accesses walk forward;
- wyrand is the RNG.

The hit rate λ = −(m/(m+1)) ln(1 − (m+1)p/m) makes this exactly the depolarizing / flip channel; the
proof and an exact series check are in the doc.

**Equivalence.** Identical circuits in both directions (Stim's `rotated_memory_z` parsed by us; our
circuit exported to Stim), d = 3, 7, 11, 15, p = 0.1% and 0.3%:
- the hit tables' signatures equal Stim's DEM error set exactly (16/16 cells);
- **0 rejections in 155,531** per-detector, pairwise and event-count tests at 10⁶ shots (1%
  family-wise error per cell);
- a +10% error on one channel is rejected at |z| = 17.

**Timing.** Stim's own circuit (`surface_code:rotated_memory_z`, rounds = d), the same `.stim` file
for everyone, ptb64 output with observables to /dev/null, single thread, min of 3, interleaved.
Mshot/s, x86 VPS (EPYC-Rome, AVX2), 1-min load 0.8–3.0. "Stim native" is Stim 1.16 built from source
with AVX2, with its 64-shot process time subtracted. "End-to-end" compares whole processes, including
parse and compile.

| d | p | Stim pip (SSE2) | Stim native (AVX2) | qsim-lab | qsim-lab / native Stim | end-to-end |
|---|---|---|---|---|---|---|
| 3 | 0.1% | 41.3 | 38.5 | 696 | **18.1×** | 10.9× |
| 7 | 0.1% | 4.43 | 4.98 | 55.1 | **11.1×** | 8.4× |
| 11 | 0.1% | 1.13 | 1.27 | 11.8 | **9.3×** | 4.8× |
| 15 | 0.1% | 0.46 | 0.48 | 4.41 | **9.2×** | 2.5× |
| 3 | 0.3% | 25.9 | 26.5 | 317 | **12.0×** | 10.4× |
| 7 | 0.3% | 2.25 | 2.64 | 24.5 | **9.3×** | 7.9× |
| 11 | 0.3% | 0.57 | 0.65 | 6.08 | **9.4×** | 6.0× |
| 15 | 0.3% | 0.23 | 0.27 | 2.24 | **8.3×** | 3.7× |

- **8.3–18× faster than AVX2-native Stim on Stim's own circuit** (8.3–17× against the better of
  pip and native).
- Stim's DEM sampler is slower than its circuit sampler here (0.09–0.11 Mshot/s at d = 15).
- Same runs, against the pre-branch SymPhase path: 10–16×. The parts are:
  - Poisson hits: 1.6–2.6×;
  - the hit table: 2.0×;
  - blocked generation: 1.4–2.2×;
  - u16 tables plus wyrand: 1.02–1.26×.
- Tried and measured, not kept:
  - bit-sliced Bernoulli words cost 7.3 random words per group-word, 30–60× slower;
  - pooling the old geometric-gap streams gained 2–4%.
- **Compile is our weak spot:** about 55 ms at d = 15 against 3.5 ms for Stim's whole small run. At
  128k shots the end-to-end lead is therefore 2.5–3.7×; for ≥ 10⁶ shots it approaches the sampling
  ratio.
- **Independent audit (`research/qec/fast-sampler-audit.md`).**
  - Sampling ratios reproduced on x86 (d = 7 / 15, p = 0.1%: 11.7× / 9.2×, load 3.9) and on the
    M1 (26.7× / 27.0×).
  - Whole-process timing on x86 against Stim's best mode (`stim detect`; the DEM route is slower):

    | d | 10⁴ shots | 10⁵ shots | 10⁶ shots |
    |---|---|---|---|
    | 7 | 0.99× | 3.7× | 8.9× |
    | 15 | **0.40×** | 2.5× | 7.0× |

    So below about 3·10⁴ shots at d = 15, Stim is faster end to end.
  - Exactness verified against an exact branching ground truth, including channels near and past
    full mixing, and on circuits the author did not test (colour, repetition, unrotated surface
    code, random circuits).
  - One front-end crash was fixed: detectors with noiseless parity 1.
  - The method is the fault-sparse detector sampling described in the Stim paper (§5.6, not
    implemented there). The new part is making its constant factors win.
- Mac (M1 Pro, under the bench lock, load 2.9–6.7): 27–40× against pip Stim (d = 15: 6.5 / 3.2
  Mshot/s against 0.24 / 0.11). Stim has no NEON backend, so this overstates the algorithmic gain;
  the x86 ratio against AVX2 Stim is the fair one.

## 3. Colour-code syndrome schedules (`research/qec/qec-r4.md` Part 2)

Within Kishony & Fowler's single-auxiliary 6.6.6 design space (arXiv:2603.28852: same qubits, same
6 + 6 CNOT layers, collision-free):
- Their circuit distance d − ⌊(d+3)/6⌋ is reproduced exactly with an independent generator, DEM and
  exact solver (d = 3–9 over d rounds, d = 11 over 1 round).
- Per-plaquette boundary schedules found by exact local search keep d_circ but **halve the number of
  minimum-weight logicals**: 388 → 197 (d = 5), 12,901 → 6,627 (d = 7), 492 → 255 (d = 9), all over
  d rounds.
- Logical error per round under noisy-CNOT noise falls by **22–31 % at d = 5 and 7** (p = 0.2–0.3 %;
  BP+OSD and Tesseract agree at d = 5), 24 % at p = 0.1 % (d = 5), 12 % under uniform depolarizing
  noise (d = 5, p = 0.1 %). **No gain at d = 9** (1.02× [0.94, 1.11], p = 0.3 %).
- No schedule tried raised the circuit distance; the search is local, so this is not a proof.
- Audit (§16): the distances and minimum-weight counts were recomputed from **Stim's** DEM of the
  exported circuits with Stim's own distance search and an independent counter: d = 3 and 5 (1 and
  d rounds), 7 (1 and 7 rounds), 9 (1 round), K–F and LNS schedules, all identical (388/197,
  12,901/6,627, 883/434, 36/15).
  The LER ratios were re-derived from the raw counts.

## 4. Which exact engine to use: simulability study and planner

`research/simulability/simulability.md`, `research/simulability/planner.md`. 314 instances in four families, every engine timed
on the Mac, request ⟨Z^{⊗n}⟩.
- One cheap work estimate per engine, `log2 t_e ≈ a_e + b_e·R_e`, fitted on three families and tested
  on the fourth, picks the fastest engine **85 %** of the time (geo slowdown 1.25×, within 2× on 91 %,
  worst 162×). Per held-out family: ct 93 %, arith 88 %, brick 70 %, qaoa 71 %. The best single
  statistic (the rotation frame's active dimension) gets 54 % / 4.5×; "always state vector" 3 % / 82×.
- **Exact MPS cost is set by its bond trace.** A symbolic replay of the MPS engine's control flow
  reproduces its operation counts exactly (291/291) and predicts Mac MPS time to 0.08 decades given
  the real trace. With cheap rigorous Schmidt-rank bounds instead of the trace: 0.69 decades.
- **Planner v0** (now used for expectation values in `pipeline::simulate`), leave-one-family-out:
  **88 % top-1, geo regret 1.09, worst 12.9×**. End to end on the Mac, including planning and
  speculation: geo regret **2.5×** over all 314 instances (planning costs 0.5–20 ms against many
  sub-millisecond best engines), 1.34 with 1 ms slack, **1.20× on the 21 instances whose best engine
  takes ≥ 0.1 s** (that includes one 10 s timeout where MPS would take 2.3 s).
- Audit (§16): the LOFO numbers reproduce from the raw CSVs (`fit.py`, `fit_planner.py`, plus an
  independent end-to-end regret script: 2.51 / 41 % within 2× / 1.33 ε). A bug in the support-bound
  feature (not a valid upper bound for Toffolis with one constant control) was fixed; it changed no
  value in either dataset.

## 5. Magic atlas (`research/simulability/magic-atlas.md`)

367 instances of real algorithms profiled in O(gates·n): T-count, the rotation frame's active
dimension d (register size of the compressed state), factor structure, stabilizer entanglement.
- Almost every algorithm saturates d = n in its first few percent of gates (QFT on generic input,
  superposed adders, Shor, Grover, Trotter after one step, QAOA, VQE).
- Ground truth at n ≤ 12: the stabilizer nullity ν equals max d in 33/33 "generic" instances and
  gate by gate for HEA, QAOA, Trotter and QPE. Audit qualification: 17 of the 33 sit at d = n, where
  ν = n is what any state without a Pauli symmetry has, and 12 at d = n − 1 with one Z₂ symmetry;
  only QPE with a stabilizer eigenstate tests d = ν well below saturation. ν was recomputed
  independently (own code, own state vectors) for 8 instances at every checkpoint: identical.
- Arithmetic is the exception: Toffoli networks on classical or two-branch data have d = n but zero
  state magic (ν = 0 at every one of the Shor oracle's 340 gate boundaries, independently checked). A
  new "magic recycling" mode absorbs stabilizer factors back into the Clifford frame and simulates the
  62-bit windowed Shor oracle (256 qubits, T-count 190,050) with a 2-qubit register in about 1.3 s.
  Context from the audit: that state is two classical branches, and a plain two-branch evaluation
  takes 0.05 s in Python. The result is that a generic Clifford+T frame engine finds the structure
  automatically, not that the instance is hard.
- Where d (or the factor size) stays small the frame engines beat the state vector by 10²–10⁴× at
  n = 24; where d = n they lose by 1–30×.

## 6. State vector against other simulators

**x86 (VPS, 4 threads, f32).** qsim-lab's cache-blocked executor wins on the QFT by aggregating
diagonals (QFT-24: 0.096 s vs Google qsim 2.24 s and Qiskit Aer 1.74 s; QFT-26: 0.42 s vs 8.4 s and
8.0 s). **On generic circuits qsim is faster**: random brickwork, 22 qubits, qsim 0.173 s vs qsim-lab
0.330 s (1.9×); runtime AVX2+FMA dispatch later narrowed this to about 1.4× at n = 24. Aer is about 5×
slower than qsim-lab. (An earlier table with qsim at 0.77 s was contaminated by a concurrent run.)

**Apple M1 Pro (8 threads).** QFT-28: 0.94 s vs qsim 27.5 s and Aer 15.5 s; brickwork-26: 3.14 s vs
5.17 s and 13.2 s; brickwork-28: 12.5 s vs 19.9 s and 53.2 s. qsim's SIMD kernels target x86, so this
comparison favours qsim-lab. qsim and Aer times include their Python front ends.

## 7. Engine-by-engine speedups (interleaved A/B against the previous implementation)

| engine | workload | speedup | audited |
|---|---|---|---|
| cache-blocked SV | QFT-24 / brickwork-22 | 8–11× / ~5× | reproduced 5.5–14.8× |
| sign-tracking tableau | syndrome rounds, d = 21 | ~100–145× | correctness verified; speed not re-timed |
| rotation-frame Pauli paths | Clifford+T, 64 qubits, 36 T | ~50–63× | reproduced 50× |
| compiler passes | Bernstein–Vazirani-23 / GHZ-24 / noisy repetition code | 298× / 90× / 604× | pre-port version reproduced |
| adaptive (frame → compressed SV) | 22 qubits, 40 T vs full SV | 24 s → 0.005 s | pending |
| HSF | 20 qubits, cut of 4 gates vs old SV | ~20× | reproduced; smaller vs the blocked SV |
| SymPhase | per-shot vs tableau, d = 3–15 | 190–890× | exact-distribution check on 190 circuits |
| FastSampler (Poisson hits) | detector sampling vs old SymPhase path, Stim's circuit d = 3–15 | 10–16× | not yet audited; exact support + 10⁶-shot equivalence vs Stim |
| repeat detection | rep-code d = 9, 10⁶ rounds / diagonal layer n = 14 | 302× / 660× (Mac) | audit §15; loses (0.55×) on large Clifford blocks with few repeats |
| phase folding | Cuccaro adders on the adaptive engine | 1.12–1.16× (Mac); random C+T n = 32: 1594 → 590 non-Clifford | audit §15 |

## 8. Surface-code memory and decoding

- Below threshold at p = 0.3 % (50,000 shots per point), logical error 0.46 % / 0.28 % / 0.12 % /
  0.054 % / 0.024 % at d = 3 / 5 / 7 / 9 / 11; d = 11 (241 qubits, 11 rounds) takes 1.25 s.
- Circuit-level depolarizing threshold with the weighted union-find decoder ≈ 0.65–0.75 %, within the
  0.5–1 % reported for union-find.
- The hook-safe CNOT order restores full distance; a standard-looking order gave d = 5 an effective
  distance of 3 (confirmed by fault injection).

## 9. Weight-6 qLDPC codes beyond the known frontier (`research/qec/code-discovery-2.md`)

An exhaustive search of weight-6 (3 + 3) two-block group-algebra codes over the 1000 groups of order
≤ 150 that the first study (abelian, rank ≤ 2) did not cover — every non-abelian group and every
abelian group of rank ≥ 3, from GAP's SmallGroups — against the published weight-6 frontier
(`literature.md`, including the 2026 non-abelian and coset codes), Lin & Pryadko's public 2BGA dataset,
the Unitary Foundation qLDPC challenge board, and all direct sums of these.
- **Two new codes**, each strictly dominating published ones and exactly certified:
  **[[224,18,12]]** over C7 × ((C4 × C2) ⋊ C2) (dominates [[294,18,10]], [[252,14,12]],
  [[288,16,12]]; k·d²/n 11.57) and **[[288,34,8]]** over A4 × A4 (dominates [[288,32,6]] and
  [[292,18,8]]; 7.56). Neither raises the weight-6 k·d²/n record.
- The weight-6 record **[[288,16,16]]** (k·d²/n 14.22, above every weight-6 code in published papers,
  best 14.11) occurs over ten groups of order 144; its distance is proved exactly here for the first
  time. Its parameters are not new: they were posted to the qLDPC challenge in July 2026 (another
  2BGA code, distance an upper bound), and exist at weight 9; weight 7 does better.
- [[192,12,14]], [[192,16,12]], [[200,16,12]] are in Lin & Pryadko's 2023 dataset with randomized
  distances; their distances are proved exactly here.
- Certification: the exact symmetry-rooted search in Rust (both CSS sectors) and an independent C
  program with no symmetry that proves no lighter nontrivial logical exists (3.7·10⁸ and 3.8·10⁸
  nodes per sector for [[288,16,16]]), plus k and weight-d witnesses checked by an independent Python
  script; codes are also rebuilt from presentations or permutation representations and re-proved.
  Novelty is relative to the 40+ papers, the dataset and the challenge board we read, not proved.
## 10. Gidney 2025's approximate modular exponentiation, simulated on every branch (`research/shor/approx-modexp.md`)

**What.** The reference `approx_modexp` of Gidney 2025 (arXiv:2505.15917; code CC-BY-4.0,
doi:10.5281/zenodo.15347487) — approximate residue arithmetic (Chevignard–Fouque–Schrottenloher),
masking, measurement-based uncomputation with deferred "vent" phase corrections — simulated exactly
on every branch of the exponent register and the mask, followed by the exact frequency-basis
measurement. The paper's own verifier follows a few random classical trajectories and, in its words,
cannot verify interference or masking. Scaled-down instances (n = 8–24 bits), not the paper's
2048-bit regime.

- **Correct on every branch.** All 2^m · 2^mask branches (up to 1.07·10⁹ per run; n = 8–14,
  Shor-style and Ekerå–Håstad; all-0, all-1 and 8 random measurement-outcome streams each):
  every ancilla returns to 0, every residue is exact, every branch ends with sign +1 (all
  kickback from 1.4–4.6 k X-basis measurements per run cancelled, including the corrections
  deferred from loop3 to unloop3). MOON_CORRECTNESS
- **Independently checked.** The paper's own `approx_modexp`, run unchanged on a genuinely quantum
  backend (full superposition, real X-basis measurements; no branches ever merge), agrees with the
  simulator branch by branch and to 7·10⁻¹⁸ (Shor) / 3.5·10⁻¹⁸ (EH) in the output distribution
  (n = 8); every table equals the paper's precomputation code (5 configurations); one loop4 step
  and one loop3/unloop3 step pair compiled to X/CNOT/Toffoli (+ X-measurements, Z/CZ fix-ups)
  equal the quint-level semantics on every input, also on the repo's dense reference state vector.
- **Masking is necessary and, with it, the approximation is benign (interference).** At N = 899
  (m = 14, f = 8) the unmasked approximate circuit's frequency distribution is at TV 0.78 from
  exact arithmetic and loses 2.6× in success probability; with a mask of W values the distance
  falls as 1/W (TV ≈ 0.65·(1 − F), far below the trace-distance bound √(1 − F)).
  With the paper's own mask rule the approximate circuit's frequency distribution is within TV
  0.021 (N = 899, S = 0.57), 0.0070 (N = 3127, Shor, S = 0.33) and 0.0062 (N = 3127,
  Ekerå–Håstad) of exact arithmetic with the same mask (trace-distance bounds 0.17, 0.10,
  0.10); the frequency-peak structure, including the zero-peak enhancement predicted by the
  paper's Eq. 42, matches exact arithmetic (N = 899: P(peak 0) = 0.5717 vs 0.5717, Eq. 42:
  0.5695).
- **The paper's success model holds where it should.** For its Figure 4 instance (N = 3127,
  g = 3122) at the mask the paper's rule picks, the success suppression due to masking is 0.680 in
  the paper's model, 0.678 (exact arithmetic) and 0.671 (approximate circuit) in the exact
  simulation (1 − S = 0.673); absolute success values differ (0.478 model vs 0.406 circuit) because
  the model's QFT is idealised modulo the period. The exact evaluation of the paper's model
  reproduces its released Monte-Carlo data (24 cells, ≤ 0.0009).
- **The paper's error analysis is pessimistic at these sizes, with one slip.** The deviation is a
  constant bias plus a random walk of rounding errors (σ ≈ 0.3·√A for A accumulator additions;
  E|δ − c| = 1.2–1.9 units against a worst case of 51–190 units in the paper's model). Eq. 28 omits
  a factor 2 (`1 − |⟨ψ1|ψ̃1⟩|² ≤ 2ε/S`); the effect on the paper's 2048-bit estimate is negligible
  (9.21 → 9.25–9.27 expected shots).
- Caveats: scaled-down instances, not the paper's regime; exact QFT instead of the paper's
  phase-gradient frequency measurement; finite sets of outcome streams; only two subroutine steps
  compiled to gates. A toy-size corner case of the paper's prime search (pruning a prime that
  divides N) is documented in the notebook.

## Honest notes

- The adaptive-switching design is an independent rediscovery of **Clifft** (arXiv:2604.27058).
- Claims withdrawn or corrected after audit: the 4–6.5× Stim lead (§2; the new 8–18× FastSampler
  lead in §2 is a different sampler, measured against native AVX2 Stim, and not yet audited); the "every gate-level Shor
  record" wording (now "this repo's"; four quantitative sentences in `shor.md`); "40-bit" for
  Willsch et al.'s N (it is 39 bits); the noise doc's "exponential form holds to ≥ 3 faults" and
  "window model to ±0.02 for every instance" (rms 0.022, max 0.040); the magic atlas's d = ν and
  62-bit-Shor framing (qualified, §5).
- Bugs caught by audits: a blocked-executor phase-split bug; a 5.9× optimistic QEC error model;
  dropped reset errors and a p = 1 baked model; a tableau reset bug; `reset_all` missing new sign
  fields; QASM parser precedence and silent drops; `simulate_with` refusing repeated terminal
  measurements and a cross-copy-fusion regression (repeat); a ZX optimiser that changed 281/300
  unitaries (dropped); the simulability support bound under-counting (§4).
