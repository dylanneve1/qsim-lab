# qsim-lab results (4 October 2026)

What this repository has shown so far, with the caveats that go with each number. Every engine is
**exact** (no truncation) and differential-tested against an independent reference state vector
(1e-12 in f64, 1e-5 in f32) or against exact outcome distributions. Each merged branch was audited by a
separate agent that tried to break its headline claim; the audits, including the claims that were
corrected or withdrawn, are in `research/audit.md` (§16 is the latest), `research/shor-r4-audit.md` and
`research/qec-r4.md` Part 1. Raw data and methods are in `research/`.

Machines:
- **VPS**: 4 vCPU AMD EPYC-Rome (AVX2), 7.7 GB, shared with other agents. Timings are only quoted when
  the 1-minute load was ≤ 4.
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

**This repo's record (Mac, `research/shor.md` round 4):** N = 1,537,596,787 (31 bits, generic: the
first balanced semiprime of a seeded generator, random base), **132 qubits, 1.70 M gates per run,
r = 256,252,500, factored in 134 s (f32), 4.28 GB**. Cost is linear in the order r, so it is exponential
in the bit length for generic N: 24–29-bit N take 0.2–44 s depending on r. Structured N = p(2p − 1)
(classically trivial, λ ≈ √(2N)) go to 52 bits (216 qubits, 7.4 M gates, 60 s), which only shows that
r, not N, sets the cost. The old N ≈ 10⁶ record circuit (1.15 M gates) now takes 0.05 s instead of 10.8 s.

**Context.** Larger simulated Shor runs exist: Willsch et al. 2023 (arXiv:2308.05047) factored the
39-bit 549,755,813,701 = 712,321 × 771,781 by simulating Shor's algorithm on a GPU supercomputer
(JUWELS Booster), with a different circuit construction, so the two are not like-for-like. What is
specific here is that every gate of a compilable X/CNOT/Toffoli circuit is simulated exactly on a
laptop.

**Under circuit noise (`research/shor-noise.md`).** Exact Pauli-noise trajectories of the same circuit,
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

## 2. Surface-code sampling vs Stim 1.16 (corrected 4 Oct 2026)

**Correction.** The earlier claim "4.0–6.5× faster than Stim" is withdrawn. It timed Stim's slow numpy
output path against qsim-lab without output, on a hand-written copy of the circuit.

**Equivalence.** Both simulators now sample identical circuits in both directions (qsim-lab's
`Circuit` serialised op by op; Stim's generated `rotated_memory_z` parsed into qsim-lab). At d = 3, 7,
11, 15 with 10⁶ shots per side, every per-detector rate, every DEM-correlated detector pair and the
observable rate agree (0 rejections in 69,476 tests at 1 % family-wise error); a +10 % error on one
noise channel is rejected at |z| = 17.

**Timing** (same `.stim` file, same output layout ptb64, single thread, min of 3, interleaved;
Mshots/s; A = qsim-lab's sequential circuit, B = Stim's generated layer-parallel circuit).

x86 (VPS, load 3.5–4.2). pip's Stim wheel runs its SSE2 build; "native" is Stim built with AVX2.

| d | circuit | Stim pip (SSE2) | Stim native (AVX2) | qsim-lab | qsim-lab / best Stim |
|---|---|---|---|---|---|
| 3 | A | 30.57 | 26.59 | 48.79 | 1.60× |
| 3 | B | 22.26 | 26.38 | 26.54 | **1.01×** |
| 7 | A | 2.46 | 2.75 | 4.40 | 1.60× |
| 7 | B | 2.28 | 2.55 | 2.50 | **0.98×** |
| 11 | A | 0.59 | 0.70 | 1.17 | 1.66× |
| 11 | B | 0.58 | 0.69 | 0.62 | **0.89×** |
| 15 | A | 0.24 | 0.27 | 0.44 | 1.66× |
| 15 | B | 0.23 | 0.26 | 0.24 | **0.89×** |

- **On Stim's own well-layered circuit we are at parity (0.89–1.01× of AVX2 Stim).** That is the
  fair headline.
- On our sequential circuit (A) we are 1.6–1.7× faster only because Stim pays per-instruction
  overhead on its ~2,000 one-gate lines; A and B are different circuits and should not be compared
  with each other.
- On the M1 the ratios are 3.0–5.6×, but Stim has no NEON backend (both the wheel and a native build
  use 64-bit words), so the Mac numbers flatter us. Mac table: `research/qec-r4.md` §1.5.
- 72–82 % of our time is drawing noise variables (~45 ns per fault), not the GF(2) evaluation. Our
  compile step costs 35–45 ms at d = 15 against ~1 ms for Stim.

## 3. Colour-code syndrome schedules (`research/qec-r4.md` Part 2)

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

`research/simulability.md`, `research/planner.md`. 314 instances in four families, every engine timed
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

## 5. Magic atlas (`research/magic-atlas.md`)

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
| repeat detection | rep-code d = 9, 10⁶ rounds / diagonal layer n = 14 | 302× / 660× (Mac) | audit §15; loses (0.55×) on large Clifford blocks with few repeats |
| phase folding | Cuccaro adders on the adaptive engine | 1.12–1.16× (Mac); random C+T n = 32: 1594 → 590 non-Clifford | audit §15 |

## 8. Surface-code memory and decoding

- Below threshold at p = 0.3 % (50,000 shots per point), logical error 0.46 % / 0.28 % / 0.12 % /
  0.054 % / 0.024 % at d = 3 / 5 / 7 / 9 / 11; d = 11 (241 qubits, 11 rounds) takes 1.25 s.
- Circuit-level depolarizing threshold with the weighted union-find decoder ≈ 0.65–0.75 %, within the
  0.5–1 % reported for union-find.
- The hook-safe CNOT order restores full distance; a standard-looking order gave d = 5 an effective
  distance of 3 (confirmed by fault injection).

## Honest notes

- The adaptive-switching design is an independent rediscovery of **Clifft** (arXiv:2604.27058).
- Claims withdrawn or corrected after audit: the 4–6.5× Stim lead (§2); the "every gate-level Shor
  record" wording (now "this repo's"; four quantitative sentences in `shor.md`); "40-bit" for
  Willsch et al.'s N (it is 39 bits); the noise doc's "exponential form holds to ≥ 3 faults" and
  "window model to ±0.02 for every instance" (rms 0.022, max 0.040); the magic atlas's d = ν and
  62-bit-Shor framing (qualified, §5).
- Bugs caught by audits: a blocked-executor phase-split bug; a 5.9× optimistic QEC error model;
  dropped reset errors and a p = 1 baked model; a tableau reset bug; `reset_all` missing new sign
  fields; QASM parser precedence and silent drops; `simulate_with` refusing repeated terminal
  measurements and a cross-copy-fusion regression (repeat); a ZX optimiser that changed 281/300
  unitaries (dropped); the simulability support bound under-counting (§4).
