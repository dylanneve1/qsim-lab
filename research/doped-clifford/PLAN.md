# IBM doped-Clifford challenge: plan (drafted 2026-10-06, parked until Peak Portal 12/12)

Target: tracker issue 228, `doped_random_graph_sampling_nq70_depth70_checks27`.
70 logical qubits, depth 70, 468 T, all two-qubit gates are CZ, qubits on an open 1D chain.
IBM value: graph-state fidelity 0.32; the doped fidelity is claimed via a bound, and the XEB bar is "significantly > 0.044".
Win conditions (from the issue): (a) classical samples with XEB significantly > 0.044, or (b) a method equally hard for the
graph state and the doped state, which simulates the graph state with DFE > 0.32.
Prior art: Manabe/Gu/Pan arXiv:2608.13110: a sweep along the chain, contraction width 35, all 2051 IBM amplitudes in
37.3 min on 256 H100 (log-XEB 0.35). That is amplitudes for IBM's bitstrings, not their own sampling; sampling projected
at 10.6 min. IBM's own estimates: 1e25 s for MPS, 1e42 s for quizx.

Key lever: the bar is a fidelity of ~0.05-0.3, not exact. Cost of approximate contraction scales roughly with the fidelity kept.
Sample budget for a 3 sigma result: N ~ 9/(f-0.044)^2 under Porter-Thomas, e.g. f=0.3 -> ~140 samples, f=0.1 -> ~2900.

Phases
0. Get the circuit and IBM's samples. Profile them: gate layers, where the T gates sit, Schmidt rank across every chain cut
   (claimed minimum 2^30), and light cones. Check how much of the circuit is diagonal (CZ/T/S commute) and whether
   layers can be merged.
1. Reproduce SUTD's chain-sweep path (width 35) in qsim-lab. Estimate cost on M1 Pro + VPS with slicing to fit 16 GB
   (2^35 c64 = 256 GB, so slice >= 2^5) before writing anything heavy.
2. Approximate routes, cheapest first:
   a. Drop slices: keep a fraction of slices so fidelity ~ fraction kept. Measure that on smaller cuts.
   b. Clifford-augmented MPS (CAMPS): absorb Cliffords into a frame. Only the 468 T gates (Pauli rotations) add
      bond dimension. Use Clifford disentangling and truncated bond dimension, and track the fidelity product.
   c. Pauli-path / Clifford-frame for the T layers, if T density per layer is low.
   d. Option (b) loophole check: does CAMPS make the graph state easy while the doped state stays hard?
      If so, option (b) doesn't apply. Be honest about it.
3. Validate on scaled-down instances (n = 30-40) against exact state vectors from the qsim-lab Metal engine: does XEB
   track predicted fidelity?
4. Full run: produce our own samples plus exact/approximate amplitudes, report XEB with a CI, and post the tracker
   issue + write-up in qsim-lab. Credit SUTD.
Constraints: Dylan's Mac (16 GB) for timings, VPS for orchestration. No GPU cluster, which is the whole point.

## Findings 2026-10-06 evening
- Exact push-T-to-end/start: the X-rank of the pushed Paulis is 70 either way, and every time-ordered split costs >= 70. Exact Clifford-frame tricks are dead.
  The late local T gates amount to a non-Clifford product-basis measurement (the #P-hard RGS setting).
- cotengra greedy/labels path search (60 s budget): depth 20 -> width 15, depth 30 -> width 23. Generic search is worse than
  the analytic chain sweep, so it was dropped.
- Chain sweep (analytic): every cut is crossed by 35 CZs (brickwork, depth 70), and CZ has operator rank 2, so the boundary is 2^35.
  256 GB c64, so slice >= 2^5 to fit 16 GB. Rough cost ~1e14 flop/amplitude: a few minutes on the M1 Pro GPU IF kernels reach ~1 TFLOPS.
  Fidelity lever: drop slices, keeping fidelity ~ fraction kept (holds for random circuits; to be checked here).
  f~0.25 -> ~200 samples for a 3 sigma result over 0.044 -> order of hours. Back-of-envelope only.
- NEXT: implement the chain sweep in qsim-lab (Metal), validate exactness at truncated depth vs the state vector, measure the real
  time per slice, then measure the fidelity-vs-slices-kept curve at depth ~40.

## Chain-sweep results (agent agt_e4102ac1, 2026-10-06 night; branch exp/chain-sweep @ e4b42e3)
- Engine built and exact (f64 err <= 8e-15 rel, f32/Metal <= 1e-6), 6 tests. Metal ~x2.07 per width bit; width 29 (D=58) = 71.6 s/amplitude on M1 Pro.
- CORRECTION to my earlier plan: slicing does NOT fit D=70 into 16 GB. In a chain sweep each bond crosses exactly one cut,
  so every one of the 69 cuts needs >=5 sliced bonds: 2^345 slices. Memory, not time, is the wall. Max exact on Mac: D=58-60.
- Fidelity = fraction of slices kept holds EXACTLY on IBM's circuit (slices orthogonal, equal norms; n=70 D=40: 0.262+-0.022 at f=1/4),
  only on average for a random control. Saves time, not memory.
- Next: approximate the boundary itself (e.g. MPS truncation along time) and measure its fidelity cost; scheduler to cut Metal passes 2-4x.

## 2026-10-07 early: two more routes closed
- Boundary MPS (agt_b6644718, exp/chain-sweep-mps): flat spectrum; F~0.3-0.43 at 0.75 chi_exact, ~0.011 at half. D=70 f=0.1 needs ~350 GB. DEAD.
- Approximate CAMPS (agt_563685bc, exp/camps 35d66d4): beats plain MPS by 14-67 nats at D<=48, but at D=70 -ln F ~ 60 at chi<=128
  (+0.55 nats per doubling). The late 249 T gates are simply dropped (39.4 nats). Readout also needs 2^E*chi with E=34. DEAD.
  IBM's README claim holds for the approximate version too.
- Single-16GB-machine routes so far all hit the same 2^34-35 wall.
- Remaining candidates: (1) hyper-optimised general TN slicing (not a chain sweep) at width <= 30. Measure slicing overhead with a
  long kahypar search; SUTD saw overhead explode, so verify. (2) Pool memory across mesh machines (no 256 GB available, likely dead).
  (3) Top-k post-selection only amplifies fidelity, doesn't fix memory. (4) A structural idea for the graph-state + T-layer form
  (e.g. split at layer 63: Clifford bulk exact as a stabilizer state, last 7 layers with 249 T as a shallow circuit; amplitude =
  sum over the light-cone of the last 7 layers; cost ~2^(boundary of the shallow block)). Worth costing next.

## Boundary-MPS results (agent agt_b6644718, 2026-10-07; branch exp/chain-sweep-mps)
- Engine `chain_mps`: the exact sweep's op stream on an MPS over the bond bits (time order, χ cap, kept-weight estimate). Exact at χ = 2^⌊D/4⌋, tests green.
- The exact boundary saturates Schmidt rank 2^min(t, D/2−t) at every time cut, with a near-flat (Clifford) spectrum. Truncation is a projection that compounds along the chain:
  ln F ≈ −j ln2 − r(j)·(#qubits after onset), j = log2(χ_ex/χ), r ≈ 0.069 j² (j ≤ 1), 0.26 (j=2), 0.6–1.3 (j=4–9). The estimate tracks the measured overlap F (D ≤ 40, n = 70, k up to 400).
- n=70 measured: j=0.42 → F 0.30–0.57; j=0.5 → 0.21±0.11 (D=40); j=1 → 0.01–0.05; j=2 → 1e-7; j=3 → 1e-12. D=70 pilot: χ=128/256/512 → fid_est 9e-42 / 6e-35 / 4e-30 (58/356/2385 s per amplitude, M1 Pro, 1 thread).
- D=70 extrapolation: f=0.1 needs χ ≈ 2^16.2 (~320 GiB MPS; ~250 GiB per half with MITM), which is MORE than the 256 GiB exact register. At 16 GB (χ ≤ 2^12.5): F ~ 1e-18 to 1e-20. Time ~1e9 s per amplitude. CLOSED.
- The only variant not ruled out: a Clifford-augmented boundary (a Clifford frame on the bond bits to strip stabilizer entanglement). Details in research/chain-sweep/BOUNDARY_MPS.md.

## 2026-10-07 afternoon: precision + folding closed on 16 GB
- Low precision (exp/chain-sweep-lowprec eb802ce): F = exp(-r R), r independent of D, and the register is Gaussian-like
  (lm2/lm3 r match Gaussian Lloyd-Max exactly). At R=71: int8 0.996 (64.5 GiB), int6 0.96 (50), int5 ~0.85 (42-44),
  int4 0.53 (36), lm3 0.08 (24), lm2 2e-4 (16).
- Folding (exp/fold-plan 25d5579): k worldlines per pass needs floor(k/2) extra stored bits; at a fixed 2^35 store R >= 70 (proved;
  71 achieved). Halving R doubles memory, so the levers cancel. 16 GB: F <= ~0.01 for every combination.
- Viable only with memory: fp16 128 GiB, F~0.999, ~60-120 machine-hours on a 48-core box (renting ruled out by Dylan).
- Also killed: ZX (width >=104), general slicing (+2^20-35 per bit), T-count reduction (468 -> 468), rotation-span rearrangement (~160 cross).
