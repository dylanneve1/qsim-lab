# Literature map: where qsim-lab's speed work sits

This note links each swarm topic to the relevant published work. It records what we're re-deriving, what's new, and what to try next.
Compiled 1 Oct 2026 from arXiv and conference sources. Each entry gives the core idea in one or two lines, plus what it means for us.

## State vector (`sv`)

- **Large-Scale Quantum Circuit Simulation on HPC Cluster via Cache Blocking, Boosting, and Gate Fusion Optimization** (arXiv:2604.12256, 2026). Cache blocking, a "merge booster" that fuses gates, and a "diagonal detector" that batches diagonal gates.
  *Us:* `exp/sv` independently arrived at the same three ideas: blocked executor, 1q fusion, and diagonal aggregation with ASAP layering. Our measured 3.7–10× (QFT-24 10.4×, brickwork-22 5.4×) is in line with the gains reported there. Our extra finding is the SoA (split re/im) layout: LLVM doesn't vectorise interleaved `Complex<f32>` multiply-adds, and splitting gives about 4× in cache.
- **Low-Level and NUMA-Aware Optimization for High-Performance Quantum Simulation** (arXiv:2506.09198, 2025). Fusing consecutive gates so the state vector is traversed fewer times, and node-level cache blocking. NUMA doesn't apply on a 4-vCPU VM.
- **Qandle: Accelerating State Vector Simulation Using Gate-Matrix Caching and Circuit Splitting** (arXiv:2404.09213). Caches gate matrices and splits circuits. *Us:* the circuit-splitting idea overlaps `compiler`'s independent-component pass.
- qsim, qHiPSTER and QuEST are the classical references for multithreaded state vectors (cited in 2604.12256).

## Stabilizer tableau (`stab`, `qec`)

- **Stim: a fast stabilizer circuit simulator** (Gidney, arXiv:2103.02202). Tableau with an inverse-tableau trick for deterministic measurement, plus *reference-sample + Pauli-frame* bulk sampling with 64+ shots bit-packed per word, and detector error models derived from the circuit.
  *Us:* `exp/stab` avoids re-laying-out the tableau between gate and measurement phases (d=21 syndrome rounds 0.97 s → 8.7 ms, ~110×). `exp/qec`'s circuit-derived DEM is exactly Stim's DEM idea; the earlier hand-written phenomenological model was the mistake the audit caught (5.9× too optimistic at d=3).
- **SymPhase: Phase Symbolization for Fast Simulation of Stabilizer Circuits** (arXiv:2311.03906, DAC 2024). Simulate the noisy circuit ONCE with *symbolic* phases: every possible Pauli fault becomes a variable in the stabilizer phases. Sampling a shot is then a GF(2) matrix–vector product over the sampled fault bits. Reported to beat Stim's frame sampler on sampling throughput.
  *Us:* the next step for many-shot QEC sampling. It's exact by construction (same distribution) and maps directly onto our bit-packed u64 machinery. It's a natural follow-on to stab's work and qec's DEM.
- **Sparse Blossom: correcting a million errors per core second with minimum-weight matching** (arXiv:2303.15933). Exact MWPM decoding at sub-µs per syndrome.
  *Us:* Union-Find is our decoder. MWPM gives a higher circuit-level threshold (~1% vs ~0.5–0.9% for UF). A weighted UF (log-likelihood weights from the circuit DEM) is the cheap middle step.
- **Logical error estimation from syndrome data of surface-code experiments** (arXiv:2606.11496, 2026). Estimates DEMs from syndrome statistics. Useful as a cross-check on our DEM.

## Clifford+T / Pauli paths (`pauli`)

- **Pauli Propagation: A Computational Framework for Simulating Quantum Systems** (Rudolph et al., arXiv:2505.21606, 2025; PauliPropagation.jl). The general Heisenberg-picture framework. Truncation strategies (Pauli weight, path weight, small coefficients) are what make it approximate. *Without* truncation it's exact, and that's our regime.
  *Us:* our `pauli_path` module is exact Pauli propagation. Their implementation notes on bit-packed symplectic storage and merge strategies are the relevant part. Truncation is lossy and rejected for qsim-lab.
- **PauliEngine: High-Performant Symbolic Arithmetic for Quantum Operations** (arXiv:2601.02233, 2026). Binary symplectic representation with bitwise operations in C++. Same representation as ours, so a useful benchmark point.
- **Sparse Pauli dynamics** (Begušić, Gray, Chan, Sci. Adv. 2024; arXiv:2409.03097). Pauli propagation for dynamics, with practical merging and hashing implementation details.
- **Classical simulability of Clifford+T circuits with Clifford-augmented MPS (CAMPS)** (arXiv:2412.17209). Disentangle T gates with Clifford circuits so that most of the "magic" ends up in a low-bond MPS; Pauli expectation values come out cheaply. *Us:* a genuinely different exact-when-untruncated route for Clifford+T that combines our tableau and MPS backends. A strong candidate for a new backend.

## ZX-calculus (`zx`)

- **Procedurally Optimised ZX-Diagram Cutting for Efficient T-Decomposition in Classical Simulation** (arXiv:2403.10964, 2024). Strong simulation by decomposing t T-gates into 2^{αt} stabiliser terms with α < 1, finding optimal vertex cuts in the ZX-diagram.
  *Us:* this is *exact* strong simulation with exponent αt < t. It's the ZX route to beating our Pauli-path 2^t-ish scaling, and goes further than T-count reduction alone.
- **Cutting stabiliser decompositions of magic state cultivation with ZX-calculus** (arXiv:2509.01224, 2025). Applies the above to QEC circuits: d=3 and d=5 cultivation states come out as sums of 4 and 8 Clifford diagrams.
- Kissinger & van de Wetering, *Reducing T-count with the ZX-calculus* (phase teleportation, 2020). Exact T-count reduction without circuit extraction. This is the target in `zx`'s brief.

## Hybrid Schrödinger–Feynman (`hsf`)

- Google qsim/qsimh (Markov et al.) is the reference HSF design: path sums over cut gates, with gate Schmidt rank 2 for CZ/CNOT.
  *Us:* `exp/hsf` (the rewrite) does depth-first paths with rank-2 cuts and Kernighan–Lin partitioning, checked against the state vector to 1e-12.

## What to try next, ranked by expected value per effort

1. **SymPhase-style symbolic-phase sampling** on top of `exp/stab` and `exp/qec`: one simulation, then a GF(2) mat-vec per shot. This is the biggest likely win for the surface-code threshold runs.
2. **Weighted Union-Find** with log-likelihood weights from the circuit-derived DEM, so the threshold moves toward the literature's ~0.7–1% range. Then a Sparse-Blossom-style MWPM if time allows.
3. **CAMPS backend** (Clifford tableau + MPS) for Clifford+T, compared against Pauli paths on the README's 64-qubit benchmark.
4. **ZX stabiliser-decomposition cutting** (2^{αt}) as an exact strong-simulation backend.
5. **sv:** compare our fusion/blocking against the 2604.12256 "merge booster" policy, and try wiring the blocked executor into the default `Circuit::run` behind a size threshold.

## Colour-code syndrome extraction (qec-r4)
- **Color code off-the-hook** (Kishony & Fowler, arXiv:2603.28852, 2026). Colour-dependent single-auxiliary schedules that keep the bulk distance at minimal depth. Boundary "fractional hooks" give d_circ = d − ⌊(d+3)/6⌋. Their companion code searches colour-uniform schedules only.
  *Us (`research/qec-r4.md`):* we reproduce their d_circ exactly (d ≤ 9 over d rounds) with our own generator, DEM and exact solver. Per-plaquette boundary schedules in the same design space halve the number of minimum-weight logicals and give 24–31% lower LER under noisy-CNOT noise at p = 0.2–0.3%. Their d_circ is not improved.
- **Tesseract decoder** (Google, 2025). Search-based near-MLE decoder for hypergraph DEMs, used by K–F; we use it as a cross-check of our BP+OSD.
