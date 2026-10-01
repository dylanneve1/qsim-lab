# qsim-lab results (1 October 2026)

All numbers were measured on the development VM (4 vCPUs AMD EPYC-Rome with AVX2, 7.7 GB RAM, shared with other services) unless stated. Every engine below is **exact**: no truncation, and checked against an independent reference state vector to 1e-12 (f64) or 1e-5 (f32), or against exact outcome distributions. An independent audit agent fuzzed each change before or right after merge; see `research/audit.md`. Raw data and methods are in `research/`.

## Head-to-head against other simulators (same circuits, same machine, 4 threads, f32, min of 3)

### State vector: qsim-lab blocked executor vs Google qsim (qsimcirq 0.22.1) and Qiskit Aer (0.17.2)
| circuit | qsim-lab | qsim | Aer |
|---|---|---|---|
| QFT, 22 qubits | **0.022 s** | 0.77 s | 0.32 s |
| QFT, 24 qubits | **0.096 s** | 2.24 s | 1.74 s |
| QFT, 26 qubits | **0.42 s** | 8.4 s | 8.0 s |
| random brickwork, 22 qubits, 1,090 gates | **0.30 s** | 0.77 s | 1.48 s |
| random brickwork, 24 qubits, 1,190 gates | **1.28 s** | 1.80 s | 6.26 s |

Caveats:
- qsim and Aer times include their Python front ends and returning the state; their best Aer/qsim settings are shown (fusion on or off, fused-gate size 2–4).
- Both are built for large servers and GPUs; this comparison is a 4-vCPU VM only.
- The QFT advantage comes from aggregating diagonal (controlled-phase) gates into single passes. On generic dense circuits the margin over qsim is ~1.4×.

### Surface-code sampling: qsim-lab SymPhase detector sampler vs Stim (1.16.0), single thread
Rotated surface-code memory, rounds = d, circuit-level noise p = 0.3%.

| distance | qsim-lab (shots/s) | Stim (shots/s) |
|---|---|---|
| 3 | **2.9×10⁷** | 7.2×10⁶ |
| 7 | **2.3×10⁶** | 7.0×10⁵ |
| 11 | **5.6×10⁵** | 1.7×10⁵ |
| 15 | **2.1×10⁵** | 4.7×10⁴ |

**Not yet an apples-to-apples claim.** Stim sampled its own generated circuit and qsim-lab sampled its own; the structure and noise placement are similar but not identical. The fair test (both simulators on one exported circuit, with matching detector statistics) is pending.

## Engine-by-engine speedups (interleaved A/B against the previous implementation; independently reproduced where noted)
| engine | workload | speedup | audited |
|---|---|---|---|
| cache-blocked SV | QFT-24 / brickwork-22 | 8–11× / ~5× | reproduced 5.5–14.8× |
| sign-tracking tableau | syndrome rounds, d=21 | ~100–145× | correctness verified; speed not re-timed |
| rotation-frame Pauli paths | Clifford+T, 64 qubits, 36 T | ~50–63× (same term count); T wall 40 → 100 | reproduced 50× |
| compiler passes | Bernstein–Vazirani-23 / GHZ-24 / noisy repetition code | 298× / 90× / 604× | pre-port version reproduced (745×, 148×) |
| adaptive (frame → compressed SV) | 22 qubits, 40 T vs full SV | 24 s → 0.005 s | pending |
| HSF | 20 qubits, cut of 4 gates vs old SV | ~20× (amplitudes 100×+) | reproduced; smaller vs the blocked SV |
| SymPhase | per-shot vs tableau, d=3–15 | 190–890× | exact-distribution check passed on 190 circuits |

## Things it can do now
- **Shor's algorithm:** factors 1,005,973 = 997 × 1009 in 0.024 s / 10 MB, using one recycled control qubit (21 qubits) and an exact sparse state. Dense f32 goes to 26-bit N. The fully gate-level version (Beauregard adder) reaches 10-bit N; a sparse-preserving ripple-carry oracle is next. With a permutation oracle at this size, the cost tracks the classical period-finding difficulty (see `research/shor.md`).
- **Surface-code memory below threshold:** p = 0.3%, 50,000 shots per point, with the logical error rate roughly halving at each distance step:

  | distance | 3 | 5 | 7 | 9 | 11 |
  |---|---|---|---|---|---|
  | logical error | 0.46% | 0.28% | 0.12% | 0.054% | 0.024% |

  d = 11 (241 qubits, 11 rounds) takes 1.25 s.
- **Threshold:** the circuit-level depolarizing threshold with the weighted union-find decoder is ≈ 0.65–0.75%, within the 0.5–1% the literature reports for union-find.
- **Hook errors:** the hook-safe CNOT schedule restores full distance. The standard-looking order gave the d = 5 code an effective distance of 3; this was confirmed by direct fault injection.

## Honest notes
- The "adaptive switching" design was later found to be an independent rediscovery of **Clifft** (arXiv:2604.27058). See `research/adaptive.md`.
- Bugs caught by the audit process before or right after merge:
  - a blocked-executor phase-split bug (hotfixed, with a regression test);
  - a 5.9× optimistic QEC error model (replaced by a circuit-derived one);
  - dropped reset errors and a p = 1 baked model;
  - a tableau reset bug on main;
  - a PR-integration bug where `reset_all` missed the new sign fields;
  - QASM parser precedence and silent-drop issues.

  All of them are documented in `research/audit.md`.
