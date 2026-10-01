# qsim-lab results (1 October 2026)

All numbers were measured on the development VM (4 vCPUs AMD EPYC-Rome with AVX2, 7.7 GB RAM, shared with other services) unless stated. Every engine below is **exact**: no truncation, and checked against an independent reference state vector to 1e-12 (f64) or 1e-5 (f32), or against exact outcome distributions. An independent audit agent fuzzed each change before or right after merge; see `research/audit.md`. Raw data and methods are in `research/`.

## Head-to-head against other simulators (same circuits, same machine, 4 threads, f32, min of 3)

### State vector: qsim-lab blocked executor vs Google qsim (qsimcirq 0.22.1) and Qiskit Aer (0.17.2)
| circuit | qsim-lab | qsim | Aer |
|---|---|---|---|
| QFT, 22 qubits | **0.022 s** | 0.77 s | 0.32 s |
| QFT, 24 qubits | **0.096 s** | 2.24 s | 1.74 s |
| QFT, 26 qubits | **0.42 s** | 8.4 s | 8.0 s |
| random brickwork, 22 qubits, 1,090 gates | 0.30 s | ~~0.77 s~~ 0.173 s (clean re-run) | 1.48 s |
| random brickwork, 24 qubits, 1,190 gates | 1.28 s | 1.80 s (contaminated; see correction) | 6.26 s |

**Correction (same day).** The qsim column above was measured while an Aer benchmark was still running on the same 4 cores, so it's contaminated. A clean re-run by the sv-monomial agent (load ~3, `research/sv-monomial.md` on branch exp/sv-monomial) has **qsim ahead on generic circuits on this x86 box**:
- random brickwork, 22 qubits: qsim (2-qubit fusion) **0.173 s** vs qsim-lab 0.330 s, so **qsim is 1.9× faster**;
- Aer is still 5× slower than qsim-lab.

qsim's edge comes from dense 2-qubit gate fusion with AVX2 kernels, which qsim-lab doesn't have yet. That work is in progress on exp/sv-monomial. The QFT advantage (diagonal aggregation) is unaffected.

**On Apple M1 Pro (8 cores, sequential runs, nothing else running)** qsim-lab is ahead of both:

| circuit | qsim-lab | qsim | Aer |
|---|---|---|---|
| QFT-28 | **0.94 s** | 27.5 s | 15.5 s |
| brickwork-26 | **3.14 s** | 5.17 s | 13.2 s |
| brickwork-28 | **12.5 s** | 19.9 s | 53.2 s |

qsim's SIMD kernels target x86 (SSE/AVX), so on ARM this comparison favours qsim-lab and shouldn't be generalised.

Other caveats:
- qsim and Aer times include their Python front end and returning the state.
- Both are built for large servers and GPUs.

### Surface-code sampling: qsim-lab SymPhase detector sampler vs Stim (1.16.0), single thread
Rotated surface-code memory, rounds = d, circuit-level noise p = 0.3%, bit-packed output, min-of-5 through `bench.sh`. Both simulators sampled the **exact same exported circuit** (`examples/stim_export.rs`) with matching gate order, noise placement (`DEPOLARIZE1`, `DEPOLARIZE2`, `X_ERROR` after reset, `MZ` readout error), and identical detector/observable definitions. All per-detector and observable firing rates agree within Bonferroni bounds (max |z| ≤ 3.34 across 1,004 statistical checks; 0 disagreements).

| distance | qubits | detectors | qsim-lab (shots/s) | Stim (shots/s) | ratio (qsim/Stim) |
|---|---|---|---|---|---|
| 3 | 17 | 16 | **5.51×10⁷** | 8.44×10⁶ | **6.53×** |
| 5 | 49 | 72 | **1.21×10⁷** | 2.80×10⁶ | **4.33×** |
| 7 | 97 | 192 | **4.57×10⁶** | 1.03×10⁶ | **4.44×** |
| 11 | 241 | 720 | **9.63×10⁵** | 2.28×10⁵ | **4.22×** |
| 15 | 449 | 1792 | **4.40×10⁵** | 1.09×10⁵ | **4.05×** |

Audited apples-to-apples comparison confirms qsim-lab SymPhase is **4.0×–6.5× faster** than Stim 1.16 across all tested distances on the development VM (see `research/audit.md` §13).

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
- **Shor's algorithm:** factors 1,005,973 = 997 × 1009 in 0.024 s / 10 MB, using one recycled control qubit (21 qubits) and an exact sparse state. Dense f32 goes to 26-bit N. **The full gate-level circuit at N ≈ 10⁶:** with a ripple-carry modular multiplier built only from X, CNOT and Toffoli gates, which keeps the exact sparse state at ≤ 2r amplitudes, 1,005,973 = 997 × 1009 factors in **10.8 s / 14 MB**. That's 64 qubits and 1,148,440 gates (415,498 Toffolis), with the same measured value and period as the oracle version. With a permutation oracle at this size, the cost tracks the classical period-finding difficulty (see `research/shor.md`).
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
