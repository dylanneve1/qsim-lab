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

qsim's edge was first attributed to dense 2-qubit fusion; `research/sv-monomial.md` §2 (branch exp/simd) refutes that: dense fusion saves no flops on brickwork and the WIP fusion was slower. Runtime AVX2+FMA dispatch cuts our CPU time ~15-25%, which narrows but does not close the gap (~1.7x -> ~1.4x at n=24); the QFT advantage (diagonal aggregation) is unaffected.

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

### Surface-code sampling: qsim-lab SymPhase detector sampler vs Stim 1.16.0, single thread (corrected 4 Oct 2026)

**Correction.** The earlier claim here, "4.0–6.5× faster than Stim", is withdrawn. It timed Stim through `sample(bit_packed=True)`, its slow in-memory numpy path, and qsim-lab without storing output. It also used a hand-written re-implementation of the circuit (`examples/stim_export.rs`). `research/qec-r4.md` Part 1 has the full redo.

**Equivalence.** Both simulators now sample **identical circuits, in both directions**:
- qsim-lab's surface-code `Circuit` serialised op by op (`stim_io::to_stim`);
- Stim's own `surface_code:rotated_memory_z`, parsed into qsim-lab (`stim_io::parse_stim`).

At d = 3, 7, 11, 15 with 10⁶ shots per side, every per-detector rate, every DEM-correlated detector pair and the observable rate agree: 0 rejections in 69,476 tests at 1% family-wise error. The DEM supports are identical. A deliberate +10% error on one noise channel is rejected at |z| = 17.

**Timing.** Same `.stim` file and the same output layout (ptb64, 64-shot words per detector), single thread, min of 3, interleaved. Throughput is in Mshots/s. "A" is qsim-lab's sequential circuit; "B" is Stim's generated, layer-parallel circuit.

**x86: VPS EPYC-Rome, 1-minute load 3.5–4.2.** pip's Stim wheel runs its SSE2 build; "native" is Stim built from source with `-DSIMD_WIDTH=256` (AVX2), via `stim detect`.

| d | circuit | Stim pip (SSE2) | Stim native (AVX2) | qsim-lab | qsim-lab / best Stim |
|---|---|---|---|---|---|
| 3 | A (ours) | 30.57 | 26.59 | 48.79 | **1.60×** |
| 3 | B (Stim) | 22.26 | 26.38 | 26.54 | **1.01×** |
| 7 | A (ours) | 2.46 | 2.75 | 4.40 | **1.60×** |
| 7 | B (Stim) | 2.28 | 2.55 | 2.50 | **0.98×** |
| 11 | A (ours) | 0.59 | 0.70 | 1.17 | **1.66×** |
| 11 | B (Stim) | 0.58 | 0.69 | 0.62 | **0.89×** |
| 15 | A (ours) | 0.24 | 0.27 | 0.44 | **1.66×** |
| 15 | B (Stim) | 0.23 | 0.26 | 0.24 | **0.89×** |

**Apple M1 Pro, load about 3, under the Mac bench lock.** Caveat: Stim has **no NEON backend**. Both the pip wheel (`_stim_polyfill`) and a `-mcpu=native` source build use 64-bit words, so these ratios flatter qsim-lab and should not be generalised.

| d | circuit | Stim pip | Stim native | qsim-lab (sparse, SmallRng) | qsim-lab / best Stim |
|---|---|---|---|---|---|
| 3 | A (ours) | 17.25 | 16.77 | 77.32 | 4.48× |
| 3 | B (Stim) | 14.29 | 13.88 | 42.87 | 3.00× |
| 7 | A (ours) | 1.27 | 1.28 | 7.00 | 5.48× |
| 7 | B (Stim) | 1.15 | 1.16 | 4.32 | 3.73× |
| 11 | A (ours) | 0.32 | 0.32 | 1.75 | 5.45× |
| 11 | B (Stim) | 0.30 | 0.30 | 1.16 | 3.87× |
| 15 | A (ours) | 0.12 | 0.12 | 0.69 | 5.62× |
| 15 | B (Stim) | 0.12 | 0.12 | 0.47 | 4.02× |

**Summary.**
- On x86, on Stim's own circuit, qsim-lab and Stim are **at parity (0.89–1.01× against AVX2 Stim)**.
- On qsim-lab's sequential circuit we are **1.6–1.7×** faster, because Stim pays per-instruction overhead for its about 2,000 one-gate lines.
- 72–82% of our time goes to drawing noise variables (about 45 ns per fault event: RNG, `ln` for the geometric skip, Pauli choice), not to the GF(2) evaluation.
- The bit-identical sparse path with `SmallRng` gains 5–25% on x86 and 30–45% on M1.
- Our compile step costs 35–45 ms at d = 15, against about 1 ms for Stim.

Data: `research/data/qec-r4/stim_*`.

## Engine-by-engine speedups (interleaved A/B against the previous implementation; independently reproduced where noted)
| engine | workload | speedup | audited |
|---|---|---|---|
| cache-blocked SV | QFT-24 / brickwork-22 | 8–11× / ~5× | reproduced 5.5–14.8× |
| sign-tracking tableau | syndrome rounds, d=21 | ~100–145× | correctness verified; speed not re-timed |
| rotation-frame Pauli paths | Clifford+T, 64 qubits, 36 T | ~50–63× (same term count); T wall 40 → 100 | reproduced 50× |
| compiler passes | Bernstein–Vazirani-23 / GHZ-24 / noisy repetition code | 298× / 90× / 604× | pre-port version reproduced (745×, 148×) |
| adaptive (frame → compressed SV) | 22 qubits, 40 T vs full SV | 24 s → 0.005 s | pending |
| HSF | 20 qubits, cut of 4 gates vs old SV | ~20× (amplitudes 100×+) | reproduced; smaller vs the blocked SV |
| SymPhase | per-shot vs tableau, d=3–15 | 190–890× | exact-distribution check passed on 190 circuits; vs Stim: parity on x86 (see above) |

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
