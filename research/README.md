# Research notebooks

Every experiment in qsim-lab is written up as a lab notebook: what was tried,
how it was measured, the numbers (including negative results), and the
caveats. Raw data and the scripts that produced it live in
[`data/<topic>/`](data/). Headline results are collected in
[RESULTS.md](../RESULTS.md); corrections to earlier claims are logged in
[process/audit.md](process/audit.md). How to write a notebook:
[CONTRIBUTING.md](../CONTRIBUTING.md#research-notebooks).

Lab-wide design document: [ARCHITECTURE.md](ARCHITECTURE.md).

## Shor's algorithm — [`shor/`](shor/)

| Notebook | Topic |
|---|---|
| [shor.md](shor/shor.md) | Scaling Shor's algorithm: sparse/sliced state, ripple and windowed oracles, gate-level runs to 31-bit generic N (start here) |
| [superopt.md](shor/superopt.md) | Superoptimising the gate-level Shor oracle (1.70 M → 1.04 M gates at 31 bits) |
| [mbu-shor.md](shor/mbu-shor.md) | Measurement-based uncomputation in the exact gate-level simulation |
| [ge-shor.md](shor/ge-shor.md) | Gidney–Ekerå techniques: windowing, Ekerå–Håstad, coset representation |
| [shor-xl.md](shor/shor-xl.md) | Willsch et al.'s 39-bit N and a 43-bit generator N factored gate by gate on one VM (cost = support); generic-N frontier, memory model, AVX-512 slice evaluator |
| [approx-modexp.md](shor/approx-modexp.md) | Exact all-branch simulation of Gidney 2025's approximate residue-arithmetic modular exponentiation (masking, measurement-based uncomputation, interference), against the paper's own code and success model |
| [shor-noise.md](shor/shor-noise.md) | Gate-level Shor under circuit noise, measured at scale |
| [ft-shor.md](shor/ft-shor.md) | Shor on error-corrected (concatenated Steane) qubits, simulated end to end at the gate level |
| [noise-oracles.md](shor/noise-oracles.md) | Noise tolerance of the new (smaller) Shor oracles under circuit noise |
| [shor-r4-audit.md](shor/shor-r4-audit.md) | Independent audit of the round-4 31-bit results |

## Quantum error correction — [`qec/`](qec/)

| Notebook | Topic |
|---|---|
| [qec.md](qec/qec.md) | Circuit-derived detector error model and the surface-code threshold (start here) |
| [schedules.md](qec/schedules.md) | Surface-code syndrome-extraction schedule search |
| [qec-r4.md](qec/qec-r4.md) | Identical-circuit Stim comparison; colour-code schedule search |
| [colour-global.md](qec/colour-global.md) | Colour-code schedules: global search and optimality certificates |
| [colour-flags.md](qec/colour-flags.md) | Colour-code flags: full circuit distance as real circuits |
| [neural-decoder.md](qec/neural-decoder.md) | A learned decoder trained on FastSampler data: surface code and colour code |
| [alphaqubit-lite.md](qec/alphaqubit-lite.md) | Open AlphaQubit-style recurrent-transformer decoder, pretrained on FastSampler, fine-tuned and tested on real Google Sycamore/Willow data |
| [fast-sampler.md](qec/fast-sampler.md) | Fast detector sampling: Poisson hits into precomputed detector tables |
| [fast-sampler-audit.md](qec/fast-sampler-audit.md) | Independent audit of the FastSampler speed and equivalence claims |
| [code-discovery.md](qec/code-discovery.md) | Exhaustive search of weight-6 two-block (BB/GB/coprime-BB) codes, n ≤ 300: exact [[n,k,d]] frontier vs the literature, depth-7 schedules, circuit-level LER |
| [code-discovery-2.md](qec/code-discovery-2.md) | Weight-6 two-block group-algebra codes over every group of order ≤ 150 (1000 non-abelian / rank-3 groups) and coset codes over Z_m × K, against the 2026 frontier (non-abelian and coset codes included) |

## Simulability, magic and engine choice — [`simulability/`](simulability/)

| Notebook | Topic |
|---|---|
| [simulability.md](simulability/simulability.md) | Phase diagram of exact simulability across engines (start here) |
| [magic-atlas.md](simulability/magic-atlas.md) | A magic atlas of real quantum algorithms |
| [lowmagic-chem.md](simulability/lowmagic-chem.md) | Low-magic quantum chemistry beyond SV size (negative result: d = tapered qubit count) |
| [magic-transition.md](simulability/magic-transition.md) | A simulability transition in monitored Clifford+T circuits |
| [planner.md](simulability/planner.md) | Planner v0, and predicting the cost of exact MPS |
| [planner-v2.md](simulability/planner-v2.md) | Planner v2: samples, amplitudes, cheaper planning |
| [adaptive.md](simulability/adaptive.md) | Adaptive representation switching (`src/engines/adaptive.rs`) |
| [spoof-utility.md](simulability/spoof-utility.md) | Classical (laptop) reproduction of IBM's 127-qubit kicked-Ising utility experiment with sparse Pauli dynamics; 433/1121-qubit lattices |

## Engine performance — [`performance/`](performance/)

| Notebook | Topic |
|---|---|
| [sv.md](performance/sv.md) | State-vector speed: cache blocking, SIMD kernels |
| [sv-monomial.md](performance/sv-monomial.md) | k-qubit dense fusion and monomial-segment fusion |
| [dense-fusion.md](performance/dense-fusion.md) | Dense k-qubit fusion in the blocked executor: 1.5-2.15x on generic 2-qubit unitaries (M1 Pro), a loss on brickwork; cost rule |
| [mac-m1.md](performance/mac-m1.md) | Apple M1 Pro: NEON FMA kernels, nested L1 tiling, block size |
| [autoimprove.md](performance/autoimprove.md) | Automated propose → verify → benchmark → keep loop for the blocked SV kernels: 1.14x single-thread / 1.26x 4-thread on M1 Pro from lookahead stage planning, per-thread scratch and a pair kernel; method and honest assessment |
| [metal.md](performance/metal.md) | Metal (Apple GPU) f32 state-vector backend |
| [ooc.md](performance/ooc.md) | Out-of-core state vector: fewer passes over disk |
| [pipeline.md](performance/pipeline.md) | One entry point; blocked executor by default |
| [stab.md](performance/stab.md) | Stabilizer tableau speed |
| [pauli.md](performance/pauli.md) | Pauli paths (Clifford+T) |
| [hsf.md](performance/hsf.md) | Hybrid Schrödinger–Feynman backend |
| [mps.md](performance/mps.md) | MPS backend optimisation |

## Compiler — [`compiler/`](compiler/)

| Notebook | Topic |
|---|---|
| [compiler.md](compiler/compiler.md) | Exact circuit-level passes and plans (`src/compile/`) |
| [dag.md](compiler/dag.md) | Circuit DAG IR (`src/dag.rs`) |
| [phasepoly.md](compiler/phasepoly.md) | Phase folding: graph-based T-count reduction |
| [repeat.md](compiler/repeat.md) | Exact exploitation of repeated blocks (`compile::repeat`) |
| [graph-compiler.md](compiler/graph-compiler.md) | Graph compiler: compile-once/bind, subgraph dedup, rewrites, basis folding, engine partitioning (`src/graph/`) |

## Theory — [`theory/`](theory/)

| Notebook | Topic |
|---|---|
| [theory-shor.md](theory/theory-shor.md) | Theorems behind the round-4 Shor observations (support law, borrowed magic, noise windows) |
| [theory-coset.md](theory/theory-coset.md) | Coset-representation error in the Gidney–Ekerå Shor circuit |
| [theory-rank.md](theory/theory-rank.md) | Low stabilizer rank in algorithm circuits: branching-rank invariant and an exact engine |
| [stabrank-lower.md](theory/stabrank-lower.md) | Stabilizer-rank lower bounds: state of the art, exact small-n values, a plateau lemma |
| [stabrank5.md](theory/stabrank5.md) | χ(\|T⟩^{⊗5}) = 6 (hence χ(\|T⟩^{⊗6}) = 6): independent computer proof by Galois-pair enumeration and one-qubit lifts |
| [theory-colour.md](theory/theory-colour.md) | Single-auxiliary syndrome extraction for the triangular colour code |
| [transition-theory.md](theory/transition-theory.md) | Why ν_eff ≈ 2.5: the transition as the Clifford MIPT in a noise field |

## Process — [`process/`](process/)

| Notebook | Topic |
|---|---|
| [audit.md](process/audit.md) | Independent audits of every merged branch, and the corrections they forced |
| [literature.md](process/literature.md) | Literature map: where qsim-lab's speed work sits |
| [ARCHIVE.md](process/ARCHIVE.md) | Archived (deleted) branches: what each was, why archived, where the surviving work lives |

## Layout

```
research/
  README.md          this index
  ARCHITECTURE.md    lab-wide design document
  <topic>/<name>.md  notebooks, one per study
  data/<name>/       raw data + the scripts that produced it, one folder per study
```

`data/` is flat by study name (not by topic folder) so a study's data path
never changes when its notebook is re-filed. Large raw files are stored
xz-compressed; see [data/README.md](data/README.md) for how to unpack them.
