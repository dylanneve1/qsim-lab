# State-Vector Research: k-Qubit Dense Fusion and Monomial-Segment Fusion

**Author:** Sub-agent `agt_f4c5c1f2` (`qsim-sv-monomial`), branch `exp/sv-monomial`  
**Host machine:** Shared 4-vCPU VM (AMD EPYC-Rome, AVX2, 7.7 GB RAM).  
**Rules:** All timed runs serialized via `/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh` (flock). Portable release builds (`debug=0`), `CARGO_BUILD_JOBS=2`.

---

## 0. Prior-Art Analysis and Novelty Assessment

### 0.1 Dense k-Qubit Gate Fusion in State-Vector Simulators
Dense gate fusion is standard practice in industrial state-vector engines:
- **Google qsim** (`qsimcirq.QSimOptions(max_fused_gate_size=k)`): Groups gates acting on small subsets of qubits into fused unitaries of up to $k \le 4$ qubits (matrix sizes $4\times 4$, $8\times 8$, $16\times 16$). This dramatically raises arithmetic intensity (flops per memory byte) and allows AVX2/AVX-512 FMA vectorization in register. On our testbed (EPYC-Rome, 4 vCPUs, brickwork 22 qubits), qsim achieves:
  - `fuse=2`: 0.1730 s
  - `fuse=3`: 0.1855 s
  - `fuse=4`: 0.2756 s
- **Qiskit Aer** (`AerSimulator(fusion_enable=True)`): Greedily fuses adjacent 1q and 2q gates into composite multi-qubit matrices. At brickwork 22 qubits:
  - `fusion=True`: 1.637 s
  - `fusion=False`: 2.786 s (1.7× speedup from fusion)
- **qsim-lab (main ceaef67)**: Features 1-qubit fusion (`fuse_1q`, multiplying runs of 1q gates on the same wire into a $2\times 2$ matrix) and diagonal phase scheduling (`schedule_diag`), but treats all 2-qubit interactions (e.g. CNOTs) as separate 1-qubit controlled passes. On brickwork 22 qubits, main blocked executor runs in **0.3304 s** — beating Aer by 5×, but trailing qsim's 2-qubit fused kernel (0.1730 s) by 1.9× because each CNOT incurs a separate streaming pass.

### 0.2 Monomial-Segment Fusion: Affine Permutations and Phase Polynomials
Monomial gates (gates with exactly one non-zero entry per row and column) include $X, Y, Z, S, S^\dagger, T, T^\dagger$, CNOT, CZ, SWAP, Toffoli (CCX), and arbitrary diagonal rotations $P(\theta)$.
- **Circuit Synthesis & Optimization Literature:**
  - Amy et al. (2014, 2018), *Polynomial-time T-depth Optimization*: Shows that Clifford+$T$ subcircuits with CNOT and $R_Z$ gates synthesize to phase polynomials $p(x) = \sum a_i x_i + \sum a_{ij} x_i x_j + \dots \pmod{2\pi}$ over affine Boolean variables.
  - Meuli et al. (2020), *Phase Polynomial Synthesis*: Optimization of quantum circuits using Reed-Muller decoding and matroid theory.
  - De Beaudrap et al. (ZX-calculus): Simplification of phase gadgets.
- **Simulation Literature:**
  - *PhasePoly.jl* (Deger et al.): Computes Pauli expectation values analytically via quadratic Gauss sums in $O(n^3)$ without ever forming a state vector.
  - *Stim* (Gidney 2021): Simulates stabilizer circuits via Pauli frame tracking and tableau operations in $O(1)$ to $O(n)$ time per Clifford gate.
  - *SymPhase / Rotation Frame*: In Clifford+$T$ simulation, tracks affine transformations symbolically until non-Clifford rotations force state-vector materialization.
- **Novelty Assessment for Dense State-Vector Kernels:**
  - While phase polynomials and affine maps are extensively studied for *circuit optimization* and *analytical expectation value computation*, their application as an **execution kernel for dense state-vector evolution** is virtually absent from the simulation literature.
  - Existing dense state-vector simulators (Aer, qsim, Qulacs, cuQuantum) treat CNOT, SWAP, and CCX as general sparse or permutation gates and apply them gate-by-gate or fuse them into dense $2^k \times 2^k$ matrices.
  - Compiling maximal monomial segments into an affine bijection $x \mapsto A x \oplus b$ over $\mathbb{F}_2$ combined with a streaming phase polynomial $\phi(x)$ in a single memory pass is structurally novel.
  - **The Engineering Question & Kill Criterion:**
    In a cache-blocked executor where amplitudes are already streamed in L2-sized chunks ($2^L$ amplitudes, e.g. 256 KiB), does the reduction in gate passes outweigh the index-arithmetic / gather overhead? If a micro-prototype does not beat main's blocked executor by $\ge 1.3\times$ on monomial-heavy circuits (Cuccaro adder, Grover oracle, CNOT+$T$), it is recorded as a negative result and killed.

---

## 1. Baseline Head-to-Head (Random Brickwork, n=22, f32)

Measured via `bench.sh` lock on shared VPS (load 2.9–3.2):

| Engine | Configuration | Min Wall (s) | Gates / Ops | Relative to Main Blocked |
|---|---|---|---|---|
| **qsim-lab (main)** | Base (gate-by-gate) | 1.3971 s | 1090 | 0.24× (4.2× slower) |
| **qsim-lab (main)** | Blocked executor (1q fusion) | **0.3304 s** | 1090 | **1.00× (baseline)** |
| **Google qsim** | `fuse=2` (2q dense fusion) | **0.1730 s** | 1090 | **1.91× faster** |
| **Google qsim** | `fuse=3` (3q dense fusion) | 0.1855 s | 1090 | 1.78× faster |
| **Google qsim** | `fuse=4` (4q dense fusion) | 0.2756 s | 1090 | 1.20× faster |
| **Qiskit Aer** | `fusion=True` | 1.6373 s | 1090 | 0.20× (5.0× slower) |
| **Qiskit Aer** | `fusion=False` | 2.7861 s | 1090 | 0.12× (8.4× slower) |

**Key Takeaways (original, now corrected: see §2):**
1. ~~qsim's 1.9x advantage comes squarely from 2-qubit dense fusion.~~ **Not supported** by the data in §2.
2. qsim at k=3,4 is slower than k=2 on 4 cores (0.173 -> 0.185 -> 0.275 s), which only says larger fused matrices cost more compute than they save.
3. ~~Implementing k in {2,3,4} dense fusion in qsim-lab should close the gap.~~ Not supported; the dense-fusion WIP (wip/fusion-avx2) was slower than main at k=2 and equal at k=3.

---

## 2. Correction (branch exp/simd): SIMD, not fusion

All runs: brickwork depth 20 (Ry, Rz on every qubit, then CNOT brick), f32, 4 vCPU EPYC-Rome shared VM, every timed run through `bench.sh` (flock). **The VM load average was 9-15 during these runs (other agents compiling/benchmarking), so wall times are noisy by up to ~1.5x between repeats; CPU time (user+sys of our process) is the stable column.** Raw table: `research/data/simd/brick_timings_loaded.txt`.

### 2.1 Flop count argument
Per layer, per amplitude: qsim `fuse=2` does one dense 4x4 complex multiply per pair (~30 flops/amp for the pair, CNOT folded in), 12 pairs/layer = ~360 flops/amp/layer. We do one fused complex 2x2 per qubit (~14 flops/amp), 24 qubits = ~336 flops/amp/layer, and CNOT is a free permutation. Dense 2q fusion saves essentially **no** flops on brickwork. The hypothesis in §0/§1 is therefore not about arithmetic.

### 2.2 Experiments (n=24, f32, min of 3 unless noted)
| build | min wall (s) | CPU s (ours) | note |
|---|---|---|---|
| ours, default (portable SSE2) | 1.88-2.0 | 5.1-5.6 | load 8-11 |
| ours, `-C target-cpu=native` | 1.59-1.67 | 4.1-4.3 | AVX2 autovectorised, **no FMA** (rustc never contracts a*b+c) |
| ours, exp/simd runtime AVX2+FMA dispatch | 1.80 (1.22 in one low-load window) | 4.5-4.7 | default build, no flags |
| ours, WIP dense fusion k=2 (wip/fusion-avx2, load ~3) | 3.74 vs 2.43 for its own k=1 | 7.8 vs 4.1 | **slower**: fusion lost |
| ours, WIP dense fusion k=3 | 2.64 | 6.0 | no gain |
| Google qsim 0.22.1, fuse=2, 4 threads | 1.07-1.25 (best windows) | n/a | includes Python front end |

n=26: default 6.7-8.6 s / 22.1 CPU-s; native 6.1-7.1 s / 17.5 CPU-s; dispatched 8.1-9.1 s / 19.4 CPU-s; qsim 5.7-6.9 s. (Wall at n=26 is dominated by load noise; qsim and native were interleaved but not simultaneous.)

Verdict, from the CPU-time column: **compiling the same portable kernels for AVX2 cuts our CPU time by ~20% (n=24) and the exp/simd dispatch by ~15-25%. Dense 2q fusion (WIP) cuts nothing.** So the claim "the gap is dense 2q fusion" is refuted; the claim "the gap is (all) SIMD" is only **partly** supported: AVX2/FMA takes our best n=24 wall from ~1.9 to ~1.6-1.8 s against qsim's ~1.1-1.25 s, i.e. the gap goes from ~1.7x to ~1.4x, not to 1.0x. I did not close it.

### 2.3 What is left (hypothesis, untested)
Profile (`QSIM_PROF=1`, thread-summed, n=24): u1-complex 5.8 s -> 2.9 s with FMA dispatch, small-bit (target bit < 2) 1.85 -> 1.42 s, load+store (AoS<->SoA) 1.5 -> 1.2 s. The 1q complex kernel now runs at ~1.1 cycles/amplitude/gate, close to what streaming 256 KiB L2-resident chunks through L1 once per gate allows (~0.8 cycle/amp for L2 bandwidth). qsim's fused 2q kernel touches each amplitude once per two qubits' worth of work, which halves the L1/L2 traffic. The likely next step is not dense fusion but two-level blocking: sub-block runs of low-qubit gates inside the L2 chunk so they stay in L1. Smaller L2 blocks (32-128 KiB) are worse (more DRAM stages), so it needs a nested plan, not a smaller block.

### 2.4 What exp/simd implements
`BlockConfig::simd` (default true). The chunk kernels (`u1_*`, `diag_kernel`, group-of-8 and small-bit paths) are generic over a `const F: bool` that selects `mul_add` (FMA) vs `a*b+c`. `run_ops` is dispatched once per run via `is_x86_feature_detected!("avx2") && ("fma")` to `run_ops_avx2`, a `#[target_feature(enable = "avx2,fma")]` wrapper over the same `#[inline(always)]` code; otherwise the portable path (F=false) runs. The only `unsafe` is that one call, guarded by the detection. FMA changes rounding only at the last bit: dispatched vs portable vs gate-by-gate agree to <1e-12 (f64) and <1e-5 (f32) in `tests/engines/simd.rs` (proptest over random universal circuits, five block/slot configs, f32 and f64) and the existing `tests/engines/blocked.rs` configs now run the dispatched path too. Measured max |dA| base vs blocked at n=18: 1.4e-8 (f32), 1.6e-17 (f64).
