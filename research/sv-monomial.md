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

**Key Takeaways:**
1. Google qsim's 1.9× advantage over qsim-lab on brickwork comes squarely from 2-qubit dense fusion: `fuse=2` eliminates 4 out of 5 memory passes on each interacting pair $(q, q+1)$.
2. Moving to $k=3$ and $k=4$ in qsim shows diminishing/negative returns on 4 cores ($0.173 \to 0.185 \to 0.275$), because $16\times 16$ complex matrix multiplies increase compute cost without sufficient additional memory reuse.
3. Therefore, implementing $k \in \{2, 3, 4\}$ dense fusion in qsim-lab with SoA kernels should close the gap with qsim.
