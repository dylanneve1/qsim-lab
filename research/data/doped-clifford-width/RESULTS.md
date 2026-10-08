# Exact Boundary Analysis for IBM's 70-Qubit Doped-Clifford Circuit (Issue 228)

**Agent:** `agt_df433371` ("doped-width-search")  
**Date:** 2026-10-07  
**Target:** Tracker Issue 228 (`doped_random_graph_sampling_nq70_depth70_checks27`)

---

## Executive Summary

1. **H1 (Stabilizer Subspace / Magic Width vs Schmidt Width): DEAD.**
   - We implemented the exact $O(m 4^m)$ fast Walsh-Hadamard Pauli characteristic transform to measure the exact stabilizer group size ($2^k$, independent generators $k$), stabilizer nullity ($\nu = w - k$), and stabilizer Rényi-2 entropy ($M_2$) on exact boundary states from `qsim-lab`'s `chain_sweep` engine.
   - Tail windows ($D = 16, 20, 24$, corresponding to bond widths $w = 8, 10, 12$) were measured across chain cuts $e \in \{10, 20, 25, 30, 34, 40, 45, 50, 60\}$.
   - In the pure Clifford control (all T gates dropped), the boundary state is 100% a stabilizer state: $k = w$, nullity $\nu = 0$, $M_2 = 0.000$, $\max_{P \neq I} |\langle \psi|P|\psi\rangle| = 1.0000$.
   - In the doped circuit, across all cuts in the bulk ($e \ge 20$), **$k = 0$**, **nullity $\nu = w$**, and **$M_2 = w - 2$** (saturating the Haar ceiling).
   - **Conclusion:** The boundary's flat Schmidt spectrum is Page/Haar flatness, NOT stabilizer entanglement. No Clifford frame can reduce the boundary dimension; the true dimension is $2^{35}$, with zero stabilizer reduction ($k = 0$).

2. **H2 (T-Gate Census and Light Cones):**
   - **Total T gates:** 468 `rz(pi/4)` gates across 70 qubits.
   - **T-free regions:** In early/middle layers (CZ layers 0..49), qubits 0..19 have **0** T gates, and qubits 50..69 have only 7 T gates. Early T gates are strictly concentrated on the middle core ($q \in [20, 49]$, especially $q \in [28, 41]$).
   - **Late concentration:** 249 of the 468 T gates (**53.2%**) sit in the last 7 CZ layers (layers 64..70). In layers 63..70, there are 267 T gates (57.1%).
   - **Light cones at the middle cut ($e = 34$):** There are 234 T gates on the left ($q \le 34$) and 234 on the right ($q > 34$). The backward causal light cone from the 35 CZ bonds on edge 34 encompasses **122 T gates**, far exceeding the Haar saturation threshold ($\sim 84$ T gates for $m=35$).

3. **Status of Slicing and Width Reduction:**
   - **Treewidth lower bound:** The circuit interaction graph is a $70 \times 35$ grid. Its graph treewidth is $\min(70, 35) = 35$. No contraction order can achieve width $< 35$ without slicing.
   - **Chain sweep:** Width 35 at all 69 intermediate cuts. Slicing down to width 33 requires slicing $\ge 2$ bonds on each of the 69 cuts ($2^{138}$ slices).
   - **General slicing (cotengra/kahypar):** Unsliced search yields width 47–71 ($2^{58.1}-2^{98}$ cost). Slicing to width 31 costs $2^{101.5}$ FLOPs ($2^{61}$ slices). Slicing overhead is $\ge 2^{20}-2^{35}$ per bit of width reduction.
   - **ZX-calculus:** `full_reduce` shrinks spider count to 516, but introduces 13,188 Hadamard edges; treewidth explodes to $\ge 104$.

4. **Recommendation for 16 GB:**
   - Exact in-RAM simulation at width $\le 33$ is impossible.
   - The only viable path is width-35 chain sweep using 4-bit block floating point storage (`int4:b16:h`, 4.5 bits/component = 36 GiB total register) with the 71-pass big-buffer schedule streaming the out-of-RAM portion (~24 GiB) to SSD (or fast external NVMe), achieving fidelity $F \approx 0.53$ (well above the 0.044 threshold).

---

## H1: Exact Measurements of Magic Width vs Schmidt Width

All measurements were performed on exact boundary states computed by `qsim-lab::chain_sweep::cut_tensors_cpu` on tail windows of depth $D$ (layers $70-D .. 70$) on an open 70-qubit chain.
Transform: in-place Fast Walsh-Hadamard Transform on $f_u(y) = \psi(y) \psi^*(y \oplus u)$, evaluating all $4^w$ Pauli expectations $c(u, s) = \langle \psi | P_{u, s} | \psi \rangle$.
Stabilizer generators $k = \log_2(\#\{P : |\langle P \rangle| = 1\})$.
Stabilizer nullity $\nu = w - k$.
Stabilizer Rényi-2 entropy $M_2 = -\log_2 \left( \frac{1}{2^w} \sum_{P} c(P)^4 \right)$.

### Tail Window $D = 16$ (Bond Width $w = 8$, Hilbert Dim 256)
*Haar-random $M_2$ ceiling: $w - 2 = 6.000$*

| Cut $e$ | Width $w$ | Circuit | $T_L$ | $T_R$ | $k$ (generators) | Nullity $\nu = w - k$ | $M_2$ (bits) | Max Non-Identity $|\langle P \rangle|$ | Time |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 10 | 8 | Doped | 46 | 278 | **0** | **8** | 5.984 | 0.3536 | 0.00s |
| 20 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 20 | 8 | Doped | 73 | 251 | **0** | **8** | 6.025 | 0.2779 | 0.00s |
| 25 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 25 | 8 | Doped | 92 | 232 | **0** | **8** | 6.023 | 0.2663 | 0.00s |
| 30 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 30 | 8 | Doped | 118 | 206 | **0** | **8** | 5.938 | 0.3893 | 0.00s |
| 34 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 34 | 8 | Doped | 140 | 184 | **0** | **8** | 6.013 | 0.2523 | 0.00s |
| 40 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 40 | 8 | Doped | 166 | 158 | **0** | **8** | 6.005 | 0.2932 | 0.00s |
| 50 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 50 | 8 | Doped | 222 | 102 | **0** | **8** | 6.030 | 0.2619 | 0.00s |
| 60 | 8 | Clifford | 0 | 0 | **8** | **0** | 0.000 | 1.0000 | 0.00s |
| 60 | 8 | Doped | 286 | 38 | **0** | **8** | 6.020 | 0.2643 | 0.00s |

---

### Tail Window $D = 20$ (Bond Width $w = 10$, Hilbert Dim 1024)
*Haar-random $M_2$ ceiling: $w - 2 = 8.000$*

| Cut $e$ | Width $w$ | Circuit | $T_L$ | $T_R$ | $k$ (generators) | Nullity $\nu = w - k$ | $M_2$ (bits) | Max Non-Identity $|\langle P \rangle|$ | Time |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.02s |
| 10 | 10 | Doped | 46 | 291 | **0** | **10** | 7.848 | 0.3088 | 0.02s |
| 20 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.02s |
| 20 | 10 | Doped | 73 | 264 | **0** | **10** | 8.005 | 0.1624 | 0.02s |
| 25 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.01s |
| 25 | 10 | Doped | 93 | 244 | **0** | **10** | 8.009 | 0.1497 | 0.02s |
| 30 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.01s |
| 30 | 10 | Doped | 122 | 215 | **0** | **10** | 7.945 | 0.3504 | 0.02s |
| 34 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.02s |
| 34 | 10 | Doped | 149 | 188 | **0** | **10** | 7.879 | 0.5260 | 0.01s |
| 40 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.02s |
| 40 | 10 | Doped | 176 | 161 | **0** | **10** | 7.992 | 0.2885 | 0.01s |
| 50 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.02s |
| 50 | 10 | Doped | 232 | 105 | **0** | **10** | 8.007 | 0.1497 | 0.01s |
| 60 | 10 | Clifford | 0 | 0 | **10** | **0** | 0.000 | 1.0000 | 0.02s |
| 60 | 10 | Doped | 297 | 40 | **0** | **10** | 7.985 | 0.2629 | 0.02s |

---

### Tail Window $D = 24$ (Bond Width $w = 12$, Hilbert Dim 4096)
*Haar-random $M_2$ ceiling: $w - 2 = 10.000$*

| Cut $e$ | Width $w$ | Circuit | $T_L$ | $T_R$ | $k$ (generators) | Nullity $\nu = w - k$ | $M_2$ (bits) | Max Non-Identity $|\langle P \rangle|$ | Time |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.26s |
| 10 | 12 | Doped | 46 | 300 | **1** | **11** | 8.520 | 1.0000 | 0.28s |
| 20 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.24s |
| 20 | 12 | Doped | 73 | 273 | **0** | **12** | 9.978 | 0.1819 | 0.24s |
| 25 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.26s |
| 25 | 12 | Doped | 93 | 253 | **0** | **12** | 9.999 | 0.1038 | 0.26s |
| 30 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.27s |
| 30 | 12 | Doped | 122 | 224 | **0** | **12** | 9.935 | 0.3695 | 0.27s |
| 34 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.26s |
| 34 | 12 | Doped | 154 | 192 | **0** | **12** | 9.895 | 0.4928 | 0.29s |
| 40 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.25s |
| 40 | 12 | Doped | 185 | 161 | **0** | **12** | 9.993 | 0.2502 | 0.35s |
| 45 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.25s |
| 45 | 12 | Doped | 202 | 144 | **0** | **12** | 9.999 | 0.1142 | 0.24s |
| 50 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.30s |
| 50 | 12 | Doped | 241 | 105 | **0** | **12** | 10.001 | 0.0832 | 0.26s |
| 60 | 12 | Clifford | 0 | 0 | **12** | **0** | 0.000 | 1.0000 | 0.26s |
| 60 | 12 | Doped | 306 | 40 | **0** | **12** | 9.989 | 0.2298 | 0.26s |

---

## H2: T-Gate Census and Light Cones

### 1. Depth Distribution
| CZ Layer Range | T Gate Count | % of Total | Active Qubits | Active Qubit Span |
|:---|:---:|:---:|:---:|:---|
| Layers 0..10 | 27 | 5.8% | 15 | q[20..69] (core concentrated) |
| Layers 11..20 | 25 | 5.3% | 13 | q[22..40] |
| Layers 21..30 | 24 | 5.1% | 12 | q[29..56] |
| Layers 31..40 | 32 | 6.8% | 14 | q[23..65] |
| Layers 41..50 | 27 | 5.8% | 14 | q[30..68] |
| Layers 51..60 | 50 | 10.7% | 31 | q[02..67] |
| Layers 61..63 | 34 | 7.3% | 28 | q[04..68] |
| **Layers 64..70** | **249** | **53.2%** | **70** | **All 70 qubits** |
| **Total** | **468** | **100%** | **70** | **q[00..69]** |

### 2. Spatial Distribution
| Qubit Decile | Total T Gates | Early (Layers 0..49) | Late (Layers 50..70) | Note |
|:---|:---:|:---:|:---:|:---|
| q[00..09] | 42 | **0** | 42 | **Completely T-free in early layers** |
| q[10..19] | 27 | **0** | 27 | **Completely T-free in early layers** |
| q[20..29] | 64 | 20 | 44 | Active from layer 2 onward |
| q[30..39] | 154 | **94** | 60 | **Dense core of early magic** |
| q[40..49] | 57 | 10 | 47 | Moderate early activity |
| q[50..59] | 76 | 4 | 72 | Mostly late activity |
| q[60..69] | 48 | 3 | 45 | Mostly late activity |

### 3. Light Cone per Chain Cut
For each chain cut $e$ (cutting edge between qubit $e$ and $e+1$), we trace the backward causal cone from the 35 CZ bonds on edge $e$:

| Chain Cut $e$ | $T_L$ ($q \le e$) | $T_R$ ($q > e$) | T in Backward Causal Cone | T in Left (Layers 0..59) | T in Left (Layers 60..70) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 23 | 445 | 13 | 0 | 23 |
| 10 | 46 | 422 | 20 | 0 | 46 |
| 15 | 59 | 409 | 6 | 0 | 59 |
| 20 | 74 | 394 | 10 | 1 | 73 |
| 25 | 98 | 370 | 17 | 7 | 91 |
| 30 | 151 | 317 | 54 | 39 | 112 |
| **34 (Middle)** | **234** | **234** | **122** | **104** | **130** |
| 40 | 293 | 175 | 152 | 141 | 152 |
| 45 | 315 | 153 | 146 | 146 | 169 |
| 50 | 356 | 112 | 164 | 153 | 203 |
| 55 | 394 | 74 | 155 | 161 | 233 |
| 60 | 425 | 43 | 153 | 173 | 252 |
| 65 | 450 | 18 | 134 | 178 | 272 |

**Key Takeaways from H2:**
- At the middle cut $e = 34$, the backward light cone holds 122 T gates. The magic saturation threshold for a 35-bit register is $\approx 2.4 \times 35 \approx 84$ T gates.
- Because $122 \gg 84$, the boundary state is saturated into the Haar regime from both space and time causal horizons.
- The T-free nature of qubits 0..19 in layers 0..49 explains why head-truncated circuits stay stabilizer-like up to qubit ~20, but in tail windows and at full depth $D=70$, the entire boundary is immersed in scrambled magic.
