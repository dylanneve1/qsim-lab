# Clifford-augmented chain-sweep boundary: a diagnostic (negative result)

Status: experiment (October 2026). **Negative result, stopped at the
diagnostic stage.** A Clifford frame on the chain-sweep boundary removes
entanglement only where the boundary's magic is below saturation. At the
middle cuts of the depth-70 circuit, both half-boundaries are saturated by a
factor of about 2.5. There the boundary is indistinguishable from a
Haar-random state, in magic and in entanglement spectrum alike. The engine of
step 3 was therefore not built.

Circuit: `../chain-sweep/nq70_depth70_checks27_doped.qasm` (tracker issue
228), truncated to CZ-depth `D` exactly as `chain_sweep::truncate` does it.
Background: [../chain-sweep/README.md](../chain-sweep/README.md) (exact
sweep), [../chain-sweep/BOUNDARY_MPS.md](../chain-sweep/BOUNDARY_MPS.md)
(boundary MPS: flat spectrum, truncation fails), and the time-direction CAMPS
note on branch `exp/camps` (`research/camps/README.md`).

## 1. The chain sweep in Clifford+T language

The exact sweep carries a vector `β` over the `m = D/2` bond bits of the
current cut. Every op in its compiled stream is one of the following:

| sweep op | what it is | Clifford+T class |
|---|---|---|
| create the bonds of edge `(i, i+1)` | `CZ = Σ_b P_b ⊗ Z^b`: copy qubit `i`'s Z value into a fresh bit `|0>` (CNOT), or apply `Z^b` controlled by an existing bit | Clifford (CNOT / CZ onto a fresh `|0>`) |
| qubit `i`'s 1q gates between bonds | `h, s, sx, sxdg` on the wire tied to a bit | Clifford |
| `rz(π/4)` on the tied wire | `diag(1, e^{iπ/4})` on that bit | T: `e^{iπ/8} e^{-iπ/8 Z}` |
| wire start `|0>`, wire end `<x_i|` | initial state and output projection | stabilizer prep / Z-basis postselection |
| bond sum `Σ_b` (the unnormalised `[[1,1],[1,−1]]`), then the bit is free | contract the bit with `<+|` (up to √2) after an H | X-basis postselection, then reset to `|0>` |

So one amplitude is a Clifford+T circuit on an `(m+1)`-bit register
(Cliffords, `exp(−iπ/8 Z_k)`, fresh `|0>` bits and **postselected**
single-bit Z/X projections), and the boundary `β` after qubit `i` is that
circuit's postselected state.

**Boundary as `C|φ>`.** Store `β = s · C|φ>`, with `C` a stim tableau on
the `m` bits, `φ` an MPS over the same bits in time order, and `s` a scalar
log-scale. The update rules are the CAMPS rules (Liu & Clark,
arXiv:2412.17209; Fux, Tirrito, Dalmonte, Fazio, PRR 2024), plus
postselection:

1. **Clifford `U`** (bond creation, 1q Cliffords, CNOT/CZ between bits):
   `C ← U C`. `φ` is untouched, at zero cost.
2. **T on bit `k`**: `exp(−iθ Z_k) C|φ> = C exp(−iθ P)|φ>` with
   `P = C† Z_k C` (a Pauli string, read off the tableau). If `P` has X/Y on
   a bit `j` where `φ` is still exactly `|0>` ("free"), OFD applies: a
   Clifford `D` with `D P D† = X_j` turns the rotation into a 1-site gate on
   `φ`, and `C ← C D†`. Otherwise `exp(−iθ P) = cos θ − i sin θ P` is a
   bond-2 MPO over `supp(P)`, followed by truncation.
3. **Postselected projection of bit `k` onto `|w>`**, a stabilizer state with
   `W|w> = |w>` (`W = Z` for `<x_i|` with `x_i = 0`, `−Z` for `x_i = 1`, `X`
   for the bond sum): `<w|_k C|φ>`. Write
   `Π = (1 + W_k)/2`, so `Π C|φ> = C (1 + Q)/2 |φ>` with `Q = C† W_k C`.
   - If `Q` anticommutes with `Z_j` for a free bit `j` (`φ = |0>_j ⊗ φ_rest`),
     the outcome probability is exactly 1/2 and the projection is **Clifford on
     the frame**. Write `Q = X_j ⊗ Q_rest` (after an `S_j` if it is `Y_j`), and
     let `D` be `Q_rest` controlled on `j`. Then `Q = D X_j D†` and
     `D|φ> = |φ>` (control in `|0>`), so
     `(1 + Q)/2 |φ> = D (1 + X_j)/2 |φ> = 2^{-1/2} D H_j |φ>`. Update
     `C ← C D H_j` and `s ← s/√2`; `φ` is untouched. This is the
     Gottesman–Knill measurement rule, restricted to free bits.
   - If `Q` commutes with all free bits' Z (equivalently, `Q` acts only on the
     non-free part of `φ`, up to Z's on free bits, which act as ±1), apply
     the bond-2 MPO `(1 + Q)/2` to `φ` (`s ← s·‖Πφ‖`, renormalise),
     truncate, and then, as in the CAMPS measurement rule (2412.17209, Alg. 4),
     pick a Clifford `D` with `D Q D† = Z_j` for some `j ∈ supp(Q)`. Then
     `D Π φ = |0>_j ⊗ φ'`: since `Πφ` is `Q`-stabilised, `D` disentangles `j`.
     `D` is a CNOT ladder over `supp(Q)` that acts on the MPS (it is not
     free), and is applied with the gates of rule 2's MPO. Set `C ← C D†`.
   - Reset of the consumed bit to `|0>` for reuse: after the projection, bit
     `k` of `β` is in `|w>`. The reset `|0><w|` is a Clifford on a
     known stabilizer state, `C ← R_k C` with `R_k|w> = |0>`.
4. **Amplitude.** At the end of the sweep every bit has been projected and
   reset, so `C|φ>` is `|0^m>` up to a phase that the tableau and `φ`'s
   remaining 1-site factors give, and the amplitude is `s` times that phase.
   (Equivalently, `s·<0^m|C|φ>`. This is a stabilizer–MPS overlap, but the
   rule-3 projections have already done it bit by bit, so the CAMPS readout
   wall `2^E χ²` of the time-direction version does not arise here.)
5. **Disentangling.** After every non-OFD MPO, run the greedy 2-bit Clifford
   disentangler (all 720 unsigned 2-qubit Cliffords on neighbouring bits,
   maximising the Rényi-2 purity of the bond) and absorb the winners into `C`.

The gain over time-direction CAMPS that motivated this route: the register is
only `m ≤ 35` bits, and every qubit's wire is **postselected and discarded**
after its last bond. So, unlike in time-direction CAMPS, magic is not
obliged to accumulate in the register. Postselection onto stabilizer states
could absorb it. Whether it does was measured before building anything.

## 2. Diagnostic: how much of the boundary can a Clifford frame remove?

`diag.py` and `diag2.py` work on the exact boundary vectors from
`chain_sweep dumpcut` (n = 70, CPU f64; `L` = qubits `0..e`, `R` =
qubits `e+1..69`, the `R` vector Walsh–Hadamard transformed, which is a local
Clifford). Measured:

- Raw Schmidt entropy `S1` across the time cuts, and the weight kept at the
  middle cut with `χ = 2^(m/2 − j)`.
- **Stabilizer Rényi-2 entropy** `M2 = −log2(Σ_P <P>⁴ / 2^m)`, estimated by
  sampling Paulis from `Ξ_P = <P>²/2^m` (`a` from the autocorrelation of
  `|β|²`, then `b` from one Walsh–Hadamard transform). This is exact in
  expectation; 120–400 samples were used. `M2` is Clifford-invariant. It is 0
  for stabilizer states, `log2(4/3) = 0.415` per T gate at most for one T on
  a stabilizer state, and `log2(2^m + 3) − 2 ≈ m − 2` for Haar-random
  states.
- A lower bound on the stabilizer **nullity** (`m − log2|A| − log2|B0|`, with `A` the
  X-parts allowed by `|β|`'s autocorrelation and `B0` the Z-only stabilisers).
- **Restarted greedy Clifford disentangling**: sweeps L→R→L of the best of
  the 720 unsigned 2-qubit Cliffords per neighbouring pair, maximising the
  Rényi-2 purity of that pair's cut, up to 8 sweeps. Restarts from random
  layers of single-qubit Cliffords (the greedy result depends on the
  starting local basis); the best middle-cut `S1` is kept.

Validation: random stabilizer states (m = 10) and stabilizer states with 3 T
gates go to `S1 = 0` at every cut. The Clifford-only control circuit
(`nq70_clifford_control.qasm`, all `rz(π/4)` removed) has nullity 0 and
`M2 = 0`, and is disentangled to `S1 = 0` at D = 24, 32 and 40 (D = 40
needed a restart: one greedy run stopped at 2 bits). The `M2` estimator
reproduces 0 (stabilizer), `log2(4/3)` (one T state) and `m − 2` (Haar, m
= 8, 10) within its error bars.

### Middle cut `e = 34` (n = 70; `T` = T gates of that half inside the truncated circuit)

| D | m | side | T | M2 ± se | Haar M2 | raw mid `S1` | after Clifford `S1` | kept j=1 raw → Clifford | kept j=2 raw → Clifford | kept j=3 raw → Clifford |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 12 | Clifford control | 0 | 0.00 | 10.0 | 6.000 | 0.000 | 0.500 → 1.000 | 0.250 → 1.000 | 0.125 → 1.000 |
| 24 | 12 | L (x₀) | 40 | 9.94 ± 0.04 | 10.0 | 5.306 | 5.185 | 0.889 → 0.907 | 0.612 → 0.660 | 0.365 → 0.417 |
| 24 | 12 | L (x₁) | 40 | 9.94 ± 0.05 | 10.0 | 5.278 | 5.105 | 0.896 → 0.916 | 0.621 → 0.684 | 0.372 → 0.448 |
| 24 | 12 | R (x₀) | 17 | 6.42 ± 0.15 | 10.0 | 5.442 | **1.524** | 0.856 → **1.000** | 0.550 → **1.000** | 0.311 → **1.000** |
| 24 | 12 | R (x₁) | 17 | 6.61 ± 0.13 | 10.0 | 5.442 | **1.181** | 0.856 → 1.000 | 0.550 → 1.000 | 0.311 → 1.000 |
| 24 | 12 | Haar-random | – | 10.00 ± 0.05 | 10.0 | 5.281 | 5.213 | 0.896 → 0.903 | 0.619 → 0.649 | 0.372 → 0.406 |
| 32 | 16 | Clifford control | 0 | 0.00 | 14.0 | 8.000 | 0.000 | 0.500 → 1.000 | 0.250 → 1.000 | 0.125 → 1.000 |
| 32 | 16 | L (x₀) | 53 | 13.98 ± 0.04 | 14.0 | 7.299 | 7.195 | 0.890 → 0.906 | 0.614 → 0.656 | 0.368 → 0.412 |
| 32 | 16 | L (x₁) | 53 | 14.03 ± 0.05 | 14.0 | 7.297 | 7.203 | 0.891 → 0.907 | 0.615 → 0.653 | 0.368 → 0.406 |
| 32 | 16 | R (x₀) | 27 | 10.57 ± 0.36 | 14.0 | 7.212 | **3.961** | 0.918 → **1.000** | 0.640 → **0.996** | 0.372 → **0.961** |
| 32 | 16 | R (x₁) | 27 | 10.77 ± 0.24 | 14.0 | 7.212 | **4.063** | 0.918 → 1.000 | 0.640 → 0.994 | 0.372 → 0.954 |
| 32 | 16 | Haar-random | – | 14.01 ± 0.05 | 14.0 | 7.278 | 7.261 | 0.894 → 0.897 | 0.623 → 0.629 | 0.376 → 0.383 |
| 40 | 20 | Clifford control | 0 | 0.00 | 18.0 | 10.000 | 0.000 | 0.500 → 1.000 | 0.250 → 1.000 | 0.125 → 1.000 |
| 40 | 20 | L (x₀) | 74 | 17.98 ± 0.05 | 18.0 | 9.087 | 9.018 | 0.929 → 0.937 | 0.690 → 0.713 | 0.434 → 0.461 |
| 40 | 20 | R (x₀) | 35 | 13.38 ± 0.36 | 18.0 | 9.320 | **6.421** | 0.889 → **0.997** | 0.604 → **0.971** | 0.350 → **0.900** |
| 40 | 20 | R (x₁) | 35 | 13.49 ± 0.30 | 18.0 | 9.320 | **6.488** | 0.889 → 0.996 | 0.604 → 0.964 | 0.350 → 0.888 |
| 48 | 24 | L (x₀) | 83 | 22.00 ± 0.08 | 22.0 | – | – | – | – | – |
| 48 | 24 | R (x₀) | 47 | 18.95 ± 0.38 | 22.0 | – | – | – | – | – |

D = 48: `M2` only. The m = 24 disentangling run was cut short when the Mac went offline.

All doped boundaries have nullity lower bound = m: no Pauli stabilises them
exactly, so an *exact* frame gains nothing. Times: under 1 s (m = 12) to
80 s (m = 20, 3–4 restarts) per boundary on the M1 Pro.

### `M2` against the T count, over cuts (one bitstring per cut)

| D | m (Haar M2) | cut e | side | T in that half | M2 ± se | M2 / T |
|---|---|---|---|---|---|---|
| 24 | 12 (10.0) | 30 | L | 18 | 7.41 ± 0.14 | 0.41 |
| 24 | 12 (10.0) | 30 | R | 39 | 9.86 ± 0.05 | saturated |
| 24 | 12 (10.0) | 34 | R | 17 | 6.42–6.61 | 0.38 |
| 24 | 12 (10.0) | 40 | R | 4 | 1.11 ± 0.05 | 0.28 |
| 24 | 12 (10.0) | 40 | L | 53 | 10.02 ± 0.05 | saturated |
| 32 | 16 (14.0) | 30 | L | 23 | 9.62 ± 0.18 | 0.42 |
| 32 | 16 (14.0) | 30 | R | 57 | 14.03 ± 0.05 | saturated |
| 32 | 16 (14.0) | 34 | R | 27 | 10.57–10.77 | 0.40 |
| 32 | 16 (14.0) | 40 | R | 7 | 2.47 ± 0.08 | 0.35 |
| 32 | 16 (14.0) | 40 | L | 73 | 14.07 ± 0.05 | saturated |
| 40 | 20 (18.0) | 25 | L | 7 | 2.10 ± 0.08 | 0.30 |
| 40 | 20 (18.0) | 25 | R | 102 | 17.86 ± 0.06 | saturated |
| 40 | 20 (18.0) | 30 | L | 28 | 11.38 ± 0.26 | 0.41 |
| 40 | 20 (18.0) | 30 | R | 81 | 18.08 ± 0.05 | saturated |
| 40 | 20 (18.0) | 34 | R | 35 | 13.38–13.49 | 0.38 |
| 40 | 20 (18.0) | 40 | L | 99 | 18.05 ± 0.05 | saturated |
| 40 | 20 (18.0) | 40 | R | 10 | 3.24 ± 0.09 | 0.32 |
| 40 | 20 (18.0) | 45 | L | 103 | 18.02 ± 0.05 | saturated |
| 40 | 20 (18.0) | 45 | R | 6 | 1.56 ± 0.06 | 0.26 |
| 48 | 24 (22.0) | 34 | L | 83 | 22.00 ± 0.08 | saturated |
| 48 | 24 (22.0) | 34 | R | 47 | 18.95 ± 0.38 | 0.40 |

(T counts by `tcount.py`: a T gate on qubit `q` is in the truncated circuit
if `q` has had fewer than `D` CZs before it.)

### What the numbers say

1. **The postselections do not absorb magic.** The boundary's `M2` is
   0.26–0.42 per T gate of its half, against 0.415 for an isolated T gate.
   It is near-additive in the T count, up to the Haar ceiling `m − 2`, and
   that ceiling is reached as soon as the half holds more than about
   `2.4 m` T gates. The hope behind this route (that projecting each
   site's wire out would keep the register's magic small) is false.
2. **Below saturation, the hypothesis is right.** The `R` halves at
   D = 24–40 hold only 17–35 T gates, and their `M2` is 3.5–4.6 bits below
   the Haar value. There a Clifford frame removes 3–4 bits of middle-cut
   entropy. At D = 40 it raises the weight kept at `χ = χ_ex/4` from 0.60
   to 0.97, and at `χ_ex/8` from 0.35 to 0.90. For a flat-spectrum
   boundary that is worth about 2 bits of χ.
3. **At saturation, the boundary is Haar-like and a frame does nothing.**
   The `L` halves (40–74 T, `M2` within 0.05 of `m − 2`) match a Haar-random
   vector of the same size to the third digit: raw middle `S1` 5.31 / 7.30
   / 9.09 against Haar 5.28 / 7.28 / (Page 9.28), kept weights 0.889 /
   0.612 against 0.896 / 0.619. D = 40's `L` is 0.2 bit below Page, and `M2` is still at the ceiling. D = 48's `L` (83 T) also has `M2` = 22.00 ± 0.08, exactly the Haar value. Clifford disentangling reduces them by
   0.07–0.12 bit, the same as it does for a Haar vector (0.02–0.07). The
   kept weight at `χ_ex/2` improves by ≤ 0.02, i.e. ≤ 0.03 bit of χ. This
   also corrects the reading in BOUNDARY_MPS.md that the near-flat spectrum
   is *Clifford* (stabilizer) entanglement. At the middle cut it is Page
   (Haar) flatness. Only the Clifford-only control and the low-T cuts have
   exactly flat stabilizer spectra.

## 3. Consequence for D = 70

The register is `m = 35` bits at every cut (Haar ceiling 33). T gates per
half at D = 70 (`tcount.py`):

| cut e | 10 | 15 | 20 | 25 | 30 | 34 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T in L (q ≤ e) | 36 | 44 | 54 | 73 | 121 | 202 | 257 | 275 | 311 | 345 | 372 |
| T in R (q > e) | 372 | 364 | 354 | 335 | 287 | 206 | 151 | 133 | 97 | 63 | 36 |
| predicted M2(L), cap 33 | ~14 | ~17 | ~21 | ~29 | sat | sat | sat | sat | sat | sat | sat |
| predicted M2(R), cap 33 | sat | sat | sat | sat | sat | sat | sat | sat | sat | ~25 | ~14 |

(Prediction `M2 ≈ 0.40 × T`, capped at 33.) The saturation threshold is about
`2.4 m ≈ 84` T gates. Every cut between `e ≈ 27` and `e ≈ 53` is saturated
from **both** sides: the middle cut by a factor of 2.4 (202 and 206 T
against ~84). Any chain sweep, one-way or meet-in-the-middle, has to hold
a boundary at such a cut. There the boundary is a Haar-like 35-bit vector,
the Clifford frame saves ≲ 0.05 bit of χ, and the boundary-MPS numbers
(`χ ≈ 2^16.2` for f = 0.1, ~320 GiB) stand unchanged. The low-magic regime
where the frame helps (≤ ~25 T in a half) only exists at the ends of the
chain, where the boundary is cheap anyway.

So the clifford-augmented boundary was **not built** (brief step 3): its exact
rules are in section 1, but the diagnostic shows it cannot change the D = 70
cost. A per-cut extrapolation from the numbers above:

- f = 0.1 at D = 70 still needs `χ ≈ 2^16.2` at the middle cuts (Haar-like
  boundary, frame worth ≲ 0.05 bit). That is ~320 GiB, against 16 GB.
- The sampling budget is unchanged: `N ≈ 9/(f − 0.044)² ≈ 2900` samples at
  f = 0.1.

## Files

- `diag.py`: cut entropies, nullity bound, greedy 720-Clifford
  disentangler (Rényi-2 purity from a 4⁴ Gram tensor, contracted on the
  smaller side of the cut).
- `diag2.py`: `M2` by Pauli sampling, restarted disentangler, kept weights.
- `m2only.py`, `tcount.py`, `haar_ref.py`: `M2` per cut, T counts per half,
  Haar calibration vectors.
- `test_diag.py`, `test2.py`: checks on stabilizer, T-doped stabilizer, Haar
  and T states.
- `nq70_clifford_control.qasm`: the circuit with every `rz(pi/4)` removed.
- `logs/summary.txt`: the summary lines of all runs above.

## Reproduce

```text
cargo build --release --example chain_sweep
B=target/release/examples/chain_sweep
$B dumpcut --n 70 --d 32 --e 34 --k 2 --seed 3 --out d32_e34
$B dumpcut --qasm research/clifford-boundary/nq70_clifford_control.qasm --n 70 --d 32 --e 34 --k 1 --seed 3 --out cliff_d32_e34
python3 research/clifford-boundary/diag2.py --restarts 4 cliff_d32_e34_0_L.bin d32_e34_0_L.bin d32_e34_0_R.bin
python3 research/clifford-boundary/m2only.py 300 d32_e34_0_L.bin
python3 research/clifford-boundary/tcount.py research/chain-sweep/nq70_depth70_checks27_doped.qasm
```
