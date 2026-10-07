# Approximate CAMPS on IBM's doped-Clifford circuit (nq70, depth 70)

Status: experiment (October 2026). Negative result. The Clifford frame removes
almost all of the *graph-state* entanglement from the MPS, and it beats a plain
MPS by tens of orders of magnitude in fidelity at equal bond dimension. It does
not survive the full circuit, though, and reading the state out costs more than
the exact chain sweep (`../chain-sweep/`).

Circuit: `../chain-sweep/nq70_depth70_checks27_doped.qasm` (tracker issue 228),
truncated to CZ-depth `D` exactly as `chain_sweep::truncate` does it.

## Method (`camps.py`)

- `psi = C |phi>`: `C` is a Clifford kept as a stim tableau of `C^-1`, and
  `phi` is an MPS (bond `chi`, SVD/eigh truncation, mixed-canonical).
  Clifford gates only update the tableau.
- `rz(pi/4)` on `q` becomes `exp(-i pi/8 P)` on `phi`, with `P = C^dag Z_q C`.
  - **OFD** (optimization-free disentangling, Liu & Clark arXiv:2412.17209). If
    `P` has X/Y on a *free* site `j` (`phi_j` still exactly `|0>`), a Clifford
    `D` (S on `j`, then CNOT/CZ controlled on `j`, plus local Cliffords on
    `phi`) maps `P -> X_j`. Since `D phi` only changes `phi` locally, the
    rotation is single-site and adds no entanglement. At most `n = 70` T gates
    can be handled this way.
  - Otherwise: a bond-2 MPO over the support of `P`, then truncation. The fidelity
    estimate is `prod(1 - discarded weight)`.
  - Optional greedy disentangler (`camps`; `camps-noDis` = OFD only). After each
    MPO, sweep the bonds in its support right and then left. On each bond, pick
    the best of all 720 unsigned 2-qubit Cliffords by Renyi-2 purity: all 720
    are scored from one 16-block Gram tensor, so the cost is one matmul. Absorb
    the winner into `C`.
- Readout `<x|C|phi> = <s|Q_x|phi>`: `s = C^dag|0>` is put in stim's
  `graph_state` form (local Cliffords on `CZ_graph |+>^n`), and
  `Q_x = C^dag X^x C`. The contraction sweeps the chain, keeping
  (MPS bond) x (pending CZ-phase pattern of the frontier). Its cost is
  `~ n chi^2 2^E` and its memory `2^E chi`, where `E` is the cut rank of `s`,
  i.e. the entanglement of the stabiliser state `C^dag|0>`.
- Plain MPS baseline (`mps`): TEBD of the same gate list with the same truncation.
- Reference: `examples/camps_ref.rs`, exact chain-sweep amplitudes for a list
  of bitstrings (`--n --d --xs`, f64).

## Validation

- `camps_ref` agrees with a dense state vector (n=20, D=40), phase included.
  It uses the same truncation and bit order (char `i` = qubit `i`).
- CAMPS is exact at full bond dimension. On qubits 24..37 (n=14) of the full
  depth-70 circuit (185 T), `chi = 128 = 2^7` gives `F = 1.000000` with and
  without disentangler, versus the state vector. At `chi = 64` the true fidelity
  matches the estimate: camps 0.00506 vs est 0.00515, mps 0.1397 vs 0.1368.
  (On this full-depth 14-qubit window the frame does *not* help; see below.)
- Readout vs dense `C|phi>` (n=12): ratio constant to 1e-15, a global phase.
- **n = 70, true fidelity vs chain-sweep amplitudes** (uniform random
  bitstrings; `F_unb = |2^n mean(a* b)|^2`, `F_ratio = |<a,b>|^2/(|a|^2|b|^2)`,
  bootstrap SE):

| D | chi | F_est (truncation) | F_true unbiased | F_true ratio | M | readout E | t/amplitude |
|---|-----|-------------------:|----------------:|-------------:|--:|----------:|------------:|
| 20 | 16 | 0.291 | 0.33 +- 0.05 | 0.315 +- 0.025 | 400 | 14 | 2.1 s |
| 20 | 64 | 0.775 | 0.78 +- 0.08 | 0.801 +- 0.012 | 400 | 13 | 3.9 s |
| 30 | 16 | 0.040 | 0.06 +- 0.06 | 0.07 +- 0.05 | 64 | 21 | 106 s |

  The truncation estimate tracks the true fidelity, and is slightly
  conservative. At D=30, `chi=32`, one amplitude took 214 s (2^21 frontier
  keys). The exact chain sweep needs ~0.05 s at D=30.

## Fidelity vs chi, n = 70 (truncation estimate, -ln F; [seconds / peak RSS MB], 1 thread, loaded 4-vCPU VPS)

```
-lnF (truncation estimate)  [time s / peak RSS MB]
mode         D   T  mpoT | chi=16   | chi=32   | chi=64   | chi=128  | chi=256 
camps        20  52   18 |   1.23 [6/45] |   0.67 [22/47] |   0.26 [13/53] |   0.04 [46/74] |   0.00 [17/76]
camps-noDis  20  52   18 |   1.27 [2/44] |   0.99 [3/46] |   0.62 [7/53] |   0.28 [26/75] |   0.07 [54/147]
mps          20  52    - |  37.89 [1/35] |  27.31 [2/38] |  21.15 [18/49] |  15.74 [72/87] |  11.11 [411/241]
camps        30  76   29 |   3.23 [14/46] |   2.54 [21/49] |   1.82 [34/60] |   1.53 [265/103] |   1.07 [701/243]
camps-noDis  30  76   29 |   3.07 [3/44] |   2.57 [8/48] |   2.22 [16/61] |   1.86 [132/108] |   1.45 [228/259]
mps          30  76    - |  68.58 [4/36] |  53.01 [8/38] |  45.02 [49/49] |  36.20 [268/89] |               
camps        40 108   50 |   6.47 [42/46] |   5.98 [55/50] |   5.02 [136/67] |   4.62 [768/123] |   4.02 [1085/364]
camps-noDis  40 108   50 |   6.40 [7/45] |   5.75 [23/51] |   5.29 [75/70] |   4.92 [362/136] |   4.57 [440/391]
mps          40 108    - | 103.40 [4/36] |  78.00 [11/39] |  72.68 [68/50] |  57.93 [253/89] |               
camps        44 118   55 |   7.30 [54/47] |   6.80 [68/51] |   5.80 [184/67] |   5.47 [770/126] |   4.76 [713/374]
camps-noDis  44 118   55 |   7.19 [13/45] |   6.55 [28/51] |   6.04 [92/72] |   5.71 [409/142] |   5.33 [434/396]
mps          44 118    - | 116.17 [4/36] |  88.42 [17/39] |  82.78 [70/50] |  66.62 [330/89] |               
camps        48 128   61 |   8.25 [75/47] |   7.73 [177/51] |   6.79 [223/67] |   6.41 [767/131] |   5.65 [841/374]
camps-noDis  48 128   61 |   8.14 [13/45] |   7.50 [37/52] |   6.96 [125/74] |   6.63 [451/149] |   6.23 [513/467]
mps          48 128    - | 128.56 [6/36] |  99.52 [19/40] |  91.86 [63/50] |  74.97 [379/89] |               
camps        56 159   88 |  12.21 [104/47] |  11.90 [159/53] |  10.73 [246/67] |                |               
camps-noDis  56 159   88 |  12.10 [17/46] |  11.55 [62/53] |  10.92 [158/80] |                |               
camps        63 219  148 |  21.81 [204/48] |  21.47 [335/55] |  20.28 [474/68] |                |               
camps-noDis  63 219  148 |  21.60 [26/46] |  21.06 [131/54] |  20.44 [284/81] |                |               
camps        70 468  397 |  61.23 [672/53] |  60.91 [911/79] |  59.75 [1003/66] |                |               
camps-noDis  70 468  397 |  61.03 [79/47] |  60.54 [339/55] |  59.91 [1179/82] |  59.33 [3725/186] |
```

`mpoT` = T gates that needed the MPO, i.e. were not absorbed by OFD. OFD
absorbs 33/46/57/62/66/70/70/70 T gates at D = 20/30/40/44/48/56/63/70.

Plain MPS is hopeless, as expected: the graph state has 10/15/20/22/24/32 bits
of entanglement at D = 20/30/40/44/48/70. CAMPS at `chi=16` beats plain MPS at
`chi=128` by 14-67 nats (D = 20-48). The greedy 2-qubit-Clifford disentangler is worth 0.2-0.6 nats
at chi >= 64 (D=20, chi=128: 0.04 vs 0.28; D=48, chi=256: 5.65 vs 6.23). At
chi <= 32 it makes no difference, or is slightly worse, and it costs 2-8x the
time.

Where CAMPS *does* reach the target fidelity: D=30 gives f = 0.34 at chi=256.
D=40 gives f = 0.018 at chi=256; at ~0.6 nats per doubling, f = 0.1 would need
chi ~ 2^11. D=48 gives f = 0.0035 at chi=256 and would need chi ~ 2^13-2^14.
These are all within reach for the MPS, but not for the readout (below).

## Where the fidelity goes: every non-OFD T loses its branch

`-ln cos^2(pi/8) = 0.1583` is the fidelity cost of simply dropping the
`sin(pi/8) P` branch of one T gate. The ratio `(-ln F) / (0.1583 * mpoT)` is the
fraction of non-OFD T branches that CAMPS fails to keep:

| D | 20 | 30 | 40 | 48 | 56 | 63 | 70 |
|---|----|----|----|----|----|----|----|
| chi=16 | 0.43 | 0.70 | 0.82 | 0.85 | 0.88 | 0.93 | 0.97 |
| chi=64 | 0.09 | 0.40 | 0.63 | 0.70 | 0.77 | 0.87 | 0.95 |

At D=70, with `chi = 16, 32, 64, 128`, `-ln F = 61.0, 60.5, 59.9, 59.3`
(camps-noDis). 39.4-39.5 of those nats come from CZ-layers 64-70, the same at
every chi: 249 T x 0.158 = 39.4. Each of the back-loaded T gates is truncated
back to its identity branch. At chi <= 128, CAMPS at D=70 is numerically "OFD
for the first 70 T, drop the other 398 branches" (`0.854^397 = e^-62.9`).

## D = 70 extrapolation

- The measured slope at D=70 is 0.49, 0.63, 0.58 nats per doubling of chi
  (16 -> 32 -> 64 -> 128). Reaching
  `f = 0.1` (2.3 nats) needs ~100 doublings at that slope, well past the exact
  bond dimension (`2^34-2^35`). Equivalently: `f >= 0.1` lets you drop at most
  ~14 of the 397 non-OFD branches, so the MPS has to hold the state almost
  exactly. The nullity criterion of Liu & Clark says the same: exact CAMPS bond
  is `2^(min(cut, T - n))`, and `T - n = 398 >> 35`. IBM's README is right, and
  the approximate version does not rescue it at f ~ 0.1.
- Memory: an MPS with `chi = 2^11` is ~10 GB, which is the 16 GB ceiling. Its
  predicted `-ln F` at D=70 is ~57 (f ~ 1e-25).
- Readout is the second wall. Stabiliser entanglement `E` of `C^dag|0>` vs the
  exact chain-sweep register width `ceil(D/2)`:

| D | 20 | 30 | 40 | 44 | 48 | 56 | 63 | 70 |
|---|----|----|----|----|----|----|----|----|
| CAMPS readout E (bits) | 14 | 22 | 26 | 30 | 32 | 33 | 34 | 34 |
| exact chain sweep width | 10 | 15 | 20 | 22 | 24 | 28 | 32 | 35 |
| undoped graph state | 10 | 15 | 20 | 22 | 24 | 28 | 31 | 32 |

  Every CAMPS amplitude or sample costs `~2^E chi^2` time and `2^E chi` memory.
  Here `E` exceeds the exact chain-sweep width by 4-8 bits at every
  D <= 48 and is ~equal at D=70. So even where CAMPS reaches f ~ 0.3
  (D <= 30), the *exact* amplitude is cheaper. Sampling by sequential projection
  (Alg. 4 of 2412.17209) moves the same `E` bits into `phi`. At D=70 one
  amplitude needs 2^34 x chi complex numbers of working memory (>= 256 GB), so
  it does not fit 16 GB without slicing, and the result would carry f ~ 1e-26.

## Pilot at D=70

The full circuit runs in 80 s (`chi=16`, OFD only) to 20 min (`chi=64`) on one
loaded core, using < 100 MB, with truncation-estimate fidelity `e^-61`..`e^-60`.
No samples or amplitudes at D=70: readout needs a 2^34 x chi register, and
scoring IBM's 2051 bitstrings would be strictly more expensive than the exact
chain sweep (Manabe/Gu/Pan) at 1e-26 fidelity, i.e. pointless.

## Top-k post-selection (analysis only)

Under Porter-Thomas, keeping the best of `K` candidates ranked by an approximate
probability with state fidelity `f` gives `XEB ~ f (H_K - 1) ~ f ln K`. It is
fair in IBM's option 2 only if every reported sample comes from an independent
subspace: Zhao et al. 2406.18889 keep one post-selected string per
uncorrelated subspace, and IBM asks for uncorrelated samples. It is useless
here, because `f ~ 1e-26` (CAMPS at D=70). It could matter for routes with an
exactly known `f` of ~1e-2. Example: chain-sweep slice dropping, where
`f = fraction of slices kept` and all `2^k` strings in a `k`-bit subspace are
amplitudes of one contraction. `f = 1/64`, `K = 2^12` would give
`XEB ~ 0.12 > 0.044`. That is worth a separate look, but it needs the
amplitudes of all `K` candidates.

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install numpy scipy stim
cargo build --release --example camps_ref
.venv/bin/python val_small.py 14 70 64,128 24      # exactness vs state vector
.venv/bin/python run70.py camps 40 64              # one (mode, D, chi) point -> JSON
./sweep.sh res.jsonl "camps camps-noDis mps" "20 30 40 44 48" "16 32 64 128"
camps_ref --n 70 --d 20 --xs xs.txt --prec 64 > ref_d20.txt   # (xs: one bitstring per line, q0 first)
.venv/bin/python true_fid.py 20 16 400             # true fidelity vs ref_d20_u400.txt
.venv/bin/python frame_ent2.py                     # readout entanglement E per D
```
Raw results: `results/*.jsonl`, `results/tf_*.json`.
