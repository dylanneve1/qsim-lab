# Spoofing "quantum utility" on a laptop: sparse Pauli dynamics for IBM's 127-qubit kicked-Ising experiment

Branch `exp/spoof-utility`. Target: Kim et al., *Evidence for the utility of
quantum computing before fault tolerance*, Nature **618**, 500 (2023). Their
experiment ran 127-qubit kicked transverse-field Ising Trotter circuits on
ibm_kyiv and reported zero-noise-extrapolated (ZNE) observables. This branch
adds a Rust sparse-Pauli-dynamics (SPD) engine, `qsim_lab::spd`, and tests how
far it gets on one M1 Pro laptop under the swarm budget (2 worker threads,
≤ 3 GB per process, no GPU).

## TL;DR

| Kim et al. figure | observable | M1 result (2 threads) | error vs exact / best classical | experiment (ZNE) error |
|---|---|---|---|---|
| 3a | `M_z`, 5 steps | **exact to 1.6e-6**, 0.02–0.06 s per point | exact (Kim et al.) | ≤ 0.024 |
| 3b | weight-10, 5 steps | **5e-8**, ≤ 7 s per point (δ = 1e-8) | exact (Kim et al.) | ≤ 0.05 |
| 3c | weight-17, 5 steps | ≤ **1.2e-3** (δ = 1e-5, stream 2; 1–4 min per point) | exact (Kim et al.) | ≤ 0.14 |
| 4a | weight-17, 5 steps + RX | ≤ **3.2e-3** (δ = 1e-4/1e-5) | BP-TNS χ→∞ (Tindall et al.) | ≤ 0.09 |
| 4b | `⟨Z_62⟩`, **20 steps** | θ ≤ 0.4 and θ ≥ 0.9: ≤ 0.009. **0.5 ≤ θ ≤ 0.8: off by 0.03–0.22** | BP-TNS χ→∞ (Tindall et al.) | ≤ 0.12 |

* Every **5-step** figure (the "verifiable" Fig. 3 and the "beyond exact" weight-17 Fig. 4a)
  is reproduced on the laptop. The error is orders of magnitude below the
  experiment's, at seconds to a few minutes per point.
* The **20-step** `⟨Z_62⟩` (Fig. 4b, the experiment's headline "beyond classical"
  point) is **not** reproduced to experimental accuracy on this budget in the
  window `0.5 ≤ θ_h ≤ 0.8`. Converged SPD there needs ~10⁹ Pauli strings at
  δ ≈ 2⁻¹⁹–2⁻²¹ (Begušić, Gray & Chan's Table S3: 760–885 M strings, 10–13 h
  on 6 cores, far more than 3 GB). Within 3 GB, SPD is *biased low* by up to 0.22,
  worse than the experiment's ZNE (0.12) at θ = 0.6.
* **The usual SPD error bar is unreliable at depth 20.** The δ-ladder
  difference `|v(δ) − v(δ/3)|` reports 0.013 at θ = 0.6 when the true error
  is 0.165. A 24-qubit heavy-hex patch at 20 steps has an exact answer, and on
  it SPD's error *grows* from 0.055 to 0.081 as δ goes 1e-4 → 1e-5. Convergence
  is non-monotone and slow. The discarded Frobenius weight `1 − ‖O‖²` is the
  honest warning light: it is ≥ 0.05 at every point that is off by more than 0.01.
* Larger lattices (433 Osprey-sized, 1121 Condor-sized): for local observables
  SPD cost depends on the light cone, not on the device. See §6.

## 1. Data and references used

* **Experiment** — the authors' data repository
  (github.com/youngseok-kim1/Evidence-for-the-utility-of-quantum-computing-before-fault-tolerance,
  = figshare 10.6084/m9.figshare.22500355). `research/data/spoof-utility/extract_kim.py`
  re-implements their notebooks' selection rule: the ZNE value is the last of
  (linear, exponential) fits with fit uncertainty < 0.5, otherwise the
  unmitigated value, with a 68 % bootstrap interval. Output: `kim_fig*_experiment.csv`.
  Kim et al.'s exact curves (3a–3c, 0.01 grid) and MPS curves are in `kim/`.
* **Coupling map** — the 144 edges published in the authors' `Fig3a.ipynb`
  (three edge-colour layers of 48). `Lattice::eagle127()` reproduces them
  exactly (test `eagle_matches_published_coupling_map`).
* **Best classical reference beyond 5 steps** — Tindall, Fishman, Stoudenmire &
  Sels, PRX Quantum 5, 010308 (2024), arXiv:2306.14887: BP-evolved tensor
  networks extrapolated to χ→∞. Data from github.com/JoeyT1994/BP-TNS-Data
  (`tindall/`): `⟨Z_62⟩` at 20 steps on a 0.05 grid, the weight-17 Fig. 4a
  observable, and `⟨Z_62⟩(t)` for t ≤ 20 at θ = 0.6, 0.8, 1.0. Begušić, Gray & Chan
  (Sci. Adv. 10, eadk4321 (2024), arXiv:2308.05077) find their MIX-TN, SPD and
  BP-TNS results agree to < 0.045 at 20 steps, and estimate MIX-TN to be
  accurate to < 0.01.
* Sign convention: our `<0|U† O U|0>` with `U = (Π e^{+iπ Z_j Z_k/4} Π e^{−iθ X_j/2})^T`
  matches Kim et al.'s stored data for all five observables, including the
  stored sign of the weight-17 data. Their Fig. 4a notebook plots `−1 ×` the
  stored values, and Tindall's file uses that plotted sign, so we compare
  against `−W17_Tindall`.

## 2. The engine (`src/spd.rs`)

Heisenberg picture: push `O` backwards through the circuit, keep a sparse sum of
Pauli strings with real coefficients, evaluate on `|0…0⟩`. The cost-relevant
design choices:

1. **The whole ZZ layer is one O(|x|) bit operation.** All 144 `RZZ(−π/2)`
   commute, and `x` never changes under them. A string anticommutes with `Z_aZ_b`
   iff `x_a ≠ x_b`. The layer is therefore `z ← z ⊕ s` with `s_q = deg(q)x_q ⊕ ⊕_{r~q} x_r`
   (precomputed per-qubit flip masks), with sign `i^{k+|x∧z|−|x∧z'|}`, where `k` is
   the number of cut edges. The cost is proportional to the number of X/Y sites, not to the number of edges.
2. **RX layer as one depth-first branching per term.** `Z → cos Z + sin Y`,
   `Y → cos Y − sin Z`. Children are enumerated depth first. The coefficient is
   monotone along the tree (factors ≤ 1), so a subtree is cut as soon as it drops
   below the branch threshold, and every child ≥ threshold is still produced.
   Children of all parents are merged in a 128-shard hash map. Merged
   coefficients < δ are dropped.
3. **The first RX layer is closed form**: `⟨0|RX†PRX|0⟩ = Π_sites {I:1, Z:cos θ, Y:−sin θ, X:0}`.
   No branching or merge is needed for the last backward layer.
4. **The last branching layer is streamed** (`stream = 1`, default). It feeds
   a linear evaluation, so its children are evaluated depth first instead of
   being stored and merged. Work is unchanged and the largest term table
   disappears: Fig. 3b at δ = 1e-5 drops from 16.6 M stored terms to 0.15 M
   (26 MB RSS). `stream = k` streams k layers, trading time for memory.
5. **Light cone and relabelling.** Only qubits within graph distance `T` of
   the observable are kept, relabelled in BFS order. The key width (64-bit words)
   then follows the cone: 37 qubits (1 word) for the weight-10 observable, 68 (2)
   for weight-17, 127 (2) for 20-step `Z_62`, 391 (7) for a bulk qubit of the
   1121-qubit lattice at 20 steps.
6. **Truncation accounting.** For each cut it records the l1 mass (a rigorous but
   loose bound `|Δ⟨O⟩| ≤ Σ|c_cut|`, checked in tests) and the squared norm. It also
   reports the normalised Frobenius norm `‖O‖²` entering the last layer, which is
   1 for unit-norm Pauli strings without truncation.
7. **Noise-aware mode.** A depolarizing channel `p` on every qubit after each ZZ
   layer is applied exactly in the Heisenberg picture as `(1 − 4p/3)^{weight}`.
   An optional Pauli-weight cap is also available.
8. **Parallelism.** rayon over parent chunks with per-shard flush buffers. All runs
   used `RAYON_NUM_THREADS=2`.

`examples/spoof_utility.rs` drives the figures (`3a|3b|3c|4a|4b|mz|z<q>`,
`--delta`, `--steps`, `--lattice 127|433|1121`, `--depol`, `--max-weight`,
`--stream`, `--branch-factor`, `--mem-gb`). `examples/spoof_patch.rs` compares
truncated SPD with the exact state vector on heavy-hex patches at any depth.

### Validation (`tests/spd.rs`, 8 tests, all green on the M1)

* `eagle_matches_published_coupling_map`. Also checks that the three published
  layers are perfect matchings. `larger_heavy_hex_lattices`: 433 / 504 edges,
  1121 / 1320 edges, degree ≤ 3, connected, no two degree-3 qubits adjacent.
* `exact_spd_matches_statevector_on_heavy_hex_patches`. δ = 0 against the dense
  state vector to 1e-10 on four Eagle patches (12–16 qubits around qubits 62,
  0, 37 and 75). Covers 1–4 steps, random θ, with and without the extra RX
  layer, `M_z`, random weight-1/2/3/5 strings, and multi-term observables with
  repeated strings. Each case runs with light cone on/off and stream = 0, 1, 2
  and 99. More than half of the values are non-zero (guard against degenerate
  zero tests).
* `exact_spd_matches_statevector_at_special_angles` (θ = 0, π/4, ±π/2, π, 3π/2).
* `exact_spd_matches_pauli_path_engine_at_127_qubits`. Full 127-qubit Eagle,
  1–3 steps, the paper's observables, against the independent exact
  Clifford+Rz Pauli-path engine (`pauli_path::expectation`) on the gate-level
  circuit (RZZ = CNOT·Rz·CNOT). At least 20 cases, all agree to 1e-10.
* `clifford_point_matches_pauli_path_at_depth_20`. θ = π/2 up to 20 steps. The
  weight-10/17 observables are ±1 stabilizers, and SPD keeps exactly one term.
* `truncation_error_is_within_the_l1_bound_and_converges` (stream 0/1/3, δ and
  weight caps).
* `noisy_spd_matches_density_matrix`: 6-qubit patch, 1–3 steps, exact density
  matrix with per-qubit depolarizing after each ZZ layer, agreement to 1e-10.
* Independently of the tests, step-by-step `⟨Z_62⟩(t)` at θ = 0.6 for t = 1…7
  (δ = 1e-6, effectively exact) reproduces Tindall et al.'s published dynamics
  to all printed digits: 0.8253, 0.6812, 0.7414, 0.7568, 0.7609, 0.7686, 0.7474.

## 3. Results on the 127-qubit circuits (M1 Pro, 2 threads)

![Kim et al. figures](data/spoof-utility/figures_kim.png)

Full per-θ tables (SPD value, δ-ladder difference, norm deficit, time,
reference, experiment and CI) are in
[`data/spoof-utility/tables.md`](data/spoof-utility/tables.md). Summary:

TABLE_SUMMARY

**Timings** (bench lock, interleaved, min of 3. Machine load average 12–19 from
other agents' work, so these are upper bounds):

TABLE_TIMING

For comparison, Begušić & Chan's single-core Python SPD needed ~15 min per
point for an exact `M_z`, and 45 min to 6 h (4 cores) to reach 1e-4 on the
high-weight observables. Their "SPD 10 s" fast results have a ~1e-3 error,
which matches ours at δ = 1e-4. Our `M_z` is exact in 0.02–0.06 s for two
reasons: `Z_q` commutes with the last ZZ layer, and the first RX layer is
closed form, which leaves three branching layers for 5 steps.

## 4. The 20-step point, where the laptop loses

![convergence](data/spoof-utility/convergence.png)

Left: at 5 steps the error against the exact curves falls steadily with δ.
Middle: `⟨Z_62⟩` at 20 steps against δ (dotted lines are BP-TNS). For
0.5 ≤ θ ≤ 0.8 the value moves *away* from the reference as δ shrinks over the
range that fits in 3 GB. Right: true error against the norm deficit
`1 − ‖O‖²`.

![depth scan](data/spoof-utility/depth_scan.png)

**Depth scan** (`⟨Z_62⟩(t)`, BP-TNS dynamics of Tindall et al. as the
reference). At θ = 0.6, SPD with δ = 3e-5 stays within ~0.01 up to 10 steps,
is off by ~0.06 at 12 steps, and is 0.17 low at 20. At θ = 0.8 the deviation
starts earlier. At θ = 1.0 the operator norm collapses (`‖O‖² → 0`) and SPD
returns ≈ 0, while the truth is 0.02–0.1.

DEPTH_TABLE

**Ground truth at depth 20: heavy-hex patches.** Ruling out a reference
problem or an engine bug needs an exact answer at 20 steps.
`examples/spoof_patch.rs` uses BFS balls around qubit 62 (trees of 12–24
qubits) and the dense state vector:

PATCH_TABLE

Even a 12–16 qubit patch needs 2–8 M Pauli strings at δ = 1e-5 after 20 steps
(the full space is 4^k), and its error is still 1–2 %. On 24 qubits the error grows
with decreasing δ through 1e-4 → 1e-5. This is the same non-monotone
behaviour Begušić, Gray & Chan report for SPD and PEPO (their Fig. S3), now
measured against an exact answer. At 20 steps and intermediate θ the Heisenberg
operator is genuinely scrambled. SPD is efficient only while the discarded
weight stays small.

Two further attempts did not help within the budget:

* **Per-gate-like merging** (`branch_factor` 0.1). Prune children at 0.1·δ and
  merge before applying δ, approximating Begušić & Chan's truncate-after-every-gate
  rule. At θ = 0.6 (127 qubits, 20 steps, δ = 1e-4) this gives 0.5055 instead
  of 0.5160. On the 24-qubit patch it gives 0.537 instead of 0.564 (exact 0.619).
  It is not closer, and it costs 25–30×.
* **Pauli-weight caps** (LOWESA / Shao et al. style) on top of δ: WEIGHT_RESULT

**Takeaway.** The experiment's 20-step point is "spoofed" in the sense that
several classical methods reproduce it (BP-TNS in minutes on a laptop CPU per
Tindall et al., MIX-TN, and 10⁹-term SPD on a workstation). Plain SPD within
3 GB on an M1 is *not* one of them. It agrees with the ZNE data roughly as
well as the earlier "SPD 10 s" results did, but it is biased by up to 0.22
against the converged value, and its δ-ladder error bar hides that. Report the
norm deficit next to every SPD number.

## 5. Noise-aware SPD

NOISE_SECTION

## 6. Larger heavy-hex lattices: 433 (Osprey) and 1121 (Condor)

LATTICE_SECTION

## 7. Known vs new

**Known** (and cited):

* SPD itself: Begušić & Chan arXiv:2306.16372; Begušić, Gray & Chan,
  Sci. Adv. 2024 (arXiv:2308.05077), including the threshold rule, the
  converged 20-step results and their non-monotone convergence. The brief's
  arXiv:2306.04797 is the related Clifford-perturbation-theory paper by
  Begušić, Hejazi & Chan.
* BP tensor networks: Tindall et al., PRX Quantum 2024. gPEPS on 127/433/1121
  qubits and 5 steps with long-time bulk dynamics: Patra et al.,
  arXiv:2309.15642. Light-cone / effective-volume arguments and a 31-qubit
  exact simulation: Kechedzhi et al., arXiv:2306.15970. Heisenberg MPO and
  benchmarking of ZNE: Anand et al., arXiv:2306.17839. PEPO: Liao et al.,
  arXiv:2308.03082. Pauli propagation with weight and frequency truncation
  (LOWESA): Rudolph et al., arXiv:2308.09109. Low-weight truncation: Shao et al.
* That noise makes such circuits classically easy: Aharonov et al. 2022,
  Schuster et al. arXiv:2407.12768.

**New here** (modest; mostly engineering and measurement):

* A Rust SPD engine for these circuits:
  * whole ZZ layer per term in O(|x|);
  * δ-pruned depth-first RX branching;
  * closed-form first layer;
  * streamed last layer, which cuts peak memory 10–100×;
  * light-cone-relabelled keys, so a 1121-qubit lattice costs what its cone costs.

  It is differential-tested against the dense SV, an independent 127-qubit exact
  engine, the published coupling map and a noisy density matrix.
* Exact `M_z` (5 steps) at < 0.1 s per point and the weight-10 observable to
  5e-8 on a laptop. All 5-step figures are reproduced at 10–1000× better
  accuracy than the experiment.
* A ground-truth measurement (exact state vector on heavy-hex patches at 20
  steps) showing that truncated SPD converges non-monotonically there. The
  δ-ladder error estimate understates the true error by ~10× on the 127-qubit
  `⟨Z_62⟩` (0.013 vs 0.165 at θ = 0.6). The norm deficit flags every bad point.
* The noise-aware and large-lattice results of §5 and §6.

## 8. Caveats

* All timings are on a shared, loaded M1 Pro (load 12–30 from other agents)
  with 2 threads. Campaign wall times are not under the bench lock. Only
  §3's timing table is.
* The BP-TNS χ→∞ values used as the 20-step reference are themselves
  approximate. Begušić, Gray & Chan quote < 0.01 for MIX-TN, and the
  independent methods agree within 0.045. Our conclusion (SPD error 0.1–0.2 at
  θ = 0.6–0.7) is far outside that band, and the patch study confirms the
  mechanism with exact references.
* Kim et al.'s "exact" weight-17 curve (Fig. 3c) is an MPS χ = 2048 LCDR
  calculation that they state is exact to plotting precision. Our residual
  ~1e-3 there is consistent with SPD truncation (it shrinks with δ) and not
  with a reference error.
* The noise model (uniform single-qubit depolarizing after every ZZ layer,
  readout folded in) is deliberately one-parameter. It is not a fit of the
  device's sparse Pauli–Lindblad noise.
