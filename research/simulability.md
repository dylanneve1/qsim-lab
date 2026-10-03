# Phase diagram of exact simulability

Branch `exp/simulability`. Author: qsim-simulability agent (round 4, 3 Oct 2026).
Code: `src/simulability.rs` (families, features, engine runners), `examples/simulability.rs` (CLI),
`tests/simulability.rs`, `research/data/simulability/{driver.py, run_mac.sh, refeature.py, fit.py}`.
Data:
- `research/data/simulability/raw/{grid}.csv`: Mac, request `<Z^{⊗n}>`; authoritative.
- `raw/{grid}.mid2.csv`: the same instances with the request `<Z_{n/2−1} Z_{n/2}>`.
- `raw/confirm.csv`: re-timing pass.
- `raw/neon_sv/`: state-vector re-time on main 05b85b9.
- `vps/*.csv.gz`: VPS replicate.
- Fits: `fit_report.json` and `winners.csv` (headline), `mid2/` (local request), `mid2/noncert*/`
  and `noncert_all_request/` (certified-zero instances removed), `all_engines/` (observable engines
  included).
- PNGs alongside.

## Headline

**For exact simulation, no single circuit statistic tells you which engine wins. One cheap work
estimate per engine does, with two fitted constants each, `log2 t_e ≈ a_e + b_e·R_e`, all computed in
milliseconds before anything is simulated. Fitted on three families and tested on the fourth, it
picks the fastest state engine 85 % of the time and is within 2× of the best on 91 % of 314
instances (geometric-mean slowdown 1.25×).**
The best single statistic, the rotation frame's active dimension `d`, gets 54 % and 4.5×. "Always
state vector" gets 3 % and 82×. T-count alone gets 19 % and 43×.
The result does not depend on the request or the kernel build:
- with a local observable `<Z_{n/2−1} Z_{n/2}>` instead of `<Z^{⊗n}>`: 85 %, 1.27×;
- on the 207 instances whose value is *not* certified zero: 78 %, 1.46× (best single statistic:
  43 %, 6.9×);
- with the state-vector times re-measured on main's new NEON kernels: 84 %, 1.29× (§6.1b).

The resource coordinates are magic (`d`), entanglement (a bond bound) and superposition (a support
bound), and each engine owns one corner of that space (§5b). Two of the work estimates are close to
exact operation counts (fitted slope ≈ 1: state vector 0.93, compressed state 0.98). The weak link
is MPS: even its *measured* maximum bond explains its run time poorly (§6.3).

## 1. Question and hypothesis

qsim-lab has several exact engines, each with a known asymptotic cost driver. The hypothesis
(ARCHITECTURE.md §1) is that the exact cost is about `min_e exp(c_e · resource_e)`, and that every
`resource_e` can be estimated in O(gates) before simulating. If that holds, a planner can choose the
engine without running anything, and the boundaries between engines form a measurable phase diagram.

Tested here: (i) whether each cheap resource is a faithful work count (log–log slope ≈ 1 and small
scatter), and (ii) whether the min over fitted per-engine models picks the winner on a circuit
family it has never seen.

## 2. Set-up

### Request
Every engine answers the same question: the exact value of `<ψ| Z^{⊗n} |ψ>` for `|ψ> = U|0^n>`.
A global parity observable was chosen so that no engine gets a light-cone shortcut. The value is
checked against the reference state vector (n ≤ 26) or against the median of the other engines.

Engines split into two kinds, and the split turned out to matter (§6.1):

| kind | engine (`run_engine` name) | what it does | exact cost driver |
|---|---|---|---|
| state | `sv` | blocked state vector (cache-blocked executor, f64) | `2^n · stages` |
| state | `sparse` | hash-map sparse state (`SparseState`) | `gates · nnz` |
| state | `mps` | MPS, no bond cap, SVD cutoff 1e-14 (rejected if discarded weight > 1e-10) | `Σ_gates χ^3` |
| state | `hsf` | hybrid Schrödinger–Feynman, KL partition, full 2^n output by GEMM | `paths · (2^{n_A} + 2^{n_B}) · gates + paths · 2^n` |
| state | `tableau` | Heisenberg tableau (Clifford circuits only) | poly |
| state | `cstate` | `adaptive::CompressedState`: Clifford frame + dense register on the `d` active qubits, always evolved | `Σ_j 2^{d_j}` |
| observable | `frame` | rotation-frame Pauli paths (Heisenberg, x-span pruning) | live terms |
| observable | `dense`, `auto` | `adaptive::expectation` with `Strategy::Dense` / `Strategy::Auto` (start in the Heisenberg picture, hand over to the dense register) | either of the above |

The phase diagrams and the headline numbers use the six **state** engines. The observable engines
are analysed separately because they can (and often do) finish without simulating anything.

### Circuit families (`simulability::build`)
| family | parameters | sweeps |
|---|---|---|
| `ct` Clifford+T | `n`, layers `L`, T-count `t`, `nn` | `L` layers of random 1q Cliffords (`I,H,S,HS`) + a CNOT layer (NN brickwork or random matching); `t` T gates (each followed by H) at random slots. Grids: n=24 NN (L ≤ 32, t ≤ 64), n=32 NN (no SV possible), n=20/24 all-to-all. |
| `brick` | `n`, depth `D`, `nn` | Haar-random 1q `U` on every qubit + CZ layer (NN brickwork or random matching). n = 12…26, D = 1…16. |
| `arith` | `bits`, `h`, `reps` | Cuccaro ripple-carry adders on two `bits`-bit registers, `h` qubits of each in `|+>` (rest random classical), `reps` additions alternating `b+=a`, `a+=b`. Pure permutation after the H layer. n = 13…25. |
| `qaoa` | `n`, `p`, `deg`, `nn` | QAOA on a ring-neighbourhood or uniformly random graph with `n·deg/2` edges: `p` rounds of `CNOT·Rz·CNOT` per edge + `Rx` mixers. n = 12…24. |

### Cheap features (`simulability::features`, O(gates · n) except the KL partition)
Each engine gets one *work estimate* `R_e` (log2 of a predicted operation count):

| engine | feature | definition | rigorous? |
|---|---|---|---|
| sv | `sv_l` | `n + log2 gates` | exact up to fusion |
| sparse | `sparse_l` | `log2 gates + sup`, `sup` = affine GF(2) bound on log2 of the support: wires are constant / affine in "branch variables" / opaque; only non-monomial 1q gates create variables | upper bound on nnz |
| mps | `mps_l` | `log2 Σ_gates Σ_cuts swept 2^{3 b_cut(t)}` (+4 bits for SWAP-routed cuts), `b_cut` = time-resolved crossing count (CNOT/CZ/CPhase = 1 bit, others 2) capped by the cut size **and by the support** (`Schmidt rank ≤ nnz ≤ 2^{#branching gates so far}`) | upper bound on χ (tested) |
| hsf | `hsf_l` | `log2(2^k · gates · 2^{max(n_A,n_B)} + 2^{k+n})`, `k` = path bits of the KL partition after modelling exact zero-path pruning (a cut CNOT/CZ whose diagonal-side qubit is still in a definite Z state adds no path) | heuristic |
| cstate | `dense_l` | `log2 Σ_j 2^{d_j}`, `d_j` = active-dimension profile of the rotation frame (`adaptive::active_dimension_profile`) | exact op count |
| tableau | — | applicable iff no non-Clifford rotation | rule |

Plus raw statistics used as single-feature baselines: `n`, gate count, T-count, rotation count, `d`, `chi_bits`, `hsf_k`, `sup`, 2q depth, and `obs_zero` (§6.1). Feature extraction takes a median 6 ms (max 28 ms, VPS) per circuit of 30–1,627 gates, 90 % of it the KL partition; that is below the 1 ms–10 s engine runs it is used to choose between, except for trivially small circuits.

### Harness (`research/data/simulability/driver.py`)
Each (instance, engine) runs in a fresh process (`examples/simulability.rs run ENGINE SPEC SEED MEM`): engine-internal wall time, wall time of the process, peak RSS from `wait4`. Budget: 10 s wall, 1 GiB for any dense register (state vector, compressed register, HSF output; sparse/MPS/frame capped at the same budget). Timed-out / over-budget runs are censored (counted as 2× timeout in the regret metric) and runs dominated by a censored one along a monotone parameter are skipped. `RAYON_NUM_THREADS=1` for every engine: the question is algorithmic work, not parallel efficiency.

### Machine, load, noise
All authoritative timings come from Dylan's MacBook Pro (M1 Pro, 6P+2E cores, 16 GB) under the
swarm bench lock, in chunks of 1.5–3 min, with 2–4 single-threaded workers in parallel. Recorded
1-min load: 1.9–13.8. During some chunks a peer's `rustc` and the owner's Logic Pro were running
(builds don't need the lock). To bound the effect, a **confirmation pass** re-timed every "close
call" (instances where the best two state engines were within 3× of each other and the best took
≥ 1 ms): 44 instances, 88 runs, min of 3, one worker, load 1.9.
- A single shot overestimates the clean min-of-3 by a median 11 % (p10 4 %, p90 21 %, max 49 %).
- The winner changed in **1 of 44** close calls, and that one involved an observable engine.

So the phase diagrams are robust to the noise. Absolute seconds carry about ±20 %.

**Cross-machine replicate.** The four base grids were also run on the VPS (EPYC, AVX2; earlier
build, no `cstate`). The winner among {sv, sparse, mps, hsf, tableau} agrees on **85/92** instances
whose best time is ≥ 1 ms. Per-engine VPS/Mac time ratios differ: sv 1.11, sparse 1.27, mps 1.40,
hsf 1.62 (medians). The boundaries therefore move by up to ~1.5× between machines. Calibration has
to be per machine (the intercepts `a_e`), but the features and slopes carry over.

## 3. Exactness
2,111 successful engine runs on 314 instances. Every engine except MPS agrees with the reference
to ≤ 1e-10. MPS agrees to ≤ 1.8e-7. That gap is the cost of its numerical-rank cutoff (relative
singular-value weight 1e-14, runs rejected if total discarded weight > 1e-10). Setting the cutoff
to 0 is not an option: it keeps numerically zero singular values, and the bond then grows to the
crossing bound. "Exact MPS" therefore always means "exact to the SVD's numerical rank".
`tests/simulability.rs` checks every engine against the state vector on all four families. It also
checks the MPS `Z`-product contraction, the bond and support bounds (as true upper bounds), and
the soundness of the vanishing certificate (§6.1). The existing suites (cross_check,
differential_fuzz, properties, adaptive, pauli_frame, …) still pass with the MPS change.

**Harness bug found by the second request (mine).** The first `tableau` runner conjugated the
observable by `C` instead of `C†`. `PauliSum::conjugate_by_clifford(X)` maps `P → X P X†`, so the
runner must pass the inverted circuit. On `<Z^{⊗n}>` the two agreed on all 20 Clifford instances.
The local request exposed it (`ct:n=20,L=2,t=0,nn=0`: 1.0 instead of 0). It is fixed and has a
150-case regression test. The 40 tableau values in the CSVs were recomputed (1 changed). Tableau
timings are unaffected and are microseconds anyway.

**Engine bug found by the sweep.** `Mps::apply_2q_adjacent` panicked with faer
`SVD did not converge` on `ct:n=24,L=32,t=16` (exact Clifford+T, highly degenerate spectrum).
Fixed in b3a51d1 (`robust_thin_svd`): retry on `m†`, then on `D·m` with a fixed diagonal phase
unitary. Both are exact reformulations, so only rounding differs. A regression test is included,
and the audit agent was told.

## 4. Cost models (`cost_models.png`)
Fit `log2 t = a_e + b_e·R_e` by OLS on the runs ≥ `T_FLOOR` = 1 ms. Below that, times are process
and allocation overhead. There is deliberately no per-engine floor: a floor learned on other
families transfers badly.

| engine | slope `b_e` | in-sample RMSE (log10) | held-out RMSE (log10) by family ct / brick / arith / qaoa |
|---|---|---|---|
| sv | 0.93 | 0.15 | 0.15 / 0.28 / 0.18 / 0.06 |
| cstate | **0.98** | 0.12 | 0.13 / 0.09 / 0.17 / 0.09 |
| sparse | 0.84 | 0.45 | 1.02 / 0.48 / 0.08 / 0.51 |
| hsf | 0.59 | 0.46 | 0.56 / 0.43 / 0.65 / 0.41 |
| mps | 0.27 | 0.84 | 0.81 / 1.29 / 0.89 / 0.93 |

(Held-out = model fitted on the other three families. Runs ≥ 1 ms only.)
- **Exact operation counts.** For `cstate`, `Σ_j 2^{d_j}` predicts run time across families to
  0.1–0.2 decades. The rotation frame's d-profile is a cost oracle for the compressed engine.
- **Sparse.** Exact on arithmetic (0.08), where the support bound is tight. Loose on Clifford+T,
  where `sup` saturates at `n` while the true support of a stabilizer-like state is smaller.
- **HSF.** The nominal path count `2^k` overestimates the work by up to 2^24 (QAOA: `k = 36`, yet
  0.03 s). The engine prunes exactly-zero paths. The pruning-aware estimate (`hsf_keff`: a cut gate
  on a qubit still in a definite Z state adds no path) brings the slope from 0.21 to 0.59.
- **MPS.** The crossing-count bound on χ is *exact* for Haar brickwork and ring-graph QAOA
  (measured max bond = bound) and loose by 2–6 bits on permutation circuits. Capping it by the
  support (Schmidt rank ≤ nnz) fixes part of that. The remaining scatter is not a bound problem:
  see §6.3.
- **What the two refined features buy** (support-capped MPS bound, pruning-aware HSF paths), against
  the nominal ones (`fit.py --v0`):
  - MPS slope 0.18 → 0.27, held-out MPS RMSE 1.1–1.6 → 0.8–1.3.
  - HSF slope 0.21 → 0.59, held-out HSF RMSE 0.7–1.1 → 0.4–0.65.
  - Pooled decision: top-1 82 % → 85 %, regret 1.38 → 1.25. QAOA regret 2.64 → 1.96.

Sensitivity to `T_FLOOR`: pooled held-out geometric regret is 1.22 / 1.25 / 1.22 for 0.2 / 1 / 3 ms
(top-1 82 / 85 / 83 %). The conclusion does not depend on the choice.

## 5. Validation: leave one family out (`fit_report.json` → `lofo`)
Decision rule: pick `argmin_e` of the predicted time over applicable state engines (memory
feasibility from the features; tableau whenever the circuit is Clifford). Regret = (time of
the chosen engine) / (time of the measured best). A censored choice counts as 2× the timeout.
ε-regret adds 1 ms to both, so that choosing a 0.5 ms engine over a 0.05 ms one is not counted as
a 10× failure.

| held-out family | n | top-1 | geo-mean regret | within 2× | geo ε-regret | worst ε-regret |
|---|---|---|---|---|---|---|
| ct (Clifford+T, 3 grids) | 156 | 93 % | 1.08 | 96 % | 1.06 | 20.5 |
| brick | 50 | 70 % | 1.51 | 84 % | 1.45 | 59 |
| arith (adders) | 60 | 88 % | 1.07 | 98 % | 1.06 | 1.9 |
| qaoa | 48 | 71 % | 1.96 | 77 % | 1.92 | 161 |
| **pooled** | **314** | **85 %** | **1.25** | **91 %** | **1.22** | 161 |

Baselines on the same folds (pooled):

| rule | top-1 | geo regret | within 2× |
|---|---|---|---|
| per-engine work model (this) | **85 %** | **1.25** | **91 %** |
| same with `+ c·log2 gates` (2 features) | 84 % | 1.31 | 90 % |
| best single statistic: `d` (≤ 3 intervals → engine, fitted) | 54 % | 4.5 | 59 % |
| rotation count | 37 % | 12 | 41 % |
| bond bound `chi_bits` | 25 % | 34 | 28 % |
| HSF cut `hsf_k` | 25 % | 31 | 28 % |
| `n` | 19 % | 37 | 23 % |
| T-count | 19 % | 43 | 22 % |
| support bound | 20 % | 152 | 23 % |
| always state vector | 3 % | 82 | 26 % |

**Answer to "which single statistic":** none. `d` is the best single axis, because it separates
"Clifford-ish" from "generic". But a rule on one axis cannot know whether a high-`d` circuit is
low-entanglement (MPS) or low-support (sparse), so it fails across families. The per-engine work
estimates are not one statistic. They are the right statistic *per engine*, and taking the min is
what transfers.

**The largest failures.**
- QAOA ring graphs at p ≥ 2 and n ≥ 20 (MPS predicted cheap, measured 20–130× slower).
- n=24 brickwork with random-matching CZ (MPS predicted, HSF best, 15–59×).
- Clifford+T at L=32, t ≥ 48 (MPS predicted, cstate best, 13–21×).

All of these are MPS over-optimism, i.e. the weak MPS cost model.

## 5b. Phase diagrams
- `phase_universal.png`: every instance in normalised resource coordinates (d/n, bond bound/(n/2),
  support bound/n), coloured by the measured winner. Each engine owns a corner. Tableau: d = 0.
  cstate: small d, or maximal entanglement with d < n. MPS: low bond bound at any d. Sparse: support
  ≲ 0.4·n at any entanglement. SV/HSF: everything near 1.
- `phase_ct24.png`, `phase_ct32.png` (T-count × layers). At n = 32 (no state vector possible) the
  diagram is three clean phases: tableau (t = 0) | compressed state (t ≲ 16–24) | MPS. At n = 24 and
  L = 32 the MPS bond saturates, and the compressed state takes the high-t corner back (0.37–0.48 s
  vs MPS 4.8–9.9 s vs SV 7 s). The held-out model gets 93 % of ct right. Its misses sit on the
  cstate/MPS boundary.
- `phase_brick.png` (depth × n, NN). MPS up to a depth D* that grows with n (D* = 12 at n = 16–20,
  16 at n = 24, > 16 at n = 26), then HSF. The two compete on the *same* resource: k = D/2 cut
  gates ↔ χ = 2^{D/2}. MPS pays about `n·D·χ³ = n·D·2^{3D/2}`, while HSF full output pays
  `2^{D/2}·(2^{n/2}·G + 2^n) ≈ 2^{n+D/2}`. That puts the crossover at `D* ≈ n − log2(n·D)` plus a
  constant (≈ 9, 12, 15, 17 for n = 16…26 before constants, vs 12, 12, 16, > 16 measured). The state vector only wins at n = 12, where everything
  is sub-millisecond.
- `phase_arith*.png` (superposed qubits h × repetitions). Sparse wins whenever the support bound is
  ≲ 0.4·n (h ≤ 4 of 12 bits: 2^8 of 2^25 basis states). At h = 6 (support 2^12) MPS takes over by
  less than 2×.
- `phase_qaoa_nn0/1.png`. On random graphs the **compressed state wins everywhere** (n ≤ 24, p ≤ 3),
  1.2–22× faster than the state vector. On ring graphs MPS wins at small p.

**A clean boundary law (compressed state vs state vector).** With both slopes ≈ 1, `cstate` beats
`sv` iff `Σ_j 2^{d_j} ≲ G·2^n·2^{a_sv − a_cstate}`, i.e. roughly iff `d < n − log2(m/G) + 0.7`, where
`m` is the number of non-Clifford rotations and `G` the gate count. QAOA has `m/G ≈ 0.4` and
`d ≤ n − 1`: cstate wins, as measured. Haar brickwork has `m/G ≈ 2` (three rotations per `U`) and
`d = n`: SV wins, as measured. Absorbing Cliffords into the frame is worth exactly `log2(G/m)` bits
of register.

## 6. Surprises (the interesting part)

### 6.1 A side result: an O(gates·n) certificate that a Pauli expectation vanishes

**Statement** (`adaptive::z_product_vanishes`). Write `U = C·R_m⋯R_1` in the rotation frame. `C` is
the Clifford part, and `R_j = exp(−iθ_j Q_j/2)` are the non-Clifford rotations with Heisenberg
axes `Q_j`. Let `W = span{x(Q_1), …, x(Q_m)} ⊆ GF(2)^n` (the same space whose dimension is the
active dimension `d`). For a Pauli `P`, if `x(C†PC) ∉ W`, then `<0^n|U†PU|0^n> = 0` exactly.

**Proof.** research/pauli.md §2, Lemma (x-span pruning), applied at stage `m`. Conjugating
`C†PC` back through the rotations only produces strings `±C†PC·Q_{i_1}⋯Q_{i_r}`, whose x-parts lie
in the coset `x(C†PC) + W`. That coset misses 0, so every string has `<0|·|0> = 0`.

**Cost.** One Heisenberg-tableau pass over the gates (O(gates·n/64) words), Gaussian elimination
of the `m` axes (O(m·n·d/64)) and one reduction. That is the same work as computing `d`, in about
a millisecond at our sizes. Nothing is simulated.

**When it fires.** If the Cliffords scramble, `x(C†PC)` behaves like a uniform vector, and the
certificate fires with probability ≈ `1 − 2^{d−n}`. It can only fire when `d < n`, i.e. while the
rotations have not yet spanned all of GF(2)^n. Measured for `P = Z^{⊗n}`:

| family | instances | predicted fires (Σ 1 − 2^{d−n}) | observed fires | value actually 0 |
|---|---|---|---|---|
| Clifford+T (3 grids) | 156 | 144.2 | **142** | 150 |
| QAOA | 48 | 29.4 | 15 | 15 |
| adders | 60 | 9.0 | 0 | 25 |
| brickwork | 50 | 0 (d = n) | 0 | 0 |

For scrambling Clifford+T circuits the uniform model is essentially exact (144 predicted vs 142
observed). On structured circuits it is not. QAOA is invariant under the global flip `X^{⊗n}`,
which aligns the observable with the circuit, and the certificate fires half as often as the
model says. The adder zeros come from a balanced output parity, which this certificate cannot see.
It is sound but incomplete: there were 8 Clifford+T zeros it did not certify.

**Why it matters for the phase diagram.** The request is part of the problem. With observable
engines included (`all_engines/`), the Heisenberg frame beats every state engine on 53/156
Clifford+T and 9/48 QAOA instances, mostly because of this early-out: frame finished with zero live
terms in 155/246 of its successful runs. A held-out model that includes those engines drops to
67 % top-1 (worst 3,600×), because the frame's cost when *not* certified is not captured by any
cheap feature (slope 0.20). In a first version `Strategy::Dense` was counted as a state engine and
"won" n = 32, d = 29 instances in 0.85 ms that the real compressed state cannot even allocate
(2^29 × 16 B). That is why `cstate` exists, and why the headline uses only state engines, whose
cost does not depend on the observable (§6.1b).

### 6.1b Re-check with a local observable and on a newer kernel
Prompted by the vanishing certificate, the whole 314-instance sweep was re-run (all nine engines,
Mac, same build) with the request `<Z_{n/2−1} Z_{n/2}>` (`OBS=mid2`), to test whether any of the
above is an artefact of the global observable (`mid2/compare_requests.txt`).

- **State-engine costs do not depend on the request.** Median time ratio mid2/all: sv 0.96,
  hsf 0.92, sparse 0.92, mps 0.96, cstate 1.00. These are within the load noise; the mid2 sweep ran
  at lower load. The winner among state engines is identical on **90/92** instances (best ≥ 1 ms).
  None of them short-circuits by construction, since the observable enters only an O(state)
  readout.
- **The local request is less often trivial, but not rarely.** The exact value is 0 on 159/314
  instances (194 for `Z^{⊗n}`), and certified 0 on 107 (157). Observable-engine short-circuits
  (finished with 0 live terms): frame 107/296, auto 81/303, dense 21/300 (155/246, 126/300 and
  23/298 for `Z^{⊗n}`).
- **Held-out results on mid2** (state engines): 84.7 % top-1, regret 1.27, 90 % within 2×.
  By family: ct 94 % / 1.08, brick 72 % / 1.51, arith 88 % / 1.07, qaoa 65 % / 2.28.
- **Only non-certified instances** (207 for mid2): 77.8 % top-1, regret 1.46, 84 % within 2×.
  Best single statistic: 43 % / 6.9×. Always-SV: 6 % / 47×. The same filter on `Z^{⊗n}`
  (157 instances): 77 % / 1.43×.
  Removing the trivial instances costs about 7 points of accuracy, because the easy Clifford+T
  cases leave. The transfer result stands.
- **With observable engines, non-certified only:** 66 % / 1.49× (worst 2,100×). The Heisenberg
  frame's cost on a request that does not vanish is still not predicted by any cheap feature (open
  question 2).

**Kernel change.** Main moved to 05b85b9 (aarch64 NEON FMA, 1 MiB default blocks) after these
timings. All 229 state-vector runs ≥ 1 ms were re-timed on `exp/simulability-neon` (this branch +
main): median 1.26× faster (p10 0.51, p90 1.45; the tail is load, 11–17 during the last chunks).
The SV slope falls from 0.93 to 0.72. Refitting with the new SV times gives 83.8 % top-1, regret
1.29 (vs 85.0 %, 1.25). The intercepts move with the build, and the decision quality does not.

### 6.2 The compressed state is the default winner for diagonal-heavy circuits
For QAOA on random graphs the rotation frame removes every CNOT and H. The compressed register has
`d = n − 1…n − 3` qubits and only `m ≈ 0.4·G` rotations, so `cstate` is the fastest state engine on
**all 24** random-graph instances. It beats the cache-blocked state vector by 1.2–22× (median 3.3×)
single-threaded, at a median 1.75× less peak RSS. This is not a "Clifford+T
engine" result. It applies to any circuit whose non-Clifford content is diagonal-ish. The pipeline
only tries the compressed state when `d + ADAPTIVE_MARGIN ≤ n` (margin 1, pipeline.rs). By this data
the condition should be `d < n − log2(m/G) + c` (§5b), which also admits `d = n` when `m/G ≲ 0.6`.

### 6.3 MPS cost is not determined by its bond dimension
Even with an *oracle* feature, `log2(G·χ_measured³)` (the measured maximum bond), MPS run time fits
with slope 0.44 and RMSE 0.54 decades. The bound gets 0.27 / 0.84. Most gates act on cuts far below
the maximum χ, and at small χ fixed SVD overhead dominates. The right feature is the *time-resolved*
bond profile, which the crossing bound approximates badly whenever a circuit disentangles: CNOTs on
definite controls, adders on classical registers, Clifford layers that undo each other. That is
open question 1.

### 6.4 Smaller ones
- `Strategy::Auto` (adaptive.rs) is within 6 % of `min(frame, dense)` at the median (geo 1.25×,
  300 instances). It misfires by up to 350× on the adders (`arith:bits=12,h=2,reps=1`: auto 1.47 s
  vs frame 4 ms). The live growth-rate meter hands over to the dense register too early on
  permutation-heavy circuits, whose Heisenberg term count stays tiny.
- HSF with full 2^n output beats the single-threaded blocked state vector 4–26× on n = 20–26 NN
  brickwork up to D = 16 (16 instances). research/hsf.md assumed this advantage had disappeared against the blocked
  executor. Single-threaded it hasn't. Its GEMM accumulation is the efficient part.
- The sparse engine's timeouts on Clifford+T are *non-monotone* in t (each T is followed by an H in
  this family, which can either branch or re-merge the support). Dominance skipping is therefore
  only applied along parameters that are monotone for that engine.

## 7. Known vs new
**Known (and re-derived here as cost drivers):**
- Stabilizer circuits are polynomial (Gottesman–Knill; Aaronson–Gottesman 2004).
- Clifford+T compresses to at most t qubits (Jozsa–Van den Nest "compute and compress";
  Yoganathan–Jozsa–Strelchuk 2019). The circuit-specific active dimension `d ≤ t` and a dynamically
  sized register are in Clifft (arXiv:2604.27058). Stabilizer-rank and ZX-cutting decompositions
  scale as 2^{αt} with α < 1 (Bravyi–Gosset 2016; Bravyi et al. 2019; arXiv:2403.10964). Those are
  magic monotones that are tighter than our `d` but far more expensive to compute.
- MPS cost is poly(n)·χ³ with χ bounded by the entanglement (Vidal 2003). Entanglement and
  contraction cost are bounded by gate counts across cuts / treewidth (Jozsa 2006; Markov–Shi 2008).
- HSF costs Π rank(cut gate) paths (Markov et al. 2018, qsimh).
- Sparse simulation costs the support size (e.g. Jaques–Häner 2022).
- Engines that *combine* resources: stabilizer tensor networks (arXiv:2403.08724), CAMPS
  (arXiv:2412.17209).
- Simulator selection: rule-based in Qiskit Aer's `automatic` method and HybridQ's dispatcher.
  Learned run-time prediction exists for single (approximate, TN) simulators across circuit
  families (arXiv:2606.11620, "family-aware" residual models) and for device selection (MQT
  Predictor).

**New here, as far as I found:**
1. A head-to-head of six *exact* engines on identical requests, with **mechanistic** per-engine
   work estimates (no learned black box, 2 constants per engine), validated **leave-one-family-out**.
   The negative result that no single statistic transfers (best `d`: 54 % vs 85 %) is, to my
   knowledge, not in the literature.
2. Cheap rigorous bound: bond ≤ min(crossing count, cut size, support), with the support counted
   time-resolved and Toffolis treated as permutations. A pruning-aware HSF path estimate
   (heuristic).
3. The O(gates·n) vanishing certificate for Pauli expectations (the lemma is research/pauli.md's;
   using it as a planner-time certificate is the new use).
4. The cstate-vs-SV law `d < n − log2(m/G) + c`: Clifford absorption buys `log2(G/m)` qubits.

## 8. Open questions (the three biggest)
1. **A cheap, tight entanglement-profile estimate.** MPS is the only engine whose cost the
   features don't predict (held-out RMSE 0.8–1.3 decades), and even the measured max bond explains
   it poorly. Candidates: stabilizer entanglement of the Clifford skeleton plus a T-count
   correction (exact for t = 0, O(n³) per cut), or a min-cut/light-cone bound on the circuit's
   tensor network (Markov–Shi). The question is whether any O(gates·poly n) quantity tracks the
   time-resolved χ profile across families.
2. **Request-aware features.** For observables, the dominant cost driver is the Heisenberg term
   count after x-span pruning. That is cheap when the certificate fires and unpredicted otherwise.
   Is there an O(gates·n) estimate of the frame's peak term count, e.g. the `min(m − k, 2 d_k)`
   meet-in-the-middle profile with a measured growth rate? The same for amplitudes (HSF's home
   ground) and for sampling.
3. **Does the phase diagram survive parallelism and scale?** Everything here is single-threaded at
   n ≤ 32 with a 1 GiB / 10 s budget. Engines parallelise differently: SV and HSF near-linearly,
   MPS (SVD-bound) poorly, sparse not at all. Intercepts are machine-specific (VPS vs Mac shifts
   boundaries by up to 1.5×). The open question is whether the min-over-engines model, with
   per-machine intercepts and a thread-count term, still transfers at n = 30–40 and 8 threads,
   where the budget and the out-of-core engine reshape the SV corner.

## 9. Reproduce
```
cargo build --release --example simulability
B=target/release/examples/simulability
$B features 'ct:n=24,L=8,t=24,nn=1' 1          # one JSON line of features
$B run cstate 'qaoa:n=20,p=2,deg=3,nn=0' 1      # one engine, one instance
cd research/data/simulability
python3 driver.py --bin $B --grid ct24 --out ct24.csv --timeout 10   # grids: ct24 ct32 ctnn0 brick arith qaoa
python3 refeature.py $B ct24.csv ct24.rf.csv    # recompute features only
python3 driver.py ... --obs mid2                # the local request (all | mid2 | mid4)
python3 fit.py OUTDIR raw/{ct24,ct32,ctnn0,brick,arith,qaoa}.csv   # headline; --all adds observable engines,
                                                # --noncert drops certified-zero instances, --v0 nominal MPS/HSF features
python3 compare_requests.py raw .mid2           # request comparison (§6.1b)
```
On the Mac: `WORKERS=2 run_mac.sh OUTDIR grid...` (takes the shared bench lock in ≤ 3 min chunks
and releases it on exit).
