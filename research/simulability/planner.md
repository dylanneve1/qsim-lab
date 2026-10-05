# Planner v0, and predicting the cost of exact MPS

Branch `exp/planner` (rebased on main 73fb9fb). Author: qsim-planner agent
(round 4, 3–4 Oct 2026).
Code:
- `src/engines/mps_cost.rs`: rigorous bond bounds and the MPS replay.
- `src/planner.rs`: Planner v0.
- `src/engines/mps.rs`: operation counters, bond trace, per-step rank caps, and an SVD robustness fix.
- `src/engines/adaptive.rs`: the `Strategy::Auto` change.
- `src/compile/plan.rs` and `src/pipeline.rs`: expectation values now go through the planner, with a debug mode.
- `src/simulability.rs`: new `hea` and `qft` families, and new engines `plan`/`planx`/`planp`/`mpsb`.
- `tests/planner.rs`.

Data and scripts are in `research/data/planner/`:
- `collect.py`: deterministic MPS data per instance, run on the VPS. Outputs `mpsdata_vps.jsonl` and `mpsdata_new_vps.jsonl`.
- `fit_planner.py`: predictor and decision study. Outputs `fit_v1/`.
- `compare_auto.py`: Auto before and after the change. Outputs `auto_compare.txt`.
- `eval_planner.py`: end-to-end Mac runs. Outputs `eval_final/`, plus `eval_v0_unstaged/`, the first, unstaged version.
- `mac/*.csv`: raw Mac timings.
- `run_mac.sh`: the locked runner.

## Headline

1. **What an exact MPS run costs is fully determined by its time-resolved bond trace.** I wrote a symbolic replay of the
   MPS engine's exact control flow: orthogonality-centre QR moves, SWAP routing, the Toffoli decomposition, and one SVD per
   adjacent two-qubit application. It tracks bond dimensions only. Fed the real run's bond trace, it reproduces the engine's
   operation counts *exactly* (291/291 runs). Those counts predict Mac MPS time to **0.08 decades RMSE held out by family**
   (slope 0.83). On the two families never used for fitting (hea, qft) the error is 0.08–0.10. The previous study concluded
   that "MPS cost is not determined by its bond dimension" (§6.3 there; the measured max bond gave 0.54 decades). That was a
   modelling artefact: the per-step profile and the control flow matter, the maximum does not.
2. **The remaining problem is predicting the bond trace, and cheap rigorous bounds get a good part of the way.** The "best"
   bound is the minimum of a crossing count, the exact stabilizer entanglement plus rotations that straddle the cut, a
   stabilizer-coset bound, and a per-cut affine support bound. Fed into the replay, it lowers the held-out MPS-time RMSE
   from **0.93 to 0.69** decades (0.97–1.0 to 0.39–0.57 on hea/qft). The planner's held-out worst case drops from
   **162× to 12.9×**.
   A capped probe run, MPS with χ ≤ 16, exact below the cap with the bound above it, gets **0.41** (0.14–0.26 on hea/qft),
   with worst case 4.3×.
3. **Planner v0** (`planner::plan` / `planner::expectation`, wired into `pipeline::simulate` for expectation values):
   - Leave-one-family-out over the 314 instances: **88.2 % top-1, geo regret 1.09, 95 % within 2×, worst 12.9×**. The
     published model: 85 %, 1.25, 91 %, 161×.
   - With the probe the same evaluation gives 93 %, 1.04×, worst 4.3×.
   - On the held-out families hea and qft (36 instances, constants fitted on the other four families), the engine choice
     is right on 88 % and 83 % of instances, with geo regret 1.03 and 1.10 and worst 1.9× and 2.0×.
   - End to end on the Mac, including planning time and speculation, the geo ε-regret is 1.34 on the dataset and about
     1.45 on hea/qft. On instances whose best engine takes ≥ 0.1 s it is 1.20× (§5): planning and speculation are not
     free.
4. **Side results.**
   - Exact MPS runs keep spurious singular values on long Clifford-heavy circuits: kept ranks such as 69, 138 and 148 on a
     *pure Clifford* circuit, whose Schmidt ranks are powers of two. Capping each SVD at the rigorous bound removes them
     (`Mps::set_step_caps`).
   - The dataset's one MPS crash was faer returning NaN factors on the M1 without reporting an error. It is fixed.
   - `Strategy::Auto`'s 350× adder misfire and its 100× Clifford+T misfire are fixed. Auto as a whole is no better on
     average, and the planner no longer relies on it (§4).

## 1. Set-up

The dataset is unchanged from research/simulability/simulability.md: 314 instances in four families (ct, brick, arith, qaoa), with
the request `<Z^{⊗n}>`. Mac timings come from the original sweep: M1 Pro, one thread, 10 s / 1 GiB budget.

**New families (held out, never fitted):**
- `hea:n,D`: `D` layers of `Ry Rz` on every qubit plus a sequential CNOT ladder.
- `qft:n,h`: the QFT, with no final SWAPs, applied to `h` qubits in `|+>` and the rest a random basis state.

That is 36 instances, every engine timed in one Mac session.

**Deterministic MPS data.** For every instance I ran the real exact MPS with its bond trace (VPS), three capped probes
(χ ≤ 8/16/32) and the replay under every bound. This is operation counts, not timing, so VPS load does not matter.

**Mac timing.** All of it under the shared bench lock, in chunks of 1.5–3 min with two single-threaded workers; 1-min
load 3.5–11.5 (peers' builds). Single shots, with no min-of-3: the previous study measured +11 % median single-shot
inflation (±20 % absolute). The end-to-end planner runs are compared with best-engine times from the original sweep, a
different session; on the held-out families everything ran in the same session.

## 2. Predicting MPS cost

### 2.1 The replay (`mps_cost::replay`)
`Mps::apply_gate` has a fixed control flow:
1. A one-qubit gate costs `dl·dr`.
2. A two-qubit gate routes the higher qubit down with SWAPs, applies the gate, and routes back.
3. Every adjacent application first moves the orthogonality centre with thin QRs, which trim bonds to `min(2dl, dr)`.
4. It then forms θ (`2dl·dm·2dr`) and does a thin SVD (`2dl × 2dr`).
5. Toffolis run as their 15-gate Clifford+T decomposition.

The replay walks the same flow and tracks only bond dimensions. Every SVD keeps `min(2dl, 2dr, est)`, where `est` comes
from a `BondSource`:
- `Trace`: the real run.
- `Bound(e)`: a rigorous bound.
- `Probe`: a capped probe, taken where it is below its cap, with the best bound above it.
- `ProbeExtrapolate`: like `Probe`, but above the cap it grows the probe's value by the bound's growth since saturation.

The engine counts the same quantities (`MpsStats`): SVD calls and `Σ m·n·min(m,n)`, QR calls and work, θ products, and
one-qubit work. The work estimate is

`R = log2(8·svd_work + qr_work + mm_work + oneq_work + 1000·(svd_calls + qr_calls))`.

The weights (8, 1000) were chosen by grid search on the oracle (`units_grid` in the report). The 1000 is the fixed
per-call cost of a small faer SVD/QR: most SVDs are tiny, which is why `G·χ³` fails. The replay takes a median 0.7 ms
(VPS, Best bound).

### 2.2 Rigorous bounds on the Schmidt rank (`mps_cost::BondBounds`)
Each bound is maintained gate by gate. Each is an upper bound on the Schmidt rank of the exact state across any
bipartition `A|B`, so the replay bounds the real bond at every SVD step, routed bipartitions included.

| bound | idea | per-gate cost |
|---|---|---|
| `cut` | `min(|A|,|B|)` | – |
| `cross` | time-resolved crossing count; each 2q gate adds its operator-Schmidt bits to every line cut it straddles (the previous study's feature); non-prefix bipartitions via `χ(A∪{q}) ≤ 2χ(A)` | O(n) |
| `stab` | write `|ψ_t> = O_t|S_t>`, with `|S_t>` the stabilizer state of the Clifford part and `O_t` the non-Clifford Pauli rotations, axes pushed forward through every later Clifford. `χ_A ≤ 2^{e_A(S_t)}·OSR(O_t) ≤ 2^{e_A + s_A}`, where `e_A = rank π_A(Stab) − |A|` is the exact stabilizer entanglement and `s_A` counts axes acting on both sides | O(n + m) words |
| `coset` | `|ψ_t> ∈ span{g|S_t> : g ∈ ⟨axes⟩}`; grouping `g_A` by cosets of `π_A(Stab)` (which maps the A-Schmidt span into itself) gives `χ_A ≤ 2^{rank π_A(Stab + ⟨axes⟩) − |A|}` (and the same for B) | rank of ≤ 3n 128-bit vectors |
| `affine` | wires are known constants, affine forms in branching variables, or opaque; the Schmidt rank ≤ the number of distinct A-parts of the support, `2^{rank(affine forms on A) + #opaque on A}`. Toffolis are applied as one permutation (exact on classical controls) | O(n) + rank |
| `best` | the minimum, evaluated cheapest first with early-stopping ranks | |

For Clifford circuits `stab` is the exact Schmidt rank (tested). For arithmetic on classical inputs `affine` is often
exact; for `arith:bits=6,h=0` it is χ = 1, where the crossing count says 32. Writing the coset bound I found two bugs in
the support bound I had copied from the simulability features. The Toffoli rule assumed a constant control means "acts
as CNOT", and the XOR of equal affine forms was set to "constant". Both are fixed here; the old `support_bound` in
simulability.rs still has the Toffoli shortcut, which affects only its tightness claims (open item).

`tests/planner.rs::every_bound_dominates_the_real_bond_at_every_step` checks every bound against the real trace step by
step on 64 circuits: family instances, plus edge-biased random circuits with Toffolis, controlled phases, long-range
SWAPs, `U` and iSWAP.

### 2.3 Results (`fit_v1/stdout.txt`, `mps_predictors.png`)
Mac MPS time (runs ≥ 1 ms, 179 points), `log2 t = a + b·R`. Held out = leave one family out.

| work estimate | slope | in-sample RMSE (log10) | held-out RMSE ct / brick / arith / qaoa | pooled held-out | worst held-out error (decades) |
|---|---|---|---|---|---|
| old `mps_l` (crossing + support, Σχ³) | 0.27 | 0.84 | 0.81 / 1.29 / 0.89 / 0.93 | **0.93** | 2.40 |
| oracle `G·χ_max³` (measured max bond) | 0.39 | 0.55 | 0.69 / 0.61 / 1.53 / 0.61 | 1.01 | 2.40 |
| replay[cross] | 0.31 | 0.82 | 0.73 / 1.30 / 0.90 / 1.01 | 0.93 | 2.24 |
| replay[stab] | 0.34 | 0.81 | 0.78 / 0.78 / 1.26 / 1.05 | 1.00 | 2.83 |
| replay[coset] | 0.33 | 0.83 | 0.89 / 0.79 / 1.00 / 1.05 | 0.94 | 2.27 |
| replay[affine] | 0.23 | 0.87 | 1.47 / 0.99 / 0.27 / 1.11 | 1.09 | 3.02 |
| **replay[best]** | 0.48 | 0.59 | 0.63 / 0.72 / 0.71 / 0.71 | **0.69** | 1.88 |
| replay + probe χ≤8 | 0.55 | 0.47 | 0.65 / 0.60 / 0.43 / 0.39 | 0.53 | 2.60 |
| **replay + probe χ≤16** | 0.62 | 0.38 | 0.53 / 0.45 / 0.31 / 0.28 | **0.41** | 1.89 |
| replay + probe χ≤16, extrapolated | 0.92 | 0.35 | 0.44 / 0.47 / 0.15 / 0.41 | 0.37 | 1.71 |
| replay + probe χ≤32, extrapolated | 0.92 | 0.25 | 0.30 / 0.29 / 0.08 / 0.30 | 0.25 | 1.31 |
| **oracle: replay of the real trace** | 0.83 | 0.07 | 0.07 / 0.06 / 0.10 / 0.08 | **0.08** | 0.37 |

On the families never fitted (models fitted on all 314 instances, then tested on hea and qft), held-out RMSE (worst):

| | hea (11) | qft (11) |
|---|---|---|
| old `mps_l` | 1.00 (1.93) | 0.97 (1.45) |
| replay[best] | 0.57 (1.15) | 0.39 (0.79) |
| probe χ≤16 | 0.26 (0.55) | 0.14 (0.25) |
| oracle | 0.10 (0.15) | 0.08 (0.14) |

Maximum bond (log2 excess over the real trace maximum, 291 runs):
- The best bound is exact on 177 runs, with mean excess 0.82 bits and worst 7.0.
- Single bounds are exact on 34–106 runs (mean excess 2.0–3.5 bits).
- Probe χ≤16 is exact on 233 runs; it is an estimate and can undershoot by up to 2.3 bits.

**What the numbers say.**
- None of the single rigorous bounds beats the old feature on its own. Each one is tight on different families: `affine`
  on arithmetic (0.27), `stab`/`coset` on brickwork (0.78). Their minimum is the first feature that transfers across
  families (0.69, worst error 1.9 decades instead of 2.4).
- The remaining gap to the oracle (0.69 vs 0.08) is bound looseness, not cost modelling. In `mps_predictors.png` brick and
  ring-QAOA sit on the replay[best] line, where the bound is exact; ct and arith fall below it, where the bound
  overestimates the bond. Those are disentangling circuits: Clifford layers that partially undo each other, and adders
  on partially classical registers. The bounds only capture the cancellation the stabilizer or affine structure
  explains.
- The probe closes most of the gap because it measures the trace where it is cheap (χ ≤ 16). Its cost, however, is
  comparable to an exact MPS run on this dataset: median probe time / MPS time 1.3–1.9 on the VPS, since most MPS runs
  never exceed χ = 16. That is why the planner uses it only as an option, in probe-or-solve form (§3).

**Rigour caveat: numerical rank inflation in the engine.** On 10 of the 291 traced runs (101 of ~10⁵ SVD steps) the real
engine kept *more* singular values than the rigorous bound allows. All 10 are ct instances: depth 32 with NN layers, or
random matching. Three are **pure Clifford** (`ct:n=20,L=4,t=0,nn=0`, `ct:n=32,L=32,t=0`), and there the kept ranks
(e.g. 130, 138, 148 where the exact rank is 128) cannot be exact Schmidt ranks: a stabilizer state's Schmidt ranks are
powers of two. Rounding accumulates singular values above the relative cutoff of 1e-14, and they stay there even at a
1e-10 cutoff. This is engine noise, not a bound failure.
`Mps::set_step_caps` caps each SVD at the replayed bound, which only discards weight that cannot belong to the exact
state. Engine `mpsb` in `run_engine_obs` does this:
- On 50 Mac runs ≥ 50 ms it was 1.05× faster (geo; range 0.98–1.18×), with identical values.
- On 7 noisy instances its discarded weight, 2–4·10⁻¹⁰, exceeds the 1e-10 exactness gate, so `mpsb` reports them as
  truncated. That weight is the numerical error the uncapped engine silently carries. Whether to relax the gate for
  capped runs is left open.

## 3. Planner v0 (`src/planner.rs`)

```
plan(circuit, request, config) -> Plan { engine, ranked, features, solved }
  stage 0  Clifford                                   -> Tableau
           state vector predicted < 0.3 ms (n, G only) -> StateVector
  stage 1  O(G·n) features: sv_l, sparse_l, dense_l (rotation-frame d-profile),
           vanishing certificate                   -> Zero if <Z..Z> provably 0
  stage 2  MPS replay with the best bound          if the cheapest stage-1 engine >= 1 ms
           HSF partition (KL)                      if HSF fits and nothing so far < 5 ms
  rank:    argmin_e 2^(a_e + b_e R_e) over applicable {SV, sparse, MPS, HSF, compressed}
  optional probe-or-solve (probe_cap): if MPS is not first, run MPS with χ <= cap for at most
           0.2 x the best predicted time; no truncation -> that IS the exact answer, else
           re-rank with the probe's extrapolated trace
execute_expectation(plan)
  MPS and sparse run speculatively: abort after max(2 ms, 1 x the runner-up's predicted
  time) or on the memory budget, then run the runner-up
  debug_reference: also run the reference SV (n <= 20) and panic on a mismatch
```

- **Constants** (`CostModel::mac_m1`): fitted on all 314 instances (Mac), with the MPS model refitted on the replayed
  work. The other engines keep the published fit.
  - sv −29.10 + 0.933·R
  - sparse −23.13 + 0.842·R
  - **mps −18.71 + 0.475·R**, with R in (8, 1000) units
  - hsf −19.63 + 0.593·R
  - cstate −28.40 + 0.979·R
- **Compressed state** runs exactly as modelled: `CompressedState`, always evolved. It does not use `Strategy::Auto`,
  whose run-time hand-over has unpredictable cost (§4).
- **Pipeline.** `pipeline::simulate` now sends every expectation-value component (after light cone and component split)
  through the planner (`PlanOptions::planner`). `SimOptions::planner_debug` turns on the reference check. Samples and
  amplitudes still use the rule-based dispatch.
- **Tests (`tests/planner.rs`, all passing):**
  - replay == engine counts;
  - bound dominance at every step;
  - stab exact on Clifford circuits;
  - every planned engine **and every forced engine** (SV, sparse, MPS, HSF, compressed) matches the independent audit
    reference state vector (`tests/audit_common`) on 64 circuits × 3 observables, with the debug check on;
  - probe-or-solve exact on both outcomes;
  - speculative abort falls back exactly;
  - the adder case;
  - dataset regret.

## 4. `Strategy::Auto`

The misfire: `arith:bits=12,h=2,reps=1`, Auto 1.47 s against the frame's 4 ms. Auto decided at the very first stage,
with zero observations, using the growth prior of 0.5 bits per span-preserving rotation. The projected Heisenberg cost
exploded, so it handed over to a 25-qubit dense register, while the true term count stayed at 4. The growth meter
ignores everything below 32 terms, so it could never learn otherwise.

**First attempt, rejected: a ski-rental guard.** "Stay in the frame until it has spent the switching price." Mac re-time
in `mac/retime_skirental_*.csv`: it fixed the adders but made exploding brickwork up to 2,400× slower. The switching
price grows with the live term count, because the hand-over evaluation costs about `terms · 2^d`, so the frame never
caught up with it. Lesson: delay has a cost that grows with term explosion.

**Shipped (`AdaptiveOptions::flat_evidence`, on by default):**
- Rotations that leave the term count unchanged count as zero-growth evidence, even below 32 terms.
- Auto keeps exploring until it has seen 8 rotations, but only while a frame rotation is cheaper than a dense one *and*
  the hand-over evaluation is still below the dense evolution cost. Exploring can therefore at most about double the
  cost of switching.

Mac re-time against frame and dense from the same session (`auto_compare.txt`):

| | old Auto | new Auto |
|---|---|---|
| `arith:bits=12,h=2,reps=1` | 1.47 s (349× the best) | **1.7 ms** |
| `ct:n=24,L=16,t=64,nn=1` | 1.35 s (100×) | **13.7 ms** (≈ 1×) |
| `arith:bits=10/8,h=2,reps=1` | 45× / 23× | ≈ 1× |
| `arith:bits=12,h=4/6,reps=1` | 314× / 180× | **339× / 262× (not fixed)** |
| all 293–300 instances, geo vs min(frame, dense) | 1.245 | 1.327 |
| median | 1.06 | 1.22 |

The targeted misfires are fixed, but Auto as a whole is *not* better: exploration costs about 20 % at the median. The
h = 4/6 adders still misfire, because their terms grow past the exploration bound before the frame finishes. A robust
Auto needs a forecast of the frame's peak term count, which is still open question 2 of the simulability study. So the
planner does not use Auto at all. On the adder the planner picks the measured winner, sparse at 63 µs; the pipeline's
expectation path used to call Auto and now calls the planner.

## 5. Planner evaluation

### 5.1 Decision quality, leave one family out (Python, `fit_v1/planner_report.json`, `regret_cdf.png`)
The same protocol as the paper: fit on three families, test on the fourth. State engines, request `<Z^{⊗n}>`. A censored
choice counts as 2× the timeout. "+spec" is the speculative-abort policy simulated with the LOFO predictions.

| MPS feature | top-1 | geo regret | within 2× | geo ε-regret | worst | +spec geo / worst | per family top-1 / geo / worst: ct, brick, arith, qaoa |
|---|---|---|---|---|---|---|---|
| old `mps_l` (paper) | 85.0 % | 1.247 | 91.4 % | 1.217 | 161.9 | 1.115 / 9.2 | 93/1.08/21 · 70/1.51/59 · 88/1.07/2.1 · 71/1.96/162 |
| **replay[best] (shipped)** | **88.2 %** | **1.092** | **95.2 %** | 1.072 | **12.9** | 1.083 / 9.2 | 94/1.04/4.6 · 74/1.22/12.9 · 88/1.07/2.1 · 83/1.17/10.9 |
| replay + probe χ≤16 | 93.0 % | 1.038 | 97.8 % | 1.028 | 4.3 | 1.046 / 9.2 | 97/1.01/2.1 · 78/1.14/4.3 · 98/1.01/1.7 · 90/1.06/3.5 |
| oracle trace | 95.5 % | 1.025 | 99.0 % | 1.022 | 8.5 | – | 98/1.00/1.2 · 86/1.13/8.5 · 98/1.01/1.7 · 94/1.01/1.6 |

All of the paper's large failures disappear with the replayed feature: QAOA at p = 3 (162×, 131×, 87×) and n = 24
brickwork with random matching (59×). What remains:
- brick D = 12–16, where HSF beats MPS by 8–13× and the bound is exact; this is an MPS-vs-HSF modelling issue;
- QAOA p = 1 on random graphs, where MPS is chosen and cstate is best (11×).

Even the oracle feature leaves brick at 86 %. Its errors are HSF and SV mispredictions, which no MPS feature can fix.

### 5.2 The shipped Rust planner on the dataset (`tests/planner.rs`)
In-sample constants, every feature computed: top-1 88.2 %, geo regret 1.087, worst 12.9×
(`brick:n=20,D=16` → MPS, HSF best). The shipped, staged configuration gives geo ε-regret 1.074. The test asserts geo
≤ 1.15, top-1 ≥ 85 %, worst ≤ 20, and staged ε-regret ≤ 1.15. The published held-out result was 1.25 / 85 % / 161×.

### 5.3 End to end on the Mac (`eval_final/`; `planx` = plan + execute, measured in-process)
Planning takes a median 0.45 ms (p90 5.7 ms, max 19 ms).

| | n | geo regret | within 2× | geo ε-regret | worst | choice top-1 | choice geo regret |
|---|---|---|---|---|---|---|---|
| dataset, all | 314 | 2.52 | 41 % | **1.34** | 31× | 86.9 % | 1.11 |
| dataset, best ≥ 1 ms | 91 | 1.54 | 76 % | | 8.5× | | |
| dataset, best ≥ 10 ms | 45 | 1.33 | 82 % | | 8.5× | | |
| dataset, best ≥ 0.1 s | 21 | **1.20** | 90 % | | 8.5× | | |
| hea (held out) | 24 | 2.07 | 42 % | 1.41 | 6.1× | **87.5 %** | **1.03** (worst 1.9×) |
| qft (held out) | 12 | 1.82 | 50 % | 1.49 | 2.7× | **83 %** | **1.10** (worst 2.0×) |

(The first, unstaged planner, `eval_v0_unstaged`, had end-to-end geo 14.8 and ε-regret 3.8. It computed every feature,
including a 13 ms KL partition for a circuit whose best engine takes 0.25 ms. Staging fixed that.)

Where the end-to-end regret comes from, in order:
1. **Planning overhead on sub-millisecond instances.** The 30× worst cases are 25 µs sparse runs behind 0.5–1 ms of
   planning.
2. **Speculation costs.** Aborting a mispredicted MPS run costs up to 1× the runner-up, by design: QAOA n = 24 p = 1
   ran 3×. It can also abort the true winner when the runner-up's prediction is too optimistic: hea n = 24, D = 8 ran
   1.8×.
3. **Choice misses**, mostly MPS vs HSF on deep brickwork. `brick:n=26,D=16` timed out after choosing the state vector,
   which took more than 10 s at n = 26, while MPS took 2.3 s.

Probe-or-solve (`planp`) does not change the end-to-end numbers on this dataset (ε-regret 1.37 vs 1.34): the probe's
budget is 0.2× the best predicted time, so it rarely runs long enough to matter.

## 6. What is new, and what is known
- Known: MPS cost ~ poly·χ³ and χ ≤ 2^{crossing gates} (Vidal; Jozsa; Markov–Shi). Stabilizer entanglement from the
  rank of the projected stabilizer group (Fattal et al. 2004; Hamma et al. 2005). Stabilizer-augmented MPS (CAMPS,
  arXiv:2412.17209) and stabilizer tensor networks (arXiv:2403.08724) *exploit* Clifford structure in the simulator.
- New here, as far as I can tell:
  1. The observation that exact MPS time is determined by the replayed operation count of its bond trace, at 0.08
     decades across families. The cost model is exact; only the trace is unknown.
  2. The `stab`/`coset` Schmidt-rank bounds for "stabilizer state plus pushed-forward Pauli rotations", used as
     planner-time features, together with the per-cut affine support bound, minimised over all of them. This is the
     first rigorous MPS feature that transfers across families: held-out RMSE 0.93 → 0.69, worst decision regret
     162× → 12.9×.
  3. Probe-or-solve: a capped MPS run is either the exact answer (no truncation) or the best available estimate of the
     trace.
  4. Bound-capped exact MPS removes numerical rank inflation that the relative cutoff alone lets through.

## 7. Open
1. **Tighter cheap traces.** The best bound misses cancellation that neither stabilizer nor affine structure explains
   (ct with partially self-inverting layers, adders with h ≥ 4). Candidates:
   - a bound on rotations after merging those whose axes commute with all later straddling axes;
   - a "stabilizer + magic" coset bound that tracks the actual coefficients for small rotation sets;
   - learning the probe's cap from the bound's saturation point.
2. **Planning overhead.** The KL partition and the replay cost 1–20 ms. For instances in the 1–10 ms range, an
   incremental replay (only the cuts a gate touches), or a cheaper HSF path estimate, would roughly halve the
   end-to-end regret.
3. **Speculation tuning.** κ = 1 is the ski-rental optimum for one misprediction, but the runner-up's prediction is
   itself noisy. Calibrate the deadline on the error distribution of the runner-up model.
4. **Samples and amplitudes requests.** The planner is wired into expectation values only. The sampling path still uses
   the rule-based dispatch, and amplitudes need the global phase, which excludes the compressed state.
5. **Auto.** It needs a forecast of the frame's peak term count. Until then, use the planner, which avoids Auto.
6. **The `mpsb` exactness gate.** Should a bound-capped run's discarded weight, which is provably numerical noise, be held
   to the 1e-10 gate?
7. ~~**simulability.rs `support_bound`.**~~ Fixed on exp/r4-audit2 (audit.md §16): the constant-control Toffoli now
   makes the target opaque; a counterexample and a fuzz test are in `tests/audit_r4b.rs`. No dataset feature value changed.

## 8. Reproduce
```
cargo build --release --example simulability
B=target/release/examples/simulability
$B mpscost  'qaoa:n=20,p=2,deg=3,nn=1' 1        # replayed cost under every bound
$B mpstrace 'ct:n=24,L=8,t=16,nn=1' 1 8,16,32   # real run: counts, replay check, bound check, probes
$B run planx 'hea:n=20,D=4' 1                   # planner end to end (also plan, planp, mpsb)
cd research/data/planner
python3 collect.py $B mpsdata_vps.jsonl 3       # GRIDS=hea,qft for the held-out families
python3 fit_planner.py fit_v1 mpsdata_vps.jsonl
PLAN_TAG=plan4 NEW_TAG=new4 python3 eval_planner.py eval_final
python3 compare_auto.py mac/retime_skirental_0.csv mac/retime_skirental_1.csv mac/auto2_0.csv mac/auto2_1.csv
cargo test --release --test planner
```
