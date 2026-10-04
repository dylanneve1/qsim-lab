# Planner v2: samples, amplitudes, cheaper planning, and the adder Auto misfire

Branch `exp/planner-v2` (from main e7e102d). Author: qsim-planner-sampling agent (round 5, 4 Oct 2026).
Builds on research/planner.md (Planner v1, expectation values only) and research/simulability.md.

Code:
- `src/planner.rs`: requests `Samples(shots)` and `Amplitudes(count)`; per-engine read-out models; evolution-only
  models; tiered planning with value-of-information gates; HSF priced on the line split; plan cache; `Prepared`
  (evolve once, read out many ways); `execute_samples`, `execute_amplitudes`. `PlannerConfig::v1()` reproduces v1.
- `src/compile/plan.rs`, `src/pipeline.rs`: terminal samples (every component that would need a state vector or the
  compressed state, ≤ 128 qubits) and amplitudes (components ≤ 63 qubits) now go through the planner;
  `Backend::Planned(engine)` reports what ran.
- `src/statevector.rs`: `sorted_uniforms`, O(shots) sorted uniforms (normalised exponential spacings) in
  `StateVector::sample`; `src/sparse.rs`: `SparseState::sample`; `src/mps.rs`: buffer-reusing `sample` and
  `amplitude` (bit-identical), `canonicalize`.
- `src/adaptive.rs`: `Strategy::Auto` explores past a switch decision with a ski-rental budget and can restart densely
  (`explore_frac`, `AdaptiveReport::restarted`); `src/pauli_frame.rs`: rotations with < 2048 live terms run on the
  calling thread.
- `src/mps_cost.rs`: `ReplayCost::final_bonds`; `src/simulability.rs`: `hsf_split_features`, `auto0` engine.
- `tests/planner_v2.rs` (new), `tests/planner.rs`, `tests/pipeline.rs` (updated).
- `examples/planner_v2.rs`: `req` (evolve once, time every read-out), `plan VARIANT REQ` (end to end), `feat`
  (features, per-tier timings, decisions for a range of `voi`), `cachedemo`.

Data and scripts: `research/data/planner-v2/` (all Mac, M1 Pro, one thread, bench lock):
- `mac/req.jsonl`: 1770 (instance, engine) read-out runs.
- `mac/hsfamp.jsonl`: 81 HSF amplitude-only runs.
- `mac/feat_mac.jsonl`: features and per-tier timings.
- `mac/e2e*.jsonl`: end-to-end sessions.
- `mac/autoab/`: Auto A/B.
- `mac/auto3/`: Auto, first re-time.
- Scripts: `collect_req.py`, `fit_v2.py`, `readout_accuracy.py`, `tune_voi.py`, `gen_e2e_jobs.py`, `eval_oracle.py`,
  `eval_v2.py`, `compare_autoab.py`, `compare_auto3.py`, `run_chunks.sh`.

## Headline

1. **The planner now covers all three requests.** Each engine's cost for a request is its evolution plus a read-out
   term. The read-out terms are near-exact operation counts with 1–2 fitted constants each:
   - state-vector sampling: `2^n` + shots;
   - sparse sampling: `2^sup` + shots;
   - MPS perfect sampling: `Σ 2χ_lχ_r` + n per shot, plus the canonical form;
   - compressed sampler: `2^d(d+1)` + shots·(d+n);
   - tableau sampling;
   - amplitudes: look-ups, MPS contractions, HSF path sums.

   Leave-one-family-out over 6 families and 350 instances (choice level, geometric ε-regret):

   | request | old hand rule | v1 ranking reused | **v2** |
   |---|---|---|---|
   | expectation | — (v1: 1.076) | 1.076 | **1.077** |
   | 1 sample | 5.87 | 1.137 | **1.069** |
   | 1 000 samples | 4.94 | 1.137 | **1.093** |
   | 100 000 samples | 2.77 | 1.822 | **1.089** |
   | 1 amplitude | 49.5 | 1.676 | **1.108** |
   | 1 000 amplitudes | 41.0 | 1.699 | **1.100** |

   Worst cases drop from 10^4–10^5× (old rules: a dense state vector or compressed register where a 50 µs sparse run
   suffices, or no answer at all for 32-qubit amplitudes) to 7–21×.
2. **Planning got cheaper.** Every expensive feature is now gated by a value-of-information test: compute it only if
   the engine it informs could beat the best prediction so far by more than `voi` × the feature's predicted cost,
   using lower bounds available from one O(gates) pass. HSF is first priced on the plain line split in O(gates); the
   4 ms Kernighan–Lin partition runs only when HSF could still win and is not already first.
   - Median planning time for expectations, Mac, same session as v1: 0.47 ms → 0.17 ms; p90 7.1 ms → 1.9 ms.
   - **End-to-end ε-regret for expectations against an in-session oracle: v1 1.32 → v2 1.15** (§4).
   - A plan cache keyed on structure and Clifford class reuses decisions across parameter sweeps: 6 ms → 6 µs per
     plan.
3. **The 12-bit adder Auto misfire is fixed.** `Strategy::Auto` now explores past a switch decision for 30 % of
   min(switch now, dense restart), then hands over or restarts densely. Same-session Mac A/B:
   - `arith:bits=12,h=4`: 2.12 s → 4.7 ms (453×);
   - `h=6`: 4.03 s → 26 ms (156×);
   - 10-bit adders: 52× faster;
   - over all 314 instances Auto is 0.95× (geo) the round-4 Auto, 0.84× on runs ≥ 1 ms;
   - worst regression: 6× on 1–10 ms brickwork.

   A latent cost surfaced along the way: the frame engine's two thread-pool hand-offs per rotation made 4-term
   Heisenberg sweeps take 25–120 ms under load. They are now skipped below 2048 live terms.
4. **Side results.**
   - O(shots) state-vector and sparse sampling: exponential spacings instead of a sort.
   - 2× faster MPS sampling.
   - The MPS read-out model is exact up to the bond profile: 0.08 decades with the real bonds, against 0.53 with
     the bound-predicted bonds. As in v1, the bond trace, not the cost model, is what limits it.

## 1. Set-up

**Instances.** The 314 instances of the simulability dataset (ct, brick, arith, qaoa), plus the 36 held-out hea/qft
instances of planner.md: 350 instances, 10–32 qubits.

**Requests.**
- `e`: `<Z^{⊗n}>`.
- `s1`, `s1k`, `s100k`: 1, 1000 and 100 000 samples of all qubits.
- `a1`, `a1k`: 1 and 1000 amplitudes of random basis states, global phase included.

**Engines.**
- Sampling: state vector (blocked evolve, then sorted-uniform sampling); sparse; MPS (exact, then perfect sampling
  from the left-canonical form); HSF (full output, then state-vector sampling); compressed state
  (`CompressedState::sampler`); tableau (Clifford circuits only).
- Amplitudes: state vector, sparse, MPS (`<x|ψ>` contraction), HSF (`amplitudes(xs)`, one path sum, no `2^n`
  output). The compressed state and the tableau drop the global phase, so they are not candidates for amplitudes.
- Pauli-path marginals stay rule-based in the pipeline (few measured qubits). They are not part of this dataset, which
  measures every qubit.

**Read-out session** (`collect_req.py` → `mac/req.jsonl`):
- One process per (instance, engine). It evolves once, then times every read-out separately: `e`, `a1`, `a1k`, `prep`
  (MPS canonical form, HSF full output, compressed sampler build), then `s1`, `s1k`, `s100k`.
- The total for a request is evolve + its read-out (+ prep for samples). A request-level truth comes from one
  evolution per engine, which keeps the session to about 8 minutes of locked time.
- Read-outs have a 6 s deadline. Censored runs count as 20 s, as in the paper (2 × the 10 s timeout).
- (instance, engine) pairs that the round-4 sweep measured as censored or slower than 4 s were not re-run. They count
  as censored too. The HSF *amplitude* path is a different computation from the full output, so those 81 instances
  were timed separately (`req hsfamp`, 8 s process timeout: 18 censored).
- Mac 1-min load during the read-out chunks: 6.7–12 (another agent's 6-worker LER campaign until 12:35, paused for the
  last two chunks: 4.1–12.2).

**Validation.** Leave one family out over six families. Constants are fitted on five, and the held-out family is
scored with them (`fit_v2.py`). Scoring follows planner.md:
- regret = measured total of the choice / best measured total;
- ε-regret adds 1 ms to both;
- censored = 20 s.

## 2. Cost models (`fit_v2.py`, `readout_accuracy.py`)

`total_e(request) = evolve_e + readout_e(request)`. Two sets of evolution models are fitted on this session, both of
the form `log2 t = a + b R_e`, with the v1 features (`R`: SV `n + log G`, sparse `log G + sup`, MPS replayed work,
HSF path model, compressed `log Σ 2^{d_j}`):
- expectation: the whole run;
- state: evolution alone; for HSF, set-up + full output.

The slopes are close to v1's (SV 0.92, compressed 0.99, sparse 0.77, MPS 0.48, HSF 0.60).

Read-out models, linear in operation counts, fitted on the read-out times alone (all ≥ 20 µs). RMSE is in log10;
"LOFO" refits without the held-out family:

| component | units | coefficients (Mac) | n | RMSE in-sample / LOFO |
|---|---|---|---|---|
| SV / HSF-output sampling | `2^n`, shots | 1.39 ns, 35.7 ns | 1601 | 0.18 / 0.19 |
| sparse sampling | `2^sup` (bound), shots | 4.7 ns, 31.9 ns | 531 | 0.32 / 0.40 |
| MPS sampling (per shot) | `Σ2χ_lχ_r`, n | 0.11 ns, 52 ns | 642 | 0.53 / 0.60 (**0.08** with the real bonds) |
| MPS canonical form | `Σ2χ_lχ_r min(2χ_r,χ_l)`, n | 0.03 ns, 1.1 µs | 216 | 0.78 / 0.95 (0.28 real) |
| MPS amplitude | `Σχ_lχ_r`, n | 0.16 ns, 33 ns | 363 | 0.62 / 0.69 (0.13 real) |
| compressed sampler build | `2^d(d+1) + n²⌈n/64⌉` | 5.2 ns | 311 | 0.55 / 0.66 |
| compressed sampling | shots·(d+n)⌈n/64⌉ | 7.2 ns | 623 | 0.24 / 0.25 |
| tableau sampling | `n²⌈n/64⌉`, shots·n(1+⌈n/64⌉) | 147 ns, 6.6 ns | 60 | 0.15 / – |
| HSF amplitudes (set-up + paths) | `G n`, `2^keff G 2^max(nA,nB)` | 1.6 µs, 0.011 ns | 664 | 0.30 / 0.35 |
| SV / sparse look-up | per amplitude | 12 ns / 11 ns | | |

Notes:
- **MPS.** The per-site term (≈ 50 ns per site per shot: loop, RNG draw, normalisation) was missing in the first fit.
  The first fit used `c · Σ2χ_lχ_r` only and under-predicted bond-1 sampling 10×. Adding the term moved
  100 000-sample LOFO regret from 1.33 to 1.09. With the real final bonds the model is an exact operation count
  (0.08 decades). With the bonds the planner can see (the replay under the rigorous bound) it is 0.53 decades. The
  bound is loose on disentangling circuits, as v1 found for the evolution.
- **HSF amplitudes.** A log-linear fit on `R_amp` (slope 0.39) under-predicted large path counts by orders of
  magnitude. The planner chose HSF on 32-qubit Clifford+T circuits that then ran past 8 s (worst LOFO regret 3200×).
  The two-term linear model has slope 1 in the path count, and it removes those.
- **Request size matters.** At 100 000 shots MPS loses many instances it wins at 1000, because perfect sampling costs
  `n χ²` per shot. The v1 ranking reused for samples (evolution only) is therefore 1.82 at 100 000 shots, against 1.14
  at 1000.

### Decision quality, leave one family out (choice level, every feature computed)

| request | rule | top-1 | geo regret | ≤ 2× | geo ε | worst |
|---|---|---|---|---|---|---|
| e | v1 (as shipped) | 85.9 % | 1.132 | 94.3 % | 1.076 | 23.2 |
| e | **v2 LOFO** | 86.8 % | 1.099 | 94.5 % | 1.077 | 15.3 |
| s1 | old rule | 43.7 % | 10.6 | 54.6 % | 5.87 | 4·10⁵ |
| s1 | v1 reused | 79.6 % | 1.227 | 91.7 % | 1.137 | 35.3 |
| s1 | **v2 LOFO** | 84.2 % | 1.097 | 95.4 % | 1.069 | 15.6 |
| s1k | old rule | 40.5 % | 7.24 | 56.6 % | 4.94 | 3·10⁵ |
| s1k | v1 reused | 75.9 % | 1.191 | 91.7 % | 1.137 | 21.2 |
| s1k | **v2 LOFO** | 82.5 % | 1.118 | 94.3 % | 1.093 | 21.2 |
| s100k | old rule | 40.0 % | 2.92 | 57.7 % | 2.77 | 7602 |
| s100k | v1 reused | 59.1 % | 1.869 | 75.9 % | 1.822 | 363 |
| s100k | **v2 LOFO** | 81.2 % | 1.095 | 96.5 % | 1.089 | 7.0 |
| a1 | old rule (SV) | 13.8 % | 118 | 20.2 % | 49.5 | 1.6·10⁵ |
| a1 | v1 reused | 74.6 % | 1.821 | 83.3 % | 1.676 | 9·10⁴ |
| a1 | **v2 LOFO** | 86.2 % | 1.134 | 93.4 % | 1.108 | 15.8 |
| a1k | old rule (SV) | 15.3 % | 76.4 | 20.7 % | 41.0 | 7·10⁴ |
| a1k | v1 reused | 71.5 % | 1.816 | 82.7 % | 1.699 | 3·10⁴ |
| a1k | **v2 LOFO** | 86.7 % | 1.125 | 93.1 % | 1.100 | 15.0 |

Per held-out family (v2, top-1 / geo / worst):
- 100 000 samples: ct 0.79/1.15/7.0, brick 0.90/1.05/2.1, arith 0.90/1.02/1.5, qaoa 0.71/1.11/2.2,
  hea 0.96/1.02/1.5, qft 0.42/1.10/1.5.
- 1000 amplitudes: ct 0.90/1.12/15, brick 0.86/1.17/9.7, arith 0.83/1.10/3.3, qaoa 0.90/1.07/2.7, hea 0.88/1.09/3.3,
  qft 0.58/1.46/9.3.

Remaining large misses:
- MPS vs HSF on deep brickwork (`brick:n=20,D=16`: MPS chosen, HSF 15× faster), the v1 failure;
- MPS's bound-predicted read-out on Clifford+T (`ct:n=24,L=16,t=48/64`: compressed or HSF chosen, MPS 10–25×
  faster);
- censoring artefacts. `ct:n=24,L=32,t=48` amplitudes "15×" is a state vector skipped as > 4 s in the prior sweep,
  so it is scored as 20 s.

"Old rule" for expectations is not a baseline (v1 already planned them). Its row is omitted from the headline.

## 3. Tiered planning and the cache

v1 staged by absolute thresholds:
- the state vector without features below 0.3 ms;
- the MPS replay above 1 ms;
- KL above 5 ms.

It spent a median 0.47 ms planning, p90 7 ms (this session), and dominated sub-millisecond instances.

v2 (`plan_v2`) works in tiers:

| tier | what | Mac cost (median / p90) | computed when |
|---|---|---|---|
| 0 | one pass: n, G, Clifford?, branching gates, non-Clifford rotations (≥ merged), MPS adjacent steps, structural hash | 4 µs / 14 µs | always |
| 1a | affine support bound (sparse) | 21 µs / 59 µs | `best − lb_sparse > voi · c_1a` |
| cert | vanishing certificate (if requested) | 73 µs / 251 µs | Clifford, or `best > voi · c_1b` |
| 1b | rotation-frame d-profile (compressed state) | 83 µs / 270 µs | `best − lb_cstate > voi · c_1b` |
| 2 | MPS replay, best rigorous bound | 244 µs / 1.2 ms | `best − lb_mps > voi · c_2` |
| 3a | HSF on the line split `[0,n/2)` (engine then runs on it) | O(G), ≈ 1a | `best − lb_hsf > voi · c_3a` |
| 3b | HSF Kernighan–Lin partition | 4.6 ms / 11.9 ms | HSF not first and `best − lb_hsf > voi · c_3b` |

- `best` is the lowest predicted total so far. Tier-0 upper bounds: sparse with `sup ≤ min(n, branching)`, compressed
  with `d_j ≤ min(n, j)`, the state vector exactly.
- `lb_e` is the engine's model at its smallest possible work: support 1, `d = 0`, every MPS bond 1 (SVD calls ≥
  adjacent steps), a balanced HSF partition without paths. Its read-out is included, so a large common read-out (the
  100 000-shot sort) does not trigger features that can only change the evolution.
- `c_*` are the fitted tier costs (`FeatureCost`, RMSE 0.12–0.37 decades).
- `voi` was tuned offline on the read-out session (`tune_voi.py`): the decision for each `voi` (deterministic), its
  planning cost from the Mac per-tier timings, and the chosen engine's measured time, with no load confound. One
  scalar per request kind, in-sample. The estimated end-to-end geo ε-regret:

  | voi | e | s1k | s100k | a1 | a1k |
  |---|---|---|---|---|---|
  | 1 | 1.161 | 1.138 | 1.112 | 1.170 | 1.137 |
  | 4 | 1.142 | 1.123 | 1.091 | **1.163** | **1.135** |
  | **8** | **1.136** | **1.109** | **1.089** | 1.184 | 1.129 |
  | 32 | 1.157 | 1.108 | 1.091 | 1.277 | 1.235 |
  | ∞ (tier 0 only) | 3.76 | 4.50 | 2.21 | 26.0 | 21.5 |

  Shipped: `voi = 8` for expectations and samples, `voi_amplitudes = 4`. Pricing HSF on the line split first lowered
  the amplitude estimates from 1.24 to 1.14–1.16, because the 4 ms KL partition is then rarely needed. The choice part
  alone is 1.07–1.10. Planning is the remaining 4–6 %.
- **Cache.** `cache_key` = structural hash (gate kinds, qubits, Clifford class of every angle: multiple of π/2,
  multiple of π) + request kind and log2 size bucket + config fingerprint + observable.
  - It reuses engine *choices*, never values. Probe-solved plans and certificate (Zero) plans are not cached: the
    certificate can depend on rotation merging, which depends on the angles.
  - Parameter sweeps (`cachedemo`, VPS):
    - QAOA n=20 p=2, expectation, 20 angle sets: 6.0 ms → 6 µs per plan, 19/20 hits.
    - hea n=20 D=8, 1000 samples: 3.1 ms → 14 µs.
  - Exactness does not depend on the cache: any engine choice is exact.

## 4. End to end on the Mac

`planner_v2 plan VARIANT REQ`: plan + execute in process, speculation on, cache off (every instance is new),
certificate off (state engines, as `planx`).

**Comparing against the read-out session is load-confounded.** The first end-to-end session (`mac/e2e.jsonl`) ran at
Mac load 15–49 (other agents' builds, `dem_distance`, `color_search`, LER workers, none under the lock). Against the
best engine of the read-out session (load 4–12) even v1 looks 1.55 and runs ≥ 0.1 s look 1.7×, so absolute numbers
from it are not meaningful.

**In-session oracle.** For every instance and request the planner run is paired with the *oracle*: the engine
measured best in the read-out session, forced, with no planning and no speculation, through the same execution path.
The two run side by side on the two workers (`gen_e2e_jobs.py`, `eval_oracle.py`), so they share the load.
ε-regret = (planner + 1 ms) / (oracle + 1 ms).

Session 2 (`mac/e2e2.jsonl`, v2 before HSF line-split pricing and the n < cap HSF fix; load 4–21):

| request | variant | n | geo regret | ≤ 2× | **geo ε-regret** | plan median / p90 | aborts |
|---|---|---|---|---|---|---|---|
| e | v1 | 348 | 2.33 | 44 % | **1.323** | 0.42 / 6.3 ms | 9 |
| e | v2 | 348 | 1.44 | 83 % | **1.149** | 0.17 / 1.9 ms | 9 |
| s1k | v2 | 348 | 1.24 | 94 % | **1.126** | 0.19 / 3.3 ms | 9 |
| s100k | v2 | 345 | 1.14 | 95 % | **1.133** | 0.10 / 5.8 ms | 2 |
| a1k | v2 | 347 | 1.38 | 88 % | **1.254** | 0.22 / 6.9 ms | 2 |

FINAL_SESSION_TABLE

Where the remaining end-to-end regret comes from (session 2, expectations):
1. Runs where the oracle takes ≥ 0.1 s are about 1.4× for both v1 and v2. The planner's choice is the oracle's on 15
   of 21. The rest:
   - speculation: MPS aborted for a runner-up whose prediction was too optimistic (`brick:n=24,D=16`, `hea:n=24,D=8`,
     ×1.6–1.7);
   - one case where the runner-up could not run (`brick:n=26,D=16`: MPS, the true winner at 2.2 s, was aborted for
     HSF, which then hit the 1 GiB budget, and the state vector took 22 s). HSF's full output plus its blocks does not
     fit at `n = cap`; the applicability check is now `n < cap`.
2. Sub-millisecond instances: planning overhead (median 0.17 ms) is now below the 1 ms ε, against v1's 0.42 ms.

## 5. `Strategy::Auto` and the 12-bit adders

**Why it misfired.**
- `arith:bits=12,h=4/6`: at stage 122 of 128 the frame holds 16–32 terms. The growth meter's pessimistic prior
  projects a term explosion, and the hand-over to a 25-qubit dense register looks cheaper (2–5 s).
- In reality the terms grow only to 4^(h+1) (1024 / 16384): the frame finishes in 10–30 ms.
- The round-4 exploration guard (`t · 2^d ≤ Σ 2^{d_j}`) is already false at t = 16 on a 2^25 register, so it never
  explored.
- Forecasting the peak term count is the open problem planner.md named, and it remains open.

**What fixes it without a forecast: ski rental with a fixed restart price.**
- When `auto_decide` first says "switch", Auto keeps sweeping for `explore_frac` (0.3) × min(cost of switching now,
  cost of a dense run from scratch `D`).
- If the frame has not finished by then, it hands over if that is still cheaper than `D`. Otherwise it restarts on
  the dense register from scratch (`AdaptiveReport::restarted`), abandoning the frame.
- The restart price is fixed, unlike the hand-over price, which grows with the live terms. That growth is why
  round 4's ski-rental guard made exploding brickwork 2400× slower. The loss is therefore bounded by about 1.3·D plus
  what the frame spent before the decision.

**Mac A/B in one session** (`compare_autoab.py`; `auto` vs `auto0` = `explore_frac 0`, same binary, back to back per
instance; load 17–49, so treat ms-scale ratios with care):

| | auto0 (round-4 logic) | **auto (v2)** |
|---|---|---|
| `arith:bits=12,h=4,reps=1` | 2.12 s | **4.7 ms** |
| `arith:bits=12,h=6,reps=1` | 4.03 s | **26 ms** |
| `arith:bits=10,h=4,reps=1` | 0.145 s | **2.8 ms** |
| `arith:bits=8,h=4,reps=1` | 8.7 ms | **1.3 ms** |
| auto/auto0, all 314 (geo) | | **0.951** (median 1.007) |
| auto/auto0, runs ≥ 1 ms (126) | | **0.841** |
| worst regression | | 6.2× (`brick:n=12,D=8`, 1.5 → 9 ms) |
| vs best(frame, dense) of the round-4 session, best ≥ 1 ms | geo 1.86, max 423 | geo 1.51, max 58 |

The first re-time (`mac/auto3`, before the frame-store change, different session, `compare_auto3.py`) showed two
"regressions" on `arith:bits=6,h=0`: 0.85 ms → 48 ms. The switch point and term visits are identical. Re-running on
the VPS showed that the *frame engine itself* takes 25–120 ms for 90 term visits, with 167 voluntary context switches.

The cause: each rotation hands its 64 shards to the rayon pool twice, even with 4 live terms and
`RAYON_NUM_THREADS=1`. Under load every hand-off costs scheduler latency. Shards are now processed on the calling
thread below 2048 live terms. This is exact: the shards are independent.

That latency also explains part of Auto's run-to-run noise on small circuits. The A/B above is with the fix, for both
arms.

## 6. Exactness

`tests/planner_v2.rs`, all fixed seeds, so deterministic:
- **Every sampling engine** (state vector, sparse, MPS, HSF, compressed sampler, tableau on Clifford circuits) on 44
  circuits:
  - families, plus random circuits with U/iSWAP/Sx/Toffoli/controlled phase;
  - the engine's exact probabilities or amplitudes are checked against the independent audit state vector (MPS/HSF/SV
    amplitudes ≤ 1e-7, sparse probabilities ≤ 1e-10);
  - 20 000 drawn samples pass a chi-square test against the reference distribution (7-sd bound; an outcome of
    reference probability < 1e-14 fails immediately).
- **Every amplitude engine** (state vector, sparse, MPS, HSF), forced and planned, returns `<x|ψ>` *with the global
  phase* (≤ 1e-7). Compressed state and tableau refuse.
- **Pipeline**: `simulate` amplitudes and terminal samples of every corpus circuit are exact (chi-square as above).
  The pipeline test that expected `Backend::Adaptive` now accepts `Backend::Planned(_)`.
- **Speculative sampling**: a forced MPS with a tiny deadline is aborted, and the fallback's samples pass the
  chi-square test.
- **Tiering and the cache never change a result.** `voi = 0`, `voi = 8` and cached plans give the reference values. A
  cached plan for one angle set, executed on another, returns the new circuit's own value.
- **`sorted_uniforms`**: sorted; Kolmogorov–Smirnov against U[0, total) at p ≈ 0.001; median order statistic.
- `tests/planner.rs` passes with `PlannerConfig::v1()`:
  - v1 reproduced: top-1 88.2 %, geo 1.087, worst 12.9×;
  - the tiered v2 choice on the round-4 sweep: geo ε 1.074.
- Also passing: adaptive (incl. the old-behaviour check with `explore_frac: 0`), pauli_frame, pipeline, compile,
  repeat, audit_repeat, simulability.

Caveats:
- With speculation on, an abort makes the samples come from a different (equally exact) engine, so a seed reproduces
  the samples only when no abort happens. `speculate: 0` restores reproducibility.
- The planner's `debug_reference` checks expectations and amplitudes, not samples (the tests do).

## 7. What is new

- Request-aware engine selection for *samples and amplitudes* as `evolve + read-out` with near-exact read-out operation
  counts. The 100 000-shot results show that a request-blind planner is wrong (1.82 → 1.09).
- Value-of-information tiering with *lower* bounds from one O(G) pass, and pricing HSF on the line split before
  paying for KL. Together they cut planning without losing choice quality: choice ε 1.07–1.10, planning ≈ 0.05–0.14 ms
  median.
- A ski-rental Auto with a *fixed* restart price, instead of the growing hand-over price that sank round 4's attempt.

## 8. Open

1. **End-to-end regret on mid-size runs (1 ms – 1 s) is about 1.3×**, versus about 1.08 at choice level. The gap is
   speculation (aborting the true winner when the runner-up's prediction is optimistic) and load noise.
   - Calibrate the deadline on the runner-up model's error distribution (planner.md open item 3).
   - Never speculate *toward* an engine that cannot be aborted (HSF, SV) when its prediction is uncertain.
2. **Re-plan after the evolution.** MPS read-out with the *real* bonds is an exact operation count (0.08 decades)
   once the state exists. A sampling request could re-decide after the MPS evolution (keep it, or hand the shots to
   another engine) at no planning cost.
3. **Bond prediction** remains the weak link for MPS (evolution and read-out), as in v1.
4. **HSF cannot be aborted.** A cancellation flag in `run_paths` would let it be speculated like MPS and sparse.
5. **Pauli-path marginals** (few measured qubits) and **mid-circuit-measurement** components are still rule-based.
6. **Censoring.** The read-out session skipped (instance, engine) pairs that the round-4 sweep measured as > 4 s. A
   few "worst cases" (≈ 15×) are such artefacts, where the skipped engine probably takes 4–10 s.
7. **Auto's exploration budget** costs up to 6× on 1–10 ms brickwork where the frame explodes within one or two
   stages, because the budget is checked between stages. A per-stage term-count check would bound it.
8. **The cache** keys on exact structure. Near-identical structures (one gate more) miss.

## 9. Reproduce

```
cargo build --release --example planner_v2 --example simulability
B=target/release/examples/planner_v2
$B req mps 'ct:n=24,L=8,t=16,nn=1' 1             # evolve once, every read-out timed
$B plan v2nc s100k 'brick:n=20,D=8,nn=1' 1        # plan + execute (v1 | v2 | v2nc | rule | force-ENGINE)
$B cachedemo 'qaoa:n=20,p=2,deg=3,nn=0' 1 e       # plan cache on a parameter sweep
cd research/data/planner-v2
python3 collect_req.py --bin $B --instances instances.txt --prior ../simulability/raw ../planner/mac --out mac/req.jsonl
python3 collect_req.py --bin $B --instances instances.txt --engines hsfamp --only-skipped mac/req.jsonl --out mac/hsfamp.jsonl --timeout 8
python3 fit_v2.py fit_v2 mac/feat_mac.jsonl mac/req.jsonl mac/hsfamp.jsonl
python3 readout_accuracy.py mac/feat_mac.jsonl mac/req.jsonl mac/hsfamp.jsonl
$B feat instances.txt > feat_voi.jsonl && python3 tune_voi.py feat_voi.jsonl mac/feat_mac.jsonl mac/req.jsonl mac/hsfamp.jsonl
python3 gen_e2e_jobs.py mac/feat_mac.jsonl mac/req.jsonl mac/hsfamp.jsonl > e2e_jobs.txt
python3 collect_req.py --bin $B --instances instances.txt --jobs e2e_jobs.txt --out mac/e2e3.jsonl
python3 eval_oracle.py eval_oracle_final.txt mac/e2e3.jsonl
python3 compare_autoab.py
cargo test --release --test planner_v2 --test planner --test pipeline
```
