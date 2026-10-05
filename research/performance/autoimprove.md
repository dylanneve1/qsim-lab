# autoimprove: an automated propose → verify → benchmark → keep loop for the blocked SV kernels

Branch `exp/autoimprove` (harness + this notebook). Accepted improvements are on
their own branches, `exp/auto-<slug>`. Bases: 86dc099 (round 1), then 895543f / eee9bb0 /
c0403bb as main moved (round 2). Date: 5 Oct 2026. Machine: MacBook Pro M1 Pro (6P+2E, 16 GB), macOS,
Homebrew cargo 1.94, `--release`; shared with other agents (load average 5–15 throughout).

**Headline.** `tools/autoimprove/` puts each candidate patch through a fixed pipeline:

1. build;
2. a 4000-case differential fuzz against the reference SV, which rejected three of three
   deliberately subtle mutants;
3. interleaved A/B timing on QFT/brickwork/random Clifford+T, f32 and f64, under the swarm's
   bench lock;
4. accept only if the geo-mean is ≥ 1.03 and no case is < 0.97×;
5. a JSONL ledger of everything.

Over about 5 hours it ran 16 evaluations of 14 distinct real candidates (some re-run as main
moved three times under it). Against current main (c0403bb) it accepted three:

| patch | idea | geo-mean |
|---|---|---|
| `la-nophase-r3` | commutation-aware lookahead stage planning; halves the full-state sweeps on brickwork | 1.070× |
| `tls-scratch` | per-thread block buffers | 1.047× |
| `pair-cx-v2-r3` | two 1q gates plus a CNOT in one sweep | 1.034× |

Together they give **1.137× geo-mean single-thread (CPU time), no case slower**, and
**1.263× with 4 threads (wall clock, n = 20–26)**, on an M1 Pro. They are pushed as
`exp/auto-*` branches.

The loop's main value was verification. Four plausible ideas were measured and dropped:
two were slower, one was neutral and one had a slower case. A first pair kernel cut block passes by 29% yet ran at 0.79×; the cause was a
vectorisation failure. Two round-1 winners regressed after dense fusion merged, and a child
fixed one of them. The automatic knob search found nothing to change: the existing knobs
were already hand-tuned.

## 1. What the loop is

`tools/autoimprove/` ([README](../../tools/autoimprove/README.md)) turns a candidate patch into a
verdict without a human in the loop. It never pushes or merges anything: accepted patches are
copied to a directory, and turning one into a branch and merging it stay human decisions.

| step | what runs | rejects on |
|---|---|---|
| build | reset the eval worktree to the base commit, `git apply`, install the drivers, `cargo build --release --example ai_bench -j 2` | apply or compile failure |
| verify | `tests/ai_gate.rs`: 4000 random circuits against `tests/audit_common::RefSv`; then the module's own tests (`blocked`, `l1_tiling`, `simd`, `differential_fuzz`, `dense_fusion` when present) | any amplitude off by more than 1e-10 (f64) or 5e-5 (f32); any test failure |
| screen | interleaved A/B on 12 cheap cases (n = 20, 22) | geo-mean < 1.01 or a case < 0.94× |
| full | interleaved A/B on 18 cases: QFT, brickwork (depth 20), random Clifford+T (16 layers) × n = 20, 22, 24 × f32/f64 | geo-mean < 1.03 or a case < 0.97× |
| fingerprint | every A/B pair compares a random linear functional of the final state, plus its norm | mismatch: a correctness check at full size, for free |
| ledger | one JSON line per candidate: the diff, build and test logs, every timing, load averages, plan shape (stages, block passes), the verdict | — |

`ai_gate` covers:

* every `Gate` variant, including the six the audit generator never emits (I, Sx, Sxdg, U,
  ISwap, ISwapdg, against textbook matrices written out in the test);
* 1–14 qubits, plus 16–18 qubits on the default config;
* 13 adversarial `BlockConfig`s (64 B to 64 KiB blocks, so plans have many stages and gathers
  even at a few qubits; 0–6 slots; tiling, fusion and diagonal scheduling on and off; both
  kernel builds), plus four dense-fusion configs on trees that have it;
* f64 and f32;
* both `apply_circuit_blocked` and `compile_kops` + `run_compiled`.

Half the seeds are fixed. The other half are mixed with a hash of the patch, so the coverage
grows with every candidate. Runtime is about 45 s with 2 threads.

**The gate catches subtle bugs.** Three deliberately wrong one-line mutants were fed through
the loop, and all three were rejected by `ai_gate` within its first run:

| mutant | bug | caught by |
|---|---|---|
| `mut-cout` | outer-control test `base & cout == cout` weakened to `!= 0`: wrong only for gates with ≥ 2 out-of-block controls | wrong amplitudes |
| `mut-diaglo` | diagonal per-bit factor assigned instead of multiplied: wrong only when two active terms of one group hit the same bit | wrong amplitudes |
| `mut-smallpairs` | low-bit control pattern test `==` → `>=` | out-of-bounds panic |

### Timing protocol, and why it is not the obvious one

The first protocol was 4 threads, wall clock, min of 3, n = 20/22/26. Its A/A calibration
(base binary against a copy of itself) gave geo-means of 1.009 and 1.040 and single-case
ratios from 0.91 to 1.18. On a Mac at load 12 on 8 cores, 4-thread wall-clock time is
dominated by descheduling. Run to run, brickwork-22 varied by 30–40%.

The protocol the loop uses now:

* **Metric:** single-thread process CPU time (`getrusage`). Descheduled time is not counted.
  Kernel and planner changes act per core, so this is the quantity they change.
* **Interleaving:** one process per measurement (min of up to 9 in-process timings, stopping
  at 0.4 s); 4 processes per side in ABBA order, so the A/B order is balanced.
* **Two statistics, both required:**
  * the ratio of the per-side minima (the CONTRIBUTING rule);
  * the median of the per-rep paired ratios (robust to one bad rep).

  Accept needs both geo-means ≥ 1.03. A case counts as slower only if both statistics put it
  below 0.97.
* **Re-measurement:** any case that looks slower gets 4 more reps per side, pooled. A/A runs
  show occasional ~10% single-case outliers (a rep landing on an efficiency core or in a
  frequency dip).
* **Bench lock:** every timing runs under `/tmp/qsim-mac-bench.lock`, in chunks of ≤ 140 s
  with ≥ 65 s gaps, and at most 19 locked minutes per rolling hour.
* **Multi-thread check:** winners get a separate 4-thread wall-clock run at n = 20/23/26
  (`confirm`).

A/A calibration under this protocol:

| protocol | base | load avg | geo (ratio of mins) | geo (paired median) | worst / best case |
|---|---|---|---|---|---|
| v0: 4 thr, wall, min of 3, A first in 2 of 3 reps | 86dc099 | 12.9 | 1.009 | — | 0.906 / 1.063 |
| v0 (repeat) | 86dc099 | 12.6 | 1.040 | — | 0.927 / 1.158 |
| v1: 4 thr, wall, in-process min, ABBA × 4 | 86dc099 | 11.9 | 1.029 | 0.996 | 0.940 / 1.168 |
| v1 (repeat) | 86dc099 | 11.3 | 1.004 | 1.020 | 0.900 / 1.178 |
| **v2: 1 thr, process CPU time, ABBA × 4** (the loop's protocol) | 86dc099 | 10.1 | 1.007 | 0.992 | 0.901 / 1.073 |
| v2 (repeat) | 86dc099 | 8.3 | 0.993 | 1.002 | 0.973 / 1.013 |
| v2 | eee9bb0 | 3.1 | 1.000 | 1.002 | 0.992 / 1.006 |

With v2 the geo-mean noise is about ±1%, well inside the 1.03 acceptance margin. Single-case
outliers of up to 10% still happen under heavy load (first v2 run), which is why a case that
looks slower is re-measured before it can reject a candidate.

## 2. The proposer

The proposer is the agent writing small, single-idea patches into
`tools/autoimprove/candidates/`. It uses three inputs:

* a `QSIM_PROF` kernel-time breakdown of the base;
* the ledger, including the plan-shape statistics (stages, block passes) that the driver
  prints per case;
* the failure mode of rejected candidates.

Children of a winner carry `# parent:`. Combinations of winners are candidates in their own
right.

Two modes need no new code:

* `knobs`: runtime `BlockConfig` fields on the same binary (block size, slots, tiling, the
  dense-fusion width and cost rule);
* `consts`: compile-time constants, each value becoming a one-line regex patch, optionally on
  top of a parent patch.

Profile of the base (1 thread, n = 20, ms, thread-summed):

| workload | u1 complex | u1 real | u1 X | low-bit 1q / controlled | diag | swap | AoS→SoA load | store |
|---|---|---|---|---|---|---|---|---|
| brick f32 | 202 | 0 | 40 | 71 | 0 | 0 | 45 | 13 |
| brick f64 | 378 | 0 | 66 | 75 | 0 | 0 | 82 | 25 |
| clifft f32 | 3 | 18 | 29 | 23 | 43 | 0 | 14 | 5 |
| qft f32 | 0 | 5 | 0 | 1 | 6 | 4 | 3 | 1 |

## 3. Round 1 (base 86dc099, before dense fusion)

Candidates in the order the loop ran them. Geo-means are given as ratio of mins / median
paired ratio / worst case (the larger of the two statistics for the slowest case).

| candidate | idea | verdict | screen | full |
|---|---|---|---|---|
| `cgu1` | `codegen-units = 1` for the qsim-lab crate | rejected (screen) | 1.008 / 1.009 / 0.983 | |
| `cmul2-split` | complex multiply-add as two 2-deep FMA chains plus an add | rejected (screen) | 0.946 / 0.954 / 0.884 | |
| `plan-lookahead` | commutation-aware stage planning (below) | **accepted** | 1.070 / 1.071 / 1.000 | **1.062 / 1.061 / 0.977** |
| `low-t-chunks` | bit-0/1 1q kernels on `as_chunks_mut::<2\|4>` arrays | **accepted** | 1.034 / 1.032 / 0.998 | **1.032 / 1.032 / 0.999** |
| `pair-cx` | two 1q gates (and a following CNOT on the same bits) in one sweep | rejected (screen) | 0.788 / 0.788 / 0.453 | |
| `combo-la-lowt` | `plan-lookahead` + `low-t-chunks` | **accepted** | 1.108 / 1.108 / 1.051 | **1.092 / 1.089 / 0.999** |
| `pair-cx-v2` | `pair-cx` with its loop in a function taking the 8 slices as `&mut` parameters | **accepted** | 1.043 / 1.040 / 1.004 | **1.040 / 1.040 / 1.002** |
| `mut-*` (3) | deliberate bugs | rejected (gate), as intended | | |

### What the accepted candidates do

* **`plan-lookahead`** (`plan_stages`, +40 lines). Each stage of the blocked executor caches
  `l` qubits: the low ones plus up to `slots` outer ones. It sweeps the whole state once. The
  old planner closed a stage at the first op whose qubits did not fit.
  * The new one keeps scanning. It takes every later op that fits the stage's slots and
    commutes with every op skipped so far: two ops commute unless one acts non-diagonally on
    a qubit the other touches. OR-ing the skipped ops' masks makes this test exact and O(1).
  * Skipped ops keep their order and seed the next stage. The scan is bounded to 1024 ops
    past the first skip.
  * Effect: brickwork-24 needs 21 instead of 42 full-state sweeps, Clifford+T-24 8 instead
    of 16; QFT is unchanged. Brickwork gains 7–21%.
  * The one weak spot is f32 Clifford+T-24: 0.964–0.977× in round 1 and 0.965× again in
    round 2. Block passes rise 274 → 288 there, because pulling phases ahead fragments the
    stages' diagonal runs. The metal backend calls the same `plan_stages`, so it inherits
    the change.
* **`low-t-chunks`** (+30/−30). Uncontrolled 1q gates on buffer bits 0 and 1 used
  `u1_group8`, a gather/scatter through per-group index maps. The profile showed these
  "low-bit" ops cost about 3× a normal sweep. The candidate views the block as `[T; 2]` /
  `[T; 4]` arrays instead (`as_chunks_mut`), a constant-stride interleave the vectoriser
  handles. f32 gains 4–9%; f64 (2 lanes) gains about 1%.
* **`pair-cx-v2`** (+250). Two adjacent uncontrolled 1q gates on distinct buffer bits (both
  ≥ 2) become one `LOp::Pair`. Its kernel loads the four amplitudes of each group once,
  applies both 2×2s in registers, and folds an immediately following CNOT between the same
  two bits in as a register permutation. In brickwork, `U1 U1 CNOT` takes one block sweep
  instead of three: passes drop from 572 to 407 at n = 20.
* **`tls-scratch`** (+50/−30). `run_stage` used `rayon`'s `for_each_init`, which allocates
  (zeroing and page-faulting) a fresh `2^l` block buffer and scratch per rayon split, in
  every stage. Each worker now keeps one buffer and its scratch in a `thread_local`.

### What the rejections taught the proposer

* **`pair-cx` → `pair-cx-v2` is the loop working as intended.** The first version cut block
  passes by 29% and was still 2× *slower* (0.45× on f32 brickwork). Its inner loop split
  eight `&mut` slices inside one function. LLVM cannot prove they do not alias, would need
  28 run-time alias checks, and so falls back to scalar code. The existing 4-slice kernels
  avoid this by taking their slices as function parameters (`noalias`). Moving the loop into
  such a function gave 1.040×, and nothing else changed.
* **`cmul2-split` was 5% slower.** "More ILP" from two short FMA chains loses to one chain
  that folds the adds into FMAs. Out-of-order execution already overlaps independent loop
  iterations.
* **`cgu1`: single-codegen-unit builds are neutral** (1.008×). Every hot kernel is already
  `#[inline(always)]`.

## 4. Round 2: re-validation on the dense-fusion main, and the children

Dense fusion merged mid-session (895543f; default width 2 on aarch64). Two merges followed:
lowmagic-chem (eee9bb0, no kernel changes) and the graph compiler (c0403bb, which touches
`src/engines/blocked.rs`). Every round-1 winner was re-run against the new main through the same
pipeline, and the loop bred children for the ones that broke.

| base | candidate | verdict | screen | full |
|---|---|---|---|---|
| 895543f | `tls-scratch`: per-thread block buffer and scratch (`thread_local`) instead of per-split allocation | **accepted** | 1.062 / 1.062 / 1.018 | **1.047 / 1.046 / 1.005** |
| eee9bb0 | `plan-lookahead` (round-1 winner) | rejected: one case at 0.966 | 1.072 / 1.072 / 0.977 | 1.055 / 1.056 / 0.966 |
| eee9bb0 | `low-t-chunks` (round-1 winner) | rejected: geo 1.027 < 1.03 | 1.027 / 1.027 / 0.998 | 1.027 / 1.026 / 0.994 |
| eee9bb0 | `la-nophase`: child of `plan-lookahead`; after a stage's first skip, phases are deferred too | **accepted** | 1.083 / 1.085 / 1.006 | **1.074 / 1.074 / 0.996** |
| eee9bb0 | `plan-la-score`: child of `plan-lookahead`; outer qubits chosen by a lookahead score | rejected: one case at 0.939 | 1.066 / 1.067 / 0.958 | 1.047 / 1.049 / 0.939 |
| c0403bb | `la-nophase-r3`: `la-nophase` with `plan_stages` kept order-preserving (see below) | **accepted** (graph/ooc/pipeline tests included) | 1.082 / 1.084 / 0.994 | **1.070 / 1.073 / 1.000** |
| c0403bb | `pair-cx-v2-r3`: the pair kernel, rebased | **accepted** | 1.028 / 1.029 / 0.982 | **1.034 / 1.032 / 0.994** |

How the children came about:

* **`plan-lookahead` → `la-nophase`.** On both bases the parent left f32 Clifford+T-24 at
  0.965×. The ledger's plan statistics pointed at the cause: stages halved (16 → 8), but
  block passes rose 274 → 288, so the lookahead was fragmenting diagonal runs.
  * Phases need no slot, so the parent pulled every commuting phase into the current stage,
    splitting the stage's diagonal runs.
  * The child defers phases once a stage has skipped anything. Clifford+T-24 f32 goes from
    0.965× to 1.052× (13 stages, 273 passes). The other cases keep the parent's gains.
  * The score-based alternative (`plan-la-score`) was worse: 0.939 on one case.
* **`low-t-chunks` is real but marginal.** It is about +4–9% on f32 and +1% on f64, which
  lands on the 1.03 threshold: 1.032× on one base, 1.027× on the next. The rules drop it.
  Its patch stays in `candidates/` for whoever wants the f32 gain.
* **`tls-scratch`** gains most at small n: 1.16× on QFT-20 f32, falling to 1.01× at n = 24,
  as the per-stage allocation is amortised over larger blocks of work. The multi-thread
  confirmation is the better test of it, since there are more rayon splits per stage.

### Graph-compiler compatibility (found at the c0403bb rebase, not by the gate)

The graph compiler (merged in c0403bb) assumes `plan_stages` keeps the op order: it maps
parameter slots to "contiguous ranges of the op list". A lookahead `plan_stages` would have
broken parameter binding silently. `ai_gate` tests the blocked executor's own entry points,
so it would not have caught this. Two fixes went in:

* **The rebased patch** (`la-nophase-r3`) keeps `pub fn plan_stages` order-preserving. It
  is the same code with a lookahead of 0, which is exactly the old greedy. A new
  `plan_stages_lookahead` is used by the blocked executor's own paths (`apply_kops_blocked`,
  `compile_kops`, `BlockedChunkExecutor`, `tile_stats`, `fusion_stats`). The graph compiler
  and Metal keep the order-preserving planner. They could adopt the lookahead if they map
  ops by index rather than by range.
* **The harness** now also runs `tests/compiler/graph.rs`, `tests/engines/ooc.rs` and `tests/compiler/pipeline.rs` as
  module tests, since those drive the planner and prepared stages from outside.

The pair kernel is rebased the same way. Pairing happens in `prepare` only, never in the
`prepare_mapped` that the graph compiler patches by `OpLoc`.

Per case, the stage planner (`la-nophase-r3` vs c0403bb). The metric is single-thread
process CPU time in seconds, min of 4+ per side; "passes" are block-level ops after
preparation.

| workload | n | prec | base | cand | speedup (mins) | paired median | stages | passes |
|---|---|---|---|---|---|---|---|---|
| qft | 20 | f32 | 0.0132 | 0.0130 | 1.009 | 0.999 | 3→3 | 49→49 |
| qft | 20 | f64 | 0.0229 | 0.0215 | 1.066 | 1.075 | 4→3 | 49→49 |
| brick | 20 | f32 | 0.1623 | 0.1356 | 1.197 | 1.207 | 29→6 | 572→536 |
| brick | 20 | f64 | 0.2702 | 0.2359 | 1.145 | 1.150 | 32→10 | 572→532 |
| clifft | 20 | f32 | 0.0764 | 0.0737 | 1.036 | 1.047 | 11→8 | 226→225 |
| clifft | 20 | f64 | 0.1221 | 0.1188 | 1.028 | 1.030 | 12→9 | 227→228 |
| qft | 22 | f32 | 0.0539 | 0.0520 | 1.036 | 1.048 | 4→3 | 54→54 |
| qft | 22 | f64 | 0.0927 | 0.0897 | 1.033 | 1.040 | 4→3 | 54→54 |
| brick | 22 | f32 | 0.6696 | 0.5899 | 1.135 | 1.135 | 36→13 | 632→592 |
| brick | 22 | f64 | 1.1565 | 1.0459 | 1.106 | 1.108 | 39→16 | 632→588 |
| clifft | 22 | f32 | 0.3026 | 0.2849 | 1.062 | 1.066 | 13→9 | 240→236 |
| clifft | 22 | f64 | 0.5047 | 0.4532 | 1.114 | 1.113 | 15→11 | 242→237 |
| qft | 24 | f32 | 0.2366 | 0.2366 | 1.000 | 1.000 | 5→5 | 59→59 |
| qft | 24 | f64 | 0.4127 | 0.4133 | 0.999 | 1.000 | 5→5 | 59→59 |
| brick | 24 | f32 | 2.8595 | 2.5869 | 1.105 | 1.103 | 42→21 | 692→652 |
| brick | 24 | f64 | 5.0336 | 4.5652 | 1.103 | 1.098 | 46→21 | 692→644 |
| clifft | 24 | f32 | 1.3020 | 1.2342 | 1.055 | 1.054 | 16→13 | 274→273 |
| clifft | 24 | f64 | 2.2020 | 2.0760 | 1.061 | 1.061 | 19→14 | 276→276 |

Two side effects show in the table:

* With longer stages, the dense-fusion cost rule now finds groups worth fusing in
  brickwork: passes go 692 → 644–652 at n = 24.
* The pair kernel's gains are workload-specific. Random Clifford+T gains 1.07–1.12×, since
  its paired gates are cheap (H, X, phases). Brickwork gains only 1.00–1.02×, although its
  passes drop 692 → 487: brickwork's in-block sweeps are compute-bound, not L2-bound.
  QFT is unchanged.

## 5. The combination: `combo6` = `la-nophase-r3` + `tls-scratch` + `pair-cx-v2-r3`

The three accepted patches on c0403bb, together: branch `exp/auto-combo`, three commits
cherry-picked from the single-patch branches. The branch diff is byte-identical to the
evaluated patch.

* Verdict: **accepted**. Full-suite geo-mean **1.137×** (paired 1.139×), worst case 1.009×.
  No case is slower.
* Gate: 4000 cases, worst error 8.1e-16 (f64), 4.0e-7 (f32). Module tests: 63 pass.
* Conditions: c0403bb, M1 Pro, 1 thread, process CPU time. Load 11–15 during the run. The
  CPU-time metric is what makes a run at that load usable.

| workload | n | prec | base CPU s | combo CPU s | speedup (mins) | paired median | stages | passes |
|---|---|---|---|---|---|---|---|---|
| qft | 20 | f32 | 0.0134 | 0.0119 | 1.124 | 1.120 | 3→3 | 49→49 |
| qft | 20 | f64 | 0.0232 | 0.0201 | 1.156 | 1.161 | 4→3 | 49→49 |
| brick | 20 | f32 | 0.1642 | 0.1305 | 1.258 | 1.266 | 29→6 | 572→383 |
| brick | 20 | f64 | 0.2722 | 0.2306 | 1.180 | 1.184 | 32→10 | 572→387 |
| clifft | 20 | f32 | 0.0774 | 0.0636 | 1.216 | 1.220 | 11→8 | 226→191 |
| clifft | 20 | f64 | 0.1238 | 0.1050 | 1.179 | 1.177 | 12→9 | 227→193 |
| qft | 22 | f32 | 0.0546 | 0.0509 | 1.072 | 1.074 | 4→3 | 54→54 |
| qft | 22 | f64 | 0.0936 | 0.0878 | 1.066 | 1.067 | 4→3 | 54→54 |
| brick | 22 | f32 | 0.6741 | 0.5785 | 1.165 | 1.168 | 36→13 | 632→432 |
| brick | 22 | f64 | 1.1583 | 1.0379 | 1.116 | 1.123 | 39→16 | 632→432 |
| clifft | 22 | f32 | 0.3036 | 0.2610 | 1.163 | 1.165 | 13→9 | 240→202 |
| clifft | 22 | f64 | 0.5056 | 0.4170 | 1.212 | 1.217 | 15→11 | 242→202 |
| qft | 24 | f32 | 0.2363 | 0.2341 | 1.009 | 1.009 | 5→5 | 59→59 |
| qft | 24 | f64 | 0.4148 | 0.4111 | 1.009 | 1.008 | 5→5 | 59→59 |
| brick | 24 | f32 | 2.8654 | 2.5449 | 1.126 | 1.127 | 42→21 | 692→473 |
| brick | 24 | f64 | 5.0197 | 4.5749 | 1.097 | 1.100 | 46→21 | 692→466 |
| clifft | 24 | f32 | 1.3044 | 1.1149 | 1.170 | 1.171 | 16→13 | 274→228 |
| clifft | 24 | f64 | 2.2101 | 1.8771 | 1.177 | 1.177 | 19→14 | 276→229 |

The gains roughly multiply: 1.070 × 1.047 × 1.034 ≈ 1.158, against 1.137 measured. QFT at
n = 24 is the one workload left essentially unchanged (1.01×). Its five stages are forced by
the H gates on 13 outer qubits and the final swaps, and lookahead cannot merge them.

### Multi-thread wall-clock confirmation (`confirm`)

combo6 vs c0403bb, 4 threads, wall clock, ABBA × 4 per side, min, n = 20/23/26 (the brief's
range). Load average was 10–17 during the run. Even so, per-side spreads stayed at 1–2%,
because the differences are large.

| workload | n | f32 speedup (mins / paired) | f64 speedup (mins / paired) | base f32 s | base f64 s |
|---|---|---|---|---|---|
| qft | 20 | 1.334 / 1.307 | 1.500 / 1.507 | 0.005 | 0.009 |
| brick | 20 | 1.545 / 1.544 | 1.483 / 1.478 | 0.054 | 0.093 |
| clifft | 20 | 1.350 / 1.346 | 1.377 / 1.378 | 0.024 | 0.040 |
| qft | 23 | 1.246 / 1.250 | 1.155 / 1.159 | 0.035 | 0.063 |
| brick | 23 | 1.305 / 1.307 | 1.227 / 1.231 | 0.423 | 0.713 |
| clifft | 23 | 1.245 / 1.251 | 1.333 / 1.332 | 0.178 | 0.292 |
| qft | 26 | 1.041 / 1.050 | 1.060 / 1.071 | 0.277 | 0.504 |
| brick | 26 | 1.172 / 1.173 | 1.141 / 1.148 | 3.378 | 5.966 |
| clifft | 26 | 1.170 / 1.165 | 1.184 / 1.186 | 1.456 | 2.571 |

Geo-mean **1.263×**, minimum 1.041×. Multi-threaded, the gains are larger than the
single-thread numbers. `tls-scratch` is the reason: with 4 workers there are more rayon
splits per stage, and so more per-split buffer allocations to save. At n = 20–23 the
allocations are a large share of a run.

## 6. Parameter search without code changes (`knobs`)

One knob at a time against the default, on the same binary: the base c0403bb. The suite is
QFT/brickwork/Clifford+T at n = 22 and 24, f32 and f64. Metric: 1 thread, CPU time.
Ratio > 1 means the knob setting is faster than the default.

| knob (default) | setting | geo (ratio of mins) | geo (paired median) | min case | max case |
|---|---|---|---|---|---|
| dense-fusion width (2) | `dense_k=0` (off) | 1.000 | 1.001 | 0.995 | 1.005 |
| | `dense_k=3` | 1.001 | 1.000 | 0.997 | 1.010 |
| dense cost rule, min heavy 1q gates per group (0 = 2^k) | `dense_min=2` | **0.882** | 0.881 | **0.612** | 1.003 |
| | `dense_min=3` | 0.998 | 0.999 | 0.981 | 1.005 |
| | `dense_min=6` | 1.000 | 1.000 | 0.992 | 1.006 |
| outer slots per stage (6) | `slots=5` | 0.987 | 0.989 | 0.960 | 1.015 |
| | `slots=7` | 0.986 | 0.986 | 0.946 | 1.028 |
| block size (1 MiB) | `block_kib=2048` | 1.003 | 1.002 | 0.969 | 1.045 |

No default is worth changing.

* On the base planner, dense fusion is neutral on this suite: its cost rule rarely fires on
  brickwork, QFT or random Clifford+T at the old stage lengths.
* Loosening the rule to 2 heavy gates per group costs 12% overall and up to 39% on one case.
  That independently confirms the hand-set threshold of `research/performance/dense-fusion.md`.
* Slots 6 and the 1 MiB block are also confirmed (`research/performance/mac-m1.md`).

One interaction is visible only in the planner patch: with lookahead stages, the default
cost rule starts firing on brickwork (passes 692 → 644–652, §4). The sweep on top of
`combo6` (`knobs-combo.json`) was queued but not run.

## 7. Honest assessment

**What the loop found.** On the current main, three patches were accepted and a fourth
remains marginal:

* `la-nophase-r3`: 1.070×;
* `tls-scratch`: 1.047×;
* `pair-cx-v2-r3`: 1.034×;
* `low-t-chunks`: marginal (1.027–1.032×, below the bar on the current main).

Together they give **1.137× single-thread** and **1.263× 4-thread wall clock** (geo-means),
with no case slower. That is real but incremental. For scale, earlier hand work on this
executor delivered:

* cache blocking with SoA layout: 4–10× over the gate-by-gate baseline;
* NEON FMA: 1.15–1.27×;
* dense fusion: 1.5–2.1× on circuits of generic 2-qubit unitaries.

The automated loop did not find anything of that size, and did not try to.

**Who did what.** The proposer was the agent:

* it read the profile and the ledger's plan statistics;
* it wrote each patch by hand;
* it diagnosed each failure. Two examples: `pair-cx` was scalar code (aliasing), and
  `plan-lookahead` fragmented diagonal runs.

The loop's contribution was the other half:

* **Verification:** it rejected all three subtle mutants, and every accepted patch passed
  4000 differential cases plus 63 module tests.
* **Rejecting plausible ideas that were slower:** `cmul2-split` (0.95×), `pair-cx` v1
  (0.79×), `cgu1` (1.01×) and `plan-la-score` (worst case 0.94×). Each looked reasonable
  on paper.
* **Catching regressions across bases:** `plan-lookahead` dropped to 0.966× on one case
  after dense fusion merged, and `low-t-chunks` fell from 1.032 to 1.027.
* **Holding claims to one standard:** every number in this notebook is interleaved A/B,
  min of ≥ 4 per side plus a paired statistic, under the bench lock.

**The fully automatic modes found little.** The knob sweep (§6) found no default worth
changing. The constant sweeps (`LO_BITS`, `SMALL`, the lookahead window) were queued but
not run: they were pushed back behind the re-validations after main moved three times
during the session. Search without new code is cheap to run, but the existing knobs were
already hand-tuned (block size, slots and tiling in `research/performance/mac-m1.md`; the
dense-fusion cost rule in `dense-fusion.md`). The headroom was in new code, not constants.

**Throughput.** One candidate takes about 15–25 minutes:

* about 1.5 min to build, 2.5 min for the gate;
* screen and full timing take about 4 locked minutes, plus the 65 s gaps and waits for
  peers holding the lock;
* about 10 min for the module tests (release builds of 8 test crates), run only for
  candidates that pass the benchmark.

The swarm's 19 locked minutes per hour cap the loop at roughly 3–4 full evaluations per
hour. Over about 5 hours it evaluated:

* 16 candidate evaluations of 14 distinct ideas, counting re-runs on new bases;
* 3 mutants;
* 7 A/A calibrations;
* 1 confirmation and 1 knob sweep.

**Is it worth keeping running?** Yes, but not as an unattended "find speedups" machine.

* **Keep: as a merge gate for performance claims.** Any kernel or planner change goes
  through `eval` against its base before it is called faster; that costs about 20 minutes
  and no human attention. The A/A, mutant and re-measurement machinery is what makes its
  verdicts trustworthy on a shared, loaded laptop.
* **Keep: the queue running whenever main moves**, re-validating the accepted patches.
  Two of five round-1 verdicts changed after dense fusion merged.
* **Don't:** run the knob/constant search on a schedule. It will mostly re-confirm the
  hand-tuned defaults. Run it when a new knob lands, e.g. the dense-fusion width and cost
  rule, or a new block-size heuristic.
* **Gaps to close first:**
  * extend `ai_gate` to the graph compiler's mapped stages and the Metal backend (§4: the
    graph compiler's order assumption was caught by reading the rebase, not by the gate);
  * add an x86 runner (every number here is M1 Pro; x86 shares no kernel code paths for
    the FMA build);
  * add Shor-oracle and noisy-circuit workloads to the suite, so the cost model is what the
    lab actually runs;
  * make the proposer step itself a scheduled agent job (today proposals are written
    interactively).

## 8. Caveats

* **One machine, one class of measurement.**
  * All timings are an M1 Pro shared with other agents.
  * The acceptance metric is single-thread CPU time. Multi-thread wall clock was checked
    only for the final combination (§5). Single-thread numbers understate `tls-scratch`
    and do not show bandwidth effects that appear only with all cores busy.
* **The suite is three synthetic families.** QFT, brickwork and random Clifford+T at
  20–26 qubits, chosen in advance. Gains on other circuits (deep diagonal phase
  polynomials, Shor oracles) are not measured.
* **The acceptance thresholds are conventions.** 1.03 geo-mean and 0.97 per case, with
  both statistics. `low-t-chunks` shows a patch can sit on the line and flip between bases.
* **A per-thread scratch buffer stays allocated after a run**, one block (1 MiB) per rayon
  thread (`tls-scratch`).
* **The pair kernel adds about 230 lines** of `src/engines/blocked.rs` for 3.4% single-thread. Most
  of the gain is on Clifford+T-like circuits; brickwork gains only 1–2%.

* **The ledger has five `error` rows.** They are the queue items consumed while a driver bug
  (a misnamed `TileStats` field) broke the base build; they were re-queued and re-run. The
  queue worker now retries a failed base build instead of consuming items.

## 9. Reproduction

* Harness, drivers, candidate patches and sweep specs: `tools/autoimprove/`. Setup and the
  queue are described in its README.
* Raw ledger, one JSON line per evaluation with diff, logs, every timing and load:
  `research/data/autoimprove/ledger.jsonl.xz`. The earlier protocol's A/A runs are in
  `ledger-v0.jsonl`. Render the tables with `tools/autoimprove/report.py <ledger>
  --cases <slug>@<base7>`.
* Branches (not merged):
  * `exp/auto-la-nophase`, `exp/auto-tls-scratch`, `exp/auto-pair-cx`: one commit each on
    c0403bb;
  * `exp/auto-combo`: the three commits stacked.



