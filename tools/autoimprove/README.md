# autoimprove: propose → verify → benchmark → keep

An automated loop for performance work on the blocked state-vector
executor (`src/blocked.rs`) and, once dense fusion is on main, its fusion
cost model. A candidate is a small patch. The harness builds it,
differential-fuzzes it against the independent reference state vector,
times it A/B against the base commit on a fixed suite, and keeps it only
when it is faster everywhere that matters. Every step is logged to a JSONL
ledger.

**Safety: nothing here pushes or merges.** Accepted patches are copied to
`~/qsim-ai/accepted/`. Turning one into an `exp/auto-<slug>` branch is a
manual step (see below), and merging stays a human decision.

Method, results and an honest assessment are in
[research/performance/autoimprove.md](../../research/performance/autoimprove.md).

## Files

| file | role |
|---|---|
| `autoimprove.py` | the harness and orchestrator (Python 3, standard library only) |
| `rust/ai_bench.rs` | bench driver, copied to `examples/ai_bench.rs` of the tree under test |
| `rust/ai_gate.rs` | correctness gate, copied to `tests/ai_gate.rs` of the tree under test |
| `candidates/*.patch` | proposer output: one idea per patch, metadata in `# slug/family/idea/parent:` header lines |
| `knobs-dense.json`, `knobs-combo.json` | runtime-knob sweeps (BlockConfig fields incl. the dense-fusion width and cost rule; no rebuild), on the base or on a candidate binary |
| `consts.json`, `consts-lookahead.json` | compile-time-constant sweeps (each value becomes a one-line patch, optionally on top of a parent patch) |
| `report.py` | markdown tables from a ledger |
| `run.sh` | the long-running queue worker |

The drivers live here, not in `examples/` or `tests/`, so CI does not pay
for them. They are installed into the tree under test at build time. Lines
tagged `// @dense` are dropped when that tree has no dense fusion.

## Pipeline (one candidate)

1. **build**: reset the eval worktree to the base commit, `git apply` the
   patch, install the drivers, then `cargo build --release --example
   ai_bench -j 2`.
2. **verify**: `cargo test --release --test ai_gate`. This runs 4000 random
   circuits by default (`AI_GATE_CASES`) against `tests/audit_common::RefSv`
   and covers:
   * every `Gate` variant, including the six the audit generator never
     emits (I, Sx, Sxdg, U, ISwap, ISwapdg), checked against textbook
     matrices written out in the test;
   * 1–14 qubits, plus 16–18 qubits on the default config;
   * 13 adversarial `BlockConfig`s (64 B to 64 KiB blocks, 0–6 slots,
     tiling, fusion and scheduling on and off, both kernel builds);
   * f64 (tolerance 1e-10) and f32 (5e-5);
   * both `apply_circuit_blocked` and `compile_kops`/`run_compiled`.

   Half the seeds are fixed; the other half are mixed with a hash of the
   patch, so coverage grows across candidates. Any failure rejects the
   candidate before any timing runs. The module's own test files
   (`blocked`, `l1_tiling`, `simd`, `dense_fusion`, `differential_fuzz`)
   take minutes to build in release, so they run only for candidates that
   pass the benchmark (step 4; `AI_TESTS_FIRST=1` runs them before timing
   instead).
3. **bench**: interleaved A/B, with A = base binary and B = candidate
   binary.
   * Metric: single-thread process CPU time (`AI_METRIC=cpu`,
     `AI_BENCH_THREADS=1`). On the shared Mac, 4-thread wall clock varies
     ±15% in A/A runs; see the notebook.
   * One process per measurement (min of up to 9 in-process timings,
     stopping at 0.4 s), 4 processes per side in ABBA order.
   * Screen suite: n = 20, 22. If it passes (geo-mean ≥ 1.01, no case
     < 0.94), the full suite runs: QFT, brickwork (depth 20) and random
     Clifford+T (16 layers), at n = 20, 22, 24, in f32 and f64. That is
     18 cases. `confirm` re-times a winner with 4 threads, wall clock, at
     n = 20, 23, 26.
   * Every A/B pair also compares a fingerprint of the final state (a
     fixed random linear functional plus the norm). A mismatch rejects the
     candidate, which gives an extra correctness check at full size.
4. **decide**: accept iff the module tests pass, the full-suite geo-mean
   speedup is ≥ **1.03** by both the ratio of mins and the median paired
   ratio, and no case is slower than **0.97×** by both. A case that looks
   slower is re-measured once (4 more reps per side, pooled).
5. **log**: one JSON record per candidate in `~/qsim-ai/ledger.jsonl`:
   * the full diff;
   * build and test results with log tails on failure;
   * every individual timing, the load average at each locked chunk, and
     the thread count;
   * the verdict.

### Swarm rules built in

* All timing runs under `/tmp/qsim-mac-bench.lock`:
  * locked chunks of ≤ 140 s (`AI_CHUNK_S`), with ≥ 65 s gaps between
    them (`AI_GAP_S`);
  * at most 19 locked minutes per rolling hour (`AI_HOUR_BUDGET_S`,
    tracked in `~/qsim-ai/lock-intervals.json`).
* Builds and the gate use 2 workers (`AI_JOBS`, `AI_GATE_THREADS`).
* Timings use 1 thread (`AI_BENCH_THREADS`); `confirm` uses 4 (`AI_CONFIRM_THREADS`).
* Scratch stays under `~/qsim-ai` (target dir about 1 GB).

## Usage (on the Mac)

```sh
mkdir -p ~/qsim-ai && cp -r tools/autoimprove ~/qsim-ai/harness
python3 ~/qsim-ai/harness/autoimprove.py setup            # worktree ~/qsim-ai/tree + base binary
python3 ~/qsim-ai/harness/autoimprove.py aa --runs 2      # A/A noise calibration (base vs a copy)
python3 ~/qsim-ai/harness/autoimprove.py eval cand.patch  # one or more candidates
cp cand.patch ~/qsim-ai/queue/ && python3 ~/qsim-ai/harness/autoimprove.py queue   # long-running worker; touch ~/qsim-ai/STOP to end
python3 ~/qsim-ai/harness/autoimprove.py knobs ~/qsim-ai/harness/knobs-dense.json
python3 ~/qsim-ai/harness/autoimprove.py consts ~/qsim-ai/harness/consts.json
python3 ~/qsim-ai/harness/autoimprove.py summary
```

`--base REF` selects the base commit. The default is the commit in
`~/qsim-ai/BASE` when that file exists, else `origin/main`. Pin the base:
the Mac clone's `origin/main` moves whenever anyone fetches.

The queue worker (`run.sh`) takes one item at a time from
`~/qsim-ai/queue/`, in file-name order, and runs each in a fresh Python
process, so edits to the harness apply from the next item:

| item | effect |
|---|---|
| `*.patch` | `eval` |
| `{"aa": {"runs": 1, "metric": "cpu", "threads": "1"}}` | A/A calibration |
| `{"confirm": "<slug>"}` | multi-thread wall-clock re-timing of an evaluated candidate |
| `{"configs": [...]}` or `{"grid": {...}}` (+ `"suite"`, `"binary"`) | knob sweep |
| `{"consts": [...]}` | constant sweep |
| `{"ci": "<slug>", "patch": "branches/<slug>.patch", "light": false}` | CI gate in its own worktree (`ci/`, `ci-target/`): fmt, clippy `-D warnings`, examples, every test target built, and the executor's tests run |
| `{"fetch": true}` | `git fetch` in the eval tree |

`touch ~/qsim-ai/STOP` ends the worker after the current item. The bench
lock is released on SIGTERM and SIGHUP. Long runs go
under `nohup` (device_exec calls time out at 300 s).

## Writing a candidate (the proposer)

A candidate is `git diff` output against the base, preceded by header
lines:

```
# slug: pair-u1
# family: kernel
# parent: (slug of the accepted patch this extends, if any)
# idea: one sentence: what changes and why it should be faster
diff --git a/src/blocked.rs b/src/blocked.rs
...
```

Keep each patch to one idea, so the ledger attributes each speedup to the
idea that produced it. Combining winners is a candidate of its own (its
`# parent:` lists both).

## From accepted patch to branch (manual)

```sh
git switch -c exp/auto-<slug> origin/main
git apply ~/qsim-ai/accepted/<slug>.patch
cargo fmt --all --check && cargo clippy --all-targets -- -D warnings && cargo test --release
git commit -am "blocked: <what> (autoimprove: geo <x>, min <y> on M1 Pro)"
```
