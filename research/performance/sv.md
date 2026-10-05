# State-vector speed experiments (branch exp/sv)

All timings: min of N wall-clock runs, release build, portable target
(x86-64 baseline, SSE2) unless noted, taken through the swarm's `bench.sh`
lock on a **shared 4-vCPU VM (AMD EPYC-Rome)**. Other agents were compiling at
the same time (load average 8-20 during most runs), so absolute times are
inflated and noisy; A/B pairs are always measured back to back in the same
locked run and the ratios are the more reliable number.

Harness: `cargo run --release --example sv_speed -- <wl> <n,...> <f32|f64> <base|blocked|both> <reps> [key=val]`
prints a markdown row per mode and the max |Δamplitude| between the modes.
`examples/kernel_micro.rs` times single-thread in-cache kernels.

## 0. Profile of the baseline

* `perf` is not installed; instrumented with `std::time` instead.
* Microbenchmark (`rep:<gate>` workloads = 200 copies of one gate, n = 15,
  fully in L2): the existing per-gate kernels run at ~2 ns/amplitude per
  thread for a dense 2x2 gate *in cache* — i.e. they are not purely
  bandwidth bound, the complex arithmetic is not vectorised.
* `kernel_micro` (single thread, 2^15 amplitudes, f32, ns per amplitude):

| target qubit | AoS `Complex` ops | AoS explicit re/im | SoA (split re/im) |
|---|---|---|---|
| 0 | 2.19 | 2.11 | 1.92 |
| 1 | 2.14 | 2.02 | 1.67 |
| 2 | 2.43 | 2.30 | 0.71 |
| 3 | 2.30 | 2.04 | 0.53 |
| 5 | 2.09 | 2.08 | 0.50 |
| 10 | 2.08 | 2.16 | 0.48 |

  LLVM does not vectorise complex multiply-adds on interleaved
  `Complex<f32>`; with split real/imaginary arrays it does (4x).

## 1. End-to-end results (merged to main 1 Oct 2026)

The executor is opt-in: `StateVector::apply_circuit_blocked(&circuit, &BlockConfig::default())`.
`Circuit::run` still uses the gate-by-gate path. Wiring it in by default is
parked on branch `exp/sv-wip` (unreviewed).

The agent's own runs: base and blocked back to back in one locked
`bench.sh` invocation, f32, portable build, shared VM at load 7–10. Commit
ea41235.

| workload | n | base min (s) | blocked min (s) | speedup |
|---|---|---|---|---|
| QFT | 20 | 0.0912 | 0.0239 | 3.8× |
| QFT | 22 | 0.4551 | 0.0668 | 6.8× |
| QFT | 24 | 1.8509 | 0.1774 | 10.4× |
| random brickwork (990 gates) | 20 | 1.1448 | 0.2075 | 5.5× |
| random brickwork (1090 gates) | 22 | 4.1341 | 0.7634 | 5.4× |
| GHZ | 22 | 0.0428 | 0.0246 | 1.7× |
| GHZ | 24 | 0.1324 | 0.0567 | 2.3× |
| Grover | 22 | 2.9715 | 0.7929 | 3.7× |

Independent reproduction by the audit agent, at 973f40b, an earlier
commit (base and blocked interleaved in one process, 5 reps each, min
reported; see `research/data/audit/` on branch `exp/audit`):

| workload | n | prec | speedup |
|---|---|---|---|
| QFT | 22 | f32 | 6.05× |
| QFT | 22 | f64 | 7.29× |
| QFT | 24 | f32 | 8.05× |
| brickwork | 22 | f32 | 4.82× |
| brickwork | 24 | f32 | 4.97× |

Accuracy: `tests/blocked.rs` checks against the gate-by-gate path from random
start states for every block configuration (f64 within 1e-12, f32 within 1e-5,
proptest). The audit's independent-reference fuzz passed with worst
|Δamp| = 1.0e-15 (f64) and 3.7e-7 (f32), including adversarial block
configurations.

Where the speed comes from:
1. Splitting amplitudes into separate real and imaginary arrays (SoA), which
   lets LLVM vectorise the complex multiply-adds. That's about 4× in cache on
   its own.
2. Running all gates whose targets fit in an L2-sized chunk on that chunk
   before moving on. Memory is streamed once per block of gates, not once
   per gate.
3. Folding consecutive diagonal gates (most of QFT) into one phase pass, with
   ASAP layering and interval stabbing.
4. Fusing runs of single-qubit gates, splitting each fused gate into a real
   rotation and a phase.

GHZ gains least because it has n gates on 2^n amplitudes, so it's
memory-bound either way.
