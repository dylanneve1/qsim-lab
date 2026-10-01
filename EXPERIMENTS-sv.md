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
