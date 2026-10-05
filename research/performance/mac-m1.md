# Apple M1 Pro: NEON FMA kernels, nested L1 tiling, block size (round 4, 3 Oct 2026)

Branch `exp/l1-tiling-r4` (exp/l1-tiling rebased onto main fb30f56, reviewed, tested, benchmarked).
Mergeable subset: `exp/neon-fma-r4` (NEON FMA + 1 MiB aarch64 default block, no tiling).

## Verdict

| change | M1 Pro (interleaved A/B, min of 3-5) | VPS EPYC (quiet window) | recommendation |
|---|---|---|---|
| NEON FMA kernels (`BlockConfig::simd` on aarch64) | **1.15-1.27x** brickwork, 1.02-1.07x QFT vs main | n/a (x86 path unchanged: main/fma = 0.98-1.05, noise) | **MERGE** (exp/neon-fma-r4) |
| 1 MiB default block on aarch64 (untiled) | 1.03-1.12x brickwork, 1.01-1.31x QFT vs 256 KiB | unchanged (cfg-gated) | **MERGE** (exp/neon-fma-r4, separate commit) |
| nested L1 tiling (`l1_tile_bytes`) | over flag-off at the default block: brickwork 1.04-1.10x, QFT 1.01-1.07x; over the best untiled block (1 MiB): brickwork **1.04-1.06x**, QFT **0.96-1.01x** | +7-11% brickwork, **-3% to -15% QFT** | **DROP** (kill criterion met) |

The kill criterion was "<1.1x on Mac and neutral on VPS". The Mac half is met: no cell reaches 1.1x over flag-off except f64 brickwork-26 at 8 threads, which hits exactly 1.10x. Measured against the best untiled block, which the tiling actually competes with, the gain is 4-6%. On the VPS it isn't neutral but mixed: brickwork improves about 1.1x and QFT regresses up to 15%, so it couldn't be turned on by default either. As an off-by-default flag it's ~250 lines (`TilePlan`, `Seg`, `tile_segments` list scheduler, `View` refactor) for a workload-dependent single-digit gain. Drop it. Keep the FMA commit, which is 13 lines, and the block default.

Best Mac config (f32 and f64): `simd` on (NEON FMA), `block_bytes = 1 MiB`, `slots = 6`, tiling off (or `l1_tile_bytes = 64-128 KiB` for a further ~5% on brickwork only). Use **6 threads for n <= 24 and 8 threads for n >= 26**. The 2 E-cores help only once the state vector is far bigger than the caches. At n <= 24, 8 threads are 2-10% slower than 6.

## 1. Review of the branch (7 commits, rebased cleanly with `git rebase --onto origin/main a05ed2d`)

- **unsafe**: the branch adds **no** `unsafe`. The repo is *not* unsafe-free on main. The pre-existing blocks are `src/engines/blocked.rs` `unsafe fn run_ops_avx2` plus its call site, from exp/simd: it needs `#[target_feature(enable = "avx2,fma")]`, and the call is guarded by `is_x86_feature_detected!`. The other two are in `src/engines/ooc.rs:73,81`: `slice::from_raw_parts{,_mut}` reinterpreting `Complex<T>` as bytes for file I/O, sound for plain-old-data `Complex<f32|f64>`. The aarch64 FMA path needs no unsafe because FMA/NEON is baseline on aarch64, so `mul_add` lowers to `fmadd`/`fmla` without a target feature.
- **NEON FMA** (2668320): routes `simd=true` on aarch64 to `run_ops_impl::<T, true>` (the `mul_add` kernels), and `simd_available()` returns true on aarch64. Correct. I fixed the `BlockConfig::simd` and `simd_available` doc comments on exp/neon-fma-r4.
- **Tiling** (`tile_segments`, `apply_tile_run`), checked by hand:
  - Dependency rule: ops conflict iff one's non-diagonal bits meet the other's support. Diagonal blocks have nd=0, U1 has nd = target, swap has nd = both bits. This is the same rule `schedule_diag` uses. Outer controls (`cout`) are not in the masks, which is fine because no op inside a stage writes an outer bit.
  - Tile-local test: U1 needs `t < k`; swap needs `b < k` (`prepare` always emits `a < b`); a diagonal block needs every bit it reads `< k`.
  - Controls above the tile select tiles through `ti & (cin >> k)`. `k >= 3` is enforced, so `apply_u1`'s `l >= 3` fast path is valid on a tile.
  - The list scheduler emits a topological order.
  - Prep cost is negligible: 0.05-0.1 ms per stage at n=24 with `QSIM_TRACE=1`, against an O(ops^2) dependency build.
  - No correctness issue found.
- **Cleanups (commit 4841de4)**:
  - removed 19 macOS `._*` resource-fork files that had been committed under `research/data/l1/`;
  - added `tests/engines/l1_tiling.rs::tiled_matches_audit_reference`, which differential-tests the tiled executor against the independent `audit_common::RefSv`: edge angles/qubits, SWAP/CCX/CPhase, n = 3..12, 7 geometries, portable and SIMD, f64 <= 1e-12 and f32 <= 1e-5;
  - `cargo fmt`, plus one clippy precedence warning in `examples/l1_micro.rs`.
- **Tests**: full `cargo test --release` is green on the VPS (x86_64, AVX2 path, 25 test binaries + doc-tests) and on the Mac (aarch64, NEON path, 25 test binaries + doc-tests, 0 failures).

## 2. Mac A/B: main vs branch, flag off / on

Setup:
- Machine: MacBook Pro M1 Pro, 6P+2E, 16 GB, macOS, Homebrew cargo 1.94.1, `--release`.
- Binaries (separate target dirs, `cmp`-checked to differ):
  - `main`: fb30f56 with `examples/l1_bench.rs` minus its tile keys (`data/mac-m1/scripts/l1_bench_for_main.rs`);
  - branch: `exp/l1-tiling-r4` 4841de4.
- n=28 f32 binaries are bench-only builds with `MAX_STATE_BYTES` raised from 1 GiB to 4 GiB (not committed). n=28 f64 (4 GiB per state, 8 GiB in the harness) was skipped.
- Each cell runs under `/tmp/qsim-mac-bench.lock`. Reps are interleaved: every rep runs both binaries, the order alternates per rep, and each binary runs all of its configs. The table reports the min.
- Circuits: QFT on |0>, and random brickwork at depth 20 (seed 42: Ry, Rz on every qubit, then a CNOT brick layer).
- Columns: `fma` is the branch with the flag off (NEON FMA, block 256 KiB); `t32`/`t64` turn on L1 tiling with 32/64 KiB tiles; `fma/best-tile` is the gain from the tiling flag alone.
- Load: Logic Pro used 80-95% of one core for the whole session, and macOS load averages were 5-14, mostly from our own 8-thread runs.
- Contamination: another agent's unlocked `cargo build` ran from 22:03 to 22:28 Mac time. Every affected cell was re-run, and `cell.sh` now logs `CONTAM` whenever rustc or cargo is alive. The n=28 brickwork rows at 6 and 8 threads, and QFT-28 at 6 threads, come from the first n=28 pass, which may overlap that build; their ratios match the clean cells.

| workload | prec | n | thr | main s | fma s | t32 s | t64 s | reps | main/fma | main/t32 | main/t64 | fma/best-tile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brick | f32 | 22 | 4 | 0.2656 | 0.2143 | 0.2044 | 0.2021 | 5 | 1.24x | 1.30x | 1.31x | 1.060x |
| brick | f32 | 22 | 6 | 0.2211 | 0.1806 | 0.1730 | 0.1708 | 5 | 1.22x | 1.28x | 1.29x | 1.057x |
| brick | f32 | 22 | 8 | 0.2234 | 0.1891 | 0.1835 | 0.1818 | 5 | 1.18x | 1.22x | 1.23x | 1.040x |
| brick | f32 | 24 | 4 | 1.1026 | 0.8700 | 0.8271 | 0.8154 | 3 | 1.27x | 1.33x | 1.35x | 1.067x |
| brick | f32 | 24 | 6 | 0.8766 | 0.7031 | 0.6666 | 0.6606 | 3 | 1.25x | 1.32x | 1.33x | 1.064x |
| brick | f32 | 24 | 8 | 0.8691 | 0.7184 | 0.6861 | 0.6826 | 3 | 1.21x | 1.27x | 1.27x | 1.052x |
| brick | f32 | 26 | 4 | 4.6619 | 3.6853 | 3.5238 | 3.4698 | 3 | 1.26x | 1.32x | 1.34x | 1.062x |
| brick | f32 | 26 | 6 | 3.6367 | 2.9104 | 2.7455 | 2.7133 | 3 | 1.25x | 1.32x | 1.34x | 1.073x |
| brick | f32 | 26 | 8 | 3.4663 | 2.7637 | 2.6562 | 2.6589 | 3 | 1.25x | 1.30x | 1.30x | 1.040x |
| brick | f32 | 28 | 4 | 20.2600 | 16.0695 | 15.0172 | 14.7668 | 3 | 1.26x | 1.35x | 1.37x | 1.088x |
| brick | f32 | 28 | 6 | 15.6580 | 12.3717 | 11.6009 | 11.4936 | 3 | 1.27x | 1.35x | 1.36x | 1.076x |
| brick | f32 | 28 | 8 | 14.5784 | 11.5736 | 10.9031 | 10.7300 | 3 | 1.26x | 1.34x | 1.36x | 1.079x |
| brick | f64 | 22 | 4 | 0.4619 | 0.3670 | 0.3488 | 0.3441 | 5 | 1.26x | 1.32x | 1.34x | 1.067x |
| brick | f64 | 22 | 6 | 0.3798 | 0.3119 | 0.2948 | 0.2931 | 5 | 1.22x | 1.29x | 1.30x | 1.064x |
| brick | f64 | 22 | 8 | 0.3852 | 0.3350 | 0.3207 | 0.3175 | 5 | 1.15x | 1.20x | 1.21x | 1.055x |
| brick | f64 | 24 | 4 | 1.9426 | 1.5540 | 1.4598 | 1.4222 | 3 | 1.25x | 1.33x | 1.37x | 1.093x |
| brick | f64 | 24 | 6 | 1.5317 | 1.2285 | 1.1668 | 1.1405 | 3 | 1.25x | 1.31x | 1.34x | 1.077x |
| brick | f64 | 24 | 8 | 1.5072 | 1.2378 | 1.1714 | 1.1643 | 3 | 1.22x | 1.29x | 1.29x | 1.063x |
| brick | f64 | 26 | 4 | 8.3361 | 6.7693 | 6.2903 | 6.1764 | 3 | 1.23x | 1.33x | 1.35x | 1.096x |
| brick | f64 | 26 | 6 | 6.5246 | 5.2423 | 4.8674 | 4.8116 | 3 | 1.24x | 1.34x | 1.36x | 1.090x |
| brick | f64 | 26 | 8 | 6.2919 | 5.1143 | 4.7351 | 4.6477 | 3 | 1.23x | 1.33x | 1.35x | 1.100x |
| qft | f32 | 22 | 4 | 0.0211 | 0.0204 | 0.0206 | 0.0198 | 5 | 1.03x | 1.02x | 1.07x | 1.030x |
| qft | f32 | 22 | 6 | 0.0183 | 0.0178 | 0.0184 | 0.0173 | 5 | 1.03x | 0.99x | 1.06x | 1.029x |
| qft | f32 | 22 | 8 | 0.0203 | 0.0196 | 0.0202 | 0.0183 | 5 | 1.04x | 1.00x | 1.11x | 1.071x |
| qft | f32 | 24 | 4 | 0.0799 | 0.0749 | 0.0795 | 0.0736 | 5 | 1.07x | 1.01x | 1.09x | 1.018x |
| qft | f32 | 24 | 6 | 0.0649 | 0.0615 | 0.0654 | 0.0611 | 5 | 1.06x | 0.99x | 1.06x | 1.007x |
| qft | f32 | 24 | 8 | 0.0679 | 0.0648 | 0.0682 | 0.0627 | 5 | 1.05x | 1.00x | 1.08x | 1.033x |
| qft | f32 | 26 | 4 | 0.3334 | 0.3214 | 0.3372 | 0.3142 | 3 | 1.04x | 0.99x | 1.06x | 1.023x |
| qft | f32 | 26 | 6 | 0.2640 | 0.2547 | 0.2647 | 0.2477 | 3 | 1.04x | 1.00x | 1.07x | 1.028x |
| qft | f32 | 26 | 8 | 0.2611 | 0.2492 | 0.2548 | 0.2449 | 3 | 1.05x | 1.02x | 1.07x | 1.018x |
| qft | f32 | 28 | 4 | 1.4556 | 1.3954 | 1.4353 | 1.3452 | 3 | 1.04x | 1.01x | 1.08x | 1.037x |
| qft | f32 | 28 | 6 | 1.1175 | 1.0726 | 1.1105 | 1.0580 | 3 | 1.04x | 1.01x | 1.06x | 1.014x |
| qft | f32 | 28 | 8 | 1.0620 | 1.0182 | 1.0740 | 1.0021 | 3 | 1.04x | 0.99x | 1.06x | 1.016x |
| qft | f64 | 22 | 4 | 0.0347 | 0.0329 | 0.0350 | 0.0325 | 5 | 1.05x | 0.99x | 1.07x | 1.012x |
| qft | f64 | 22 | 6 | 0.0296 | 0.0287 | 0.0301 | 0.0281 | 5 | 1.03x | 0.98x | 1.05x | 1.021x |
| qft | f64 | 22 | 8 | 0.0332 | 0.0323 | 0.0340 | 0.0318 | 5 | 1.03x | 0.98x | 1.04x | 1.016x |
| qft | f64 | 24 | 4 | 0.1450 | 0.1395 | 0.1479 | 0.1349 | 5 | 1.04x | 0.98x | 1.07x | 1.034x |
| qft | f64 | 24 | 6 | 0.1167 | 0.1123 | 0.1179 | 0.1108 | 5 | 1.04x | 0.99x | 1.05x | 1.014x |
| qft | f64 | 24 | 8 | 0.1203 | 0.1184 | 0.1234 | 0.1161 | 5 | 1.02x | 0.97x | 1.04x | 1.020x |
| qft | f64 | 26 | 4 | 0.6188 | 0.5907 | 0.6137 | 0.5631 | 3 | 1.05x | 1.01x | 1.10x | 1.049x |
| qft | f64 | 26 | 6 | 0.4796 | 0.4538 | 0.4732 | 0.4372 | 3 | 1.06x | 1.01x | 1.10x | 1.038x |
| qft | f64 | 26 | 8 | 0.4590 | 0.4442 | 0.4576 | 0.4261 | 3 | 1.03x | 1.00x | 1.08x | 1.042x |

Sanity check: the branch with `simd=0` gives the same time as main (brickwork-24 f32, 8 threads: 0.8886 vs 0.8893 s), so the whole fma gain comes from `mul_add`.

## 3. Block-size / tile / slot sweep (branch binary, 8 threads at n=26, 6 at n=24, min of 3 (QFT: 5))

Speedups are relative to the 256 KiB default block, untiled, with `simd` on. The b256 anchor is repeated in every cell, so its min is taken over 6-12 runs, which makes the comparison slightly conservative. Raw data: `data/mac-m1/sweep*.raw`. The f32 brickwork rows come from the clean re-run `sweepr`, the f64 brickwork rows from `sweep`, and the QFT rows from `sweepq`.

| workload | b512 | **b1024** | b2048 | b4096 | b1024+t32 | b1024+t64 | b1024+t128 | b1024 slots 4 / 8 |
|---|---|---|---|---|---|---|---|---|
| brick f32 n=24 (6t) | 1.026 | 1.032 | 1.016 | 0.911 | 1.065 | 1.079 | 1.085 | 0.974 / 1.019 |
| brick f32 n=26 (8t) | 1.026 | 1.036 | 1.015 | 0.939 | 1.069 | 1.084 | 1.092 | 0.984 / 1.016 |
| brick f64 n=24 (6t) | 1.035 | 1.044 | 1.024 | 0.908 | 1.094 | 1.109 | 1.106 | 0.999 / 1.028 |
| brick f64 n=26 (8t) | 1.047 | 1.072 | 1.046 | 0.912 | 1.113 | 1.121 | 1.124 | 1.000 / 1.031 |
| QFT f32 n=24 (6t) | 1.008 | 1.011 | 1.082 | 0.935 | | | | |
| QFT f32 n=26 (8t) | 1.029 | 1.094 | 1.031 | 0.910 | | 1.062 | | |
| QFT f64 n=24 (6t) | 1.106 | 1.108 | 1.055 | 0.996 | | | | |
| QFT f64 n=26 (8t) | 1.059 | 1.050 | 1.095 | 0.918 | | 1.036 | | |
| QFT f32 n=28 (8t, vs fma) | | 1.108 | | | | 1.101 | | |
| brick f32 n=28 (8t, vs fma) | | 1.087 | | | | 1.127 | | |

Reading the sweep:
- **1 MiB is the M1 Pro block.** It is the best or within 7% of the best in every row (2 MiB wins QFT f32-24 by 7% and QFT f64-26 by 4%) and never loses to 256 KiB. 4 MiB is always worse, by 6-9%: at 4 MiB x 8 threads, two P-clusters' 12 MiB L2 overflows. 2 MiB loses on brickwork. Slots 6 (the default) beats 4 and 8.
- **Tiling at the best block** adds 4-6% on brickwork (b1024 to b1024+t64 / b1024+t128: f32-24 1.045/1.051, f32-26 1.047/1.054, f64-24 1.062/1.060, f64-26 1.045/1.049; brickwork-28 f32 1.036) and **0.96-1.01x on QFT** (f32-26 0.2250 to 0.2318 s; f64-26 0.4102 to 0.4160 s; QFT-24 8t 0.0620 to 0.0645 s). A 128 KiB tile (the whole P-core L1D) is as good as 64 KiB. The tiling gain at the default block is larger (Section 2) only because a 256 KiB block is too small for the M1 to begin with.

## 4. VPS (x86_64 EPYC-Rome, AVX2): quiet window, 2 threads, nice 15, 1-min load 1.8-2.0 at cell start, min of 5-9

| workload | prec | n | thr | main s | fma s | t32 s | t64 s | reps | main/fma | main/t32 | main/t64 | fma/best-tile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brick | f32 | 22 | 2 | 0.5394 | 0.5399 | 0.4853 | 0.4920 | 9 | 1.00x | 1.11x | 1.10x | 1.113x |
| brick | f32 | 24 | 2 | 2.2971 | 2.2917 | 2.0667 | 2.1371 | 5 | 1.00x | 1.11x | 1.07x | 1.109x |
| brick | f64 | 22 | 2 | 0.7956 | 0.8014 | 0.7331 | 0.7447 | 9 | 0.99x | 1.09x | 1.07x | 1.093x |
| brick | f64 | 24 | 2 | 3.4880 | 3.5434 | 3.2449 | 3.3221 | 5 | 0.98x | 1.07x | 1.05x | 1.092x |
| qft | f32 | 22 | 2 | 0.0466 | 0.0427 | 0.0472 | 0.0441 | 9 | 1.09x | 0.99x | 1.06x | 0.968x |
| qft | f32 | 24 | 2 | 0.1810 | 0.1805 | 0.1971 | 0.1861 | 9 | 1.00x | 0.92x | 0.97x | 0.970x |
| qft | f32 | 26 | 2 | 0.7962 | 0.7989 | 0.8606 | 0.8211 | 5 | 1.00x | 0.93x | 0.97x | 0.973x |
| qft | f64 | 22 | 2 | 0.0691 | 0.0657 | 0.0749 | 0.0707 | 9 | 1.05x | 0.92x | 0.98x | 0.929x |
| qft | f64 | 24 | 2 | 0.2935 | 0.2836 | 0.3292 | 0.3108 | 9 | 1.03x | 0.89x | 0.94x | 0.912x |
| qft | f64 | 26 | 2 | 1.2213 | 1.2368 | 1.4294 | 1.3085 | 5 | 0.99x | 0.85x | 0.93x | 0.945x |

Here `main/fma` is the noise floor (same AVX2 code). Tiling (t32 = the whole 32 KiB L1D on Zen 2) gives **+7-11% on brickwork and -3 to -15% on QFT**. The QFT stages are dominated by diagonal blocks, which `schedule_diag` already merges into one pass per stage, so tiling has little reuse to win back. It also adds overhead: inside a tile run, `apply_diag_group` rebuilds its lo/hi product tables for every tile (2^(l-k) times per block instead of once). This is the likely cause of the QFT loss; I didn't profile it. An earlier VPS attempt at 4 threads under load 5-7 (`vps_noisy.raw`) was unusable: main and fma, which are identical code, differed by up to 1.4x.

## 5. Head-to-head on the Mac (same circuits as RESULTS.md, f32, all 8 threads, min of 3)

qsimcirq 0.22.1 and qiskit-aer 0.17.2 from `~/qsim-bench-venv` (already installed; nothing new was installed), using the RESULTS.md scripts (`data/mac-m1/scripts/{qsim,aer}_bench.py`) with threads set to 8. qsim takes its best of fuse 2/3/4 and Aer its best of fusion on/off. qsim and Aer times include the Python front end, and qsim's also returns the state. qsim-lab times the in-place apply only. The brickwork angles differ by seed (qsim-lab uses 42, the scripts use 7) but the gate structure is identical.

| circuit | qsim-lab main | qsim-lab NEON FMA | qsim-lab FMA + 1 MiB block | qsim (best fuse) | Aer (best fusion) | qsim-lab best vs qsim / Aer |
|---|---|---|---|---|---|---|
| QFT-22 | 0.0203 s | 0.0202 s | **0.0154 s** | 0.362 s (f=3) | 0.310 s (off) | 23x / 20x |
| QFT-24 | 0.0669 s | 0.0661 s | **0.0620 s** | 1.630 s (f=3) | 1.025 s (off) | 26x / 17x |
| QFT-26 | 0.2580 s | 0.2497 s | **0.2285 s** | 6.890 s (f=3) | 4.029 s (off) | 30x / 18x |
| brickwork-22 | 0.2237 s | 0.1885 s | **0.1687 s** | 0.334 s (f=3) | 0.932 s (on) | 2.0x / 5.5x |
| brickwork-24 | 0.8689 s | 0.7173 s | **0.6699 s** | 1.359 s (f=3) | 3.587 s (on) | 2.0x / 5.4x |

(At 6 threads qsim-lab with FMA + 1 MiB runs QFT-22/24/26 in 0.0153/0.0616/0.2240 s and brickwork-22/24 in 0.1762/0.6746 s. Tiling on top gives brickwork 0.161/0.644 s.) This is a second machine for the SOTA claim, with the same caveat RESULTS.md already states: qsim's SIMD kernels target x86 SSE/AVX, so on ARM it runs its portable path. On the x86 VPS a clean run still has **qsim ahead on brickwork** (0.173 s vs qsim-lab 0.30 s at 22 qubits), so the brickwork lead here is a statement about ARM only. The QFT lead (17-30x) comes from diagonal aggregation and holds on both machines. The first qsim QFT pass overlapped another agent's build (`sota.out`, marked `contam=YES`), so the numbers above come from the clean re-run in `sota2.out`.

## 6. Confirmation: exp/neon-fma-r4 defaults vs main (Mac, interleaved, min of 3)

`exp/neon-fma-r4` (47c34fd) built in its own target dir, with `BlockConfig::default()` and no keys, against main fb30f56. The full `cargo test --release` on aarch64 is green (24 test binaries + doc-tests, 0 failures).

| workload | prec | n | thr | main s | neon-fma-r4 default s | reps | speedup |
|---|---|---|---|---|---|---|---|
| brick | f32 | 24 | 6 | 0.8781 | 0.6798 | 3 | 1.29x |
| brick | f64 | 26 | 8 | 6.3810 | 4.8532 | 3 | 1.31x |
| qft | f32 | 22 | 8 | 0.0201 | 0.0151 | 3 | 1.33x |
| qft | f32 | 24 | 8 | 0.0672 | 0.0642 | 3 | 1.05x |
| qft | f32 | 26 | 8 | 0.2565 | 0.2242 | 3 | 1.14x |

This is what merging exp/neon-fma-r4 gives on Apple silicon: **1.29-1.31x on brickwork and 1.05-1.33x on QFT**. x86_64 code is unchanged.

## Files
- Mac raw data, logs, and harness: `research/data/mac-m1/`. `*.raw` lines are `threads binary | workload | n | prec | config | seconds | ...`. Summarise with `scripts/parse.py` (A/B) and `scripts/sweep.py` (sweep); the drivers are `scripts/cell.sh` and `phase1-5.sh`.
- Earlier Mac sweeps by the round-3 agent (`research/data/l1/e*`, 6 threads, with Logic Pro not running) agree: tiling gave 1.03-1.06x at a fixed block, and the best config was b1024 + t32/t64.

## 7. Status on main (4 Oct 2026, integration pass)

The tiling code was merged to main behind its flag so the code and this negative result live together: `BlockConfig::l1_tile_bytes` stays **0 (off) by default** on both architectures. Rebased onto main 3a37671 (only conflict: this file, where main's version was already the superset), reviewed again (ops are reordered only across pairs that commute under the `schedule_diag` rule; `tile_local` requires every bit a diagonal block reads, and every target, to lie inside the tile), and gated with fmt, clippy `--all-targets -D warnings`, the full `cargo test --release` on x86_64 (53 test binaries, 541 passed, 0 failed) and the doc build. `tests/engines/l1_tiling.rs` (incl. `tiled_matches_audit_reference` against `audit_common::RefSv`) and the tiling configs in `tests/engines/blocked.rs` keep it tested. The original branches are archived (`research/process/ARCHIVE.md`).
