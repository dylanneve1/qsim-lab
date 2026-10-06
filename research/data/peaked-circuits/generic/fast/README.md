# /tmp/mlx-port: MLX port of middle-out TNO growth, with profile and speedups

The short answer is that **MLX does not help this workload.** The growth is bound by Python and quimb overhead
plus many small QR/SVDs, and MLX 0.32 has no GPU QR/SVD/eigh (complex QR isn't available even on its CPU path).
The MLX backend works and gives identical peaks, but it is **13–16× slower** on P4/P10 and **4–7× slower** on P9
tnoq growth. complex64 on the CPU gives at most ~1.0–1.4×.
The large win was algorithmic. The swap-aware P9 route (tno.py/tnos.py) spent 96% of its time in full SVDs of
merged-pair matrices. `tno_fast.py` replaces these with QR-reduced core SVDs, which have the same singular values
and give the same trajectory step for step. Measured on the Mac, it is 20–70× faster where the original
stalls and far faster after that (see the table).
For the tnoq/solve_generic path, `tnoq_fast.py` is a lean numpy re-implementation of the quimb compression. It gives the
same trajectory and the same P4/P10 peaks, with 1.7× less CPU for the full solves and 2× on P9 growth (section 7).

Every number below comes from a log in `logs/` (VPS) or `maclogs/` (Mac, pulled from /tmp/peaked-mac/mlx/logs).
**Caveat:** other agents' P9/P6 jobs loaded the Mac heavily during all runs (load average 10–26 on 8 cores), and
the VPS had load ~8 on 4 cores. Absolute times are noisy at ±30% or more. Compare only ratios that are much larger than that.
Peaks are compared by sha1 hash and are never written out.

## Files
| file | what |
|---|---|
| backend.py | array backends `np128` (reference), `np64` (numpy complex64), `mlx` (mlx complex64, GPU contractions), `mlxcpu` (mlx on CPU stream). Registers mlx versions of quimb's decomposition kernels and autoray linalg as numpy round trips. |
| tnob.py | tnoq.py with a selectable backend, timers, and a wall-clock cap (same algorithm). |
| solve_b.py | solve_generic.py using tnob (growth on the backend, final exact contraction always numpy complex128). |
| tno_fast.py | QR-reduced `TNO.gate` / `unswap_pass` for the swap-aware route. `tno_fast.install()` monkeypatches tno.py. |
| bench.py | `solve FILE BACKEND` (full pipeline) and `grow FILE C BACKEND MAX_ELEMS TMAX [prof]` (tnoq growth). |
| bench_tnos.py | swap-aware P9 growth (tnos.grow_adaptive, unswap=True), variants orig/orig64/fast1/fast/fast64. |
| prof.py, prof_tnos.py | cProfile drivers. kernels.py: kernel micro-benchmarks. probe_mlx.py: what MLX supports. |
| tnoq_fast.py | lean numpy re-implementation of tnoq growth + quimb virtual-tree compression (drop-in `grow`; options DTYPE, BATCH_QR, QR_IMPL). solve_f.py = solve_generic using it. |
| bench_fast.py, fsum.py, qr_micro.py, chain_vps.sh, chain2_vps.sh | VPS benchmarks of tnoq_fast vs tnoq (single-thread BLAS, CPU time recorded). |
| test_fast.py | validation of tno_fast (test_tno.py suite + random-circuit equivalence). compare_tnos.py, table.py: summaries. |
| gparse.py, struct_probe.py, tnoq.py, tno.py, tnos.py, solve_generic.py, test_tno.py | unmodified copies from /tmp/peaked-generic. |

Mac setup: `~/mlx-venv` has numpy 2.5.3 (Accelerate), scipy 1.18.1, quimb 1.11.2, cotengra 0.8.2, autoray 0.11.0,
numba 0.68.0, and mlx 0.32.3. Work dir: /tmp/peaked-mac/mlx. Chains: chain1–5.sh.

## 1. Profile (where the time goes)
| run | total | breakdown |
|---|---|---|
| P4 full `solve_generic` (VPS, cProfile, `logs/prof_P4_vps.log`) | 197 s | centre scan 147 s (75%), main growth 12 s. The growth is all quimb `tensor_network_ag_compress`: virtual-tree gauging 75 s, 373k tiny QR splits 46 s, cotengra `auto-hq` path search for per-site contractions 27 s. By leaf: quimb Python 36%, autoray dispatch 16%, cotengra 14%, numpy misc 5%, **tensordot 3%, SVD 2%, eigh 0.5%**. W ≤ 2e4 elements, max bond 14. |
| P9 tnoq growth c=50 → 1.2e6 elements (Mac, cProfile, `maclogs/p9_grow_np128_prof.log`) | 30 s | All of it in compress. ~65k QRs (numba) ≈ 11 s (36%), tensordot 3.6 s (12%), the rest (~50%) quimb/autoray Python. |
| P9 swap-aware `tnos.grow_adaptive(unswap=True)` c=50 (VPS, cProfile, `logs/prof_tnos_p9_vps.log`) | 624 s (stopped) | **602 s (96.5%) in `numpy.linalg.svd` inside `tno.TNO._split`.** That is 2681 SVDs of the full merged-pair matrix (0.22 s each): unswap_pass 419 s, gate 187 s. quimb compress 18 s, tensordot 2.7 s. |

So in tnoq at most ~12% (P9) or ~3% (P4) of the time is contraction work that a GPU could take over. In tnos the
time is LAPACK SVD, which MLX can't run on the GPU.

## 2. What MLX 0.32.3 can do on the M1 Pro (`probe_mlx.py`)
- GPU: complex64 matmul, tensordot, einsum, and conj all work.
- `linalg.qr/svd/eigh/cholesky`: not supported on the GPU for any dtype.
- On the CPU stream: complex SVD works, but only as full_matrices (no economy SVD). Complex eigh works.
  **Complex QR and Cholesky are rejected even on the CPU.**

So `backend.py` routes every decomposition GPU → numpy/LAPACK → GPU.

## 3. MLX backend: validation and timings (Mac, full pipeline, `bench.py solve`, `maclogs/solve_*.log`)
Growth (centre scan and main growth) runs on the given backend. The final exact contraction always runs in numpy complex128.
| circuit | backend | total s | scan s | peak = np128 peak | p | \|Δp\| vs np128 |
|---|---|---|---|---|---|---|
| P10 heavy_hex_4020 | np128 | 63.0 | 44.5 | (ref; also = earlier VPS P10 peak) | 0.703448 | 0 |
| | np64 | 69.1 | 44.2 | yes | 0.703448 | 7.5e-8 |
| | **mlx** | **806.6** | 737.2 | yes | 0.703448 | 4.8e-8 |
| | mlxcpu | 95.2 | 61.6 | yes | 0.703448 | 1.8e-7 |
| P4 golden_mountain | np128 | 38.4 | 27.1 | (ref; also = parent's P4 peak) | 0.010455 | 0 |
| | np64 | 38.2 | 26.1 | yes | 0.010455 | 2.2e-8 |
| | **mlx** | **616.5** | 564.0 | yes | 0.010455 | 1.0e-9 |
| | mlxcpu | 189.1 | 165.3 | yes | 0.010455 | 1.5e-8 |

All four backends find the same peak and agree on p to ≤2e-7, well inside the 1e-3 requirement.
MLX on the GPU is **12.8× (P10) and 16× (P4) slower**. Even MLX on the CPU is 1.5–5× slower, which is
pure MLX dispatch overhead on tiny arrays. complex64 numpy: no change (0.9–1.0×).

P9 tnoq growth c=50, 12 steps to 1.2e6 elements, identical elems/bond trajectory for all backends (`bench.py grow`):
| backend | runs (s) | last step (5e5→1.2e6) s |
|---|---|---|
| np128 | 17.5, 16.2 | 9.2, 8.7 |
| np64 | 12.8, 11.4 (+ 25.9 in an earlier, busier slot) | 6.2, 5.4 |
| **mlx** | **115.1, 68.0, 84.5** | 39.7, 18.5, 39.8 |
| mlxcpu | 54.0 | 23.2 |
=> complex64 on CPU: **~1.4×**. MLX GPU: **~4–7× slower**.

## 4. Kernel micro-benchmarks (Mac, `kernels.py`, `maclogs/kernels.log`, median of 7)
| op | np128 | np64 | mlx GPU | mlx GPU incl. np↔mx transfer |
|---|---|---|---|---|
| matmul 64² | 0.12 ms | 0.03 ms | 0.50 ms | 0.49 ms |
| matmul 256² | 0.53 ms | 0.14 ms | 0.44 ms | 0.53 ms |
| matmul 512² | 7.0 ms | 1.8 ms | 1.6 ms | 2.2 ms |
| matmul 1024² | 49 ms | 16 ms | 7.5 ms | 9.3 ms |
| matmul 2048² | 438 ms | 134 ms | 54 ms | 61 ms |
| tensordot site-like 16k elems (d=8) | 0.05 ms | 0.02 ms | 0.55 ms | 0.53 ms |
| tensordot 262k elems (d=16) | 1.5 ms | 0.5 ms | 0.8 ms | 1.4 ms |
| tensordot 4.2M elems (d=32) | 18 ms | 9.5 ms | 5.6 ms | 11.3 ms |

- QR (numpy only, since MLX has no complex QR): complex64 gives no gain (4096×64: 39 vs 37 ms; 16384×256: 1483 vs 1459 ms).
- SVD n×n, np128 / np64 / mlx-cpu (c64, full): 64: 1.1 / 1.4 / 0.7 ms. 256: 27 / 29 / 15 ms. 1024: 2110 / 2063 / 638 ms.
- An MLX GPU call has a floor of ~0.35–0.5 ms. MLX's GPU only wins for single contractions of ≳1M elements
  (matmul ≳ 512²), and the transfer roughly halves that gain. The tensors in this workload are almost all far below that size.

## 5. The speedup that works: QR-reduced splits for the swap-aware route (`tno_fast.py`)
`TNO._split` SVDs the whole merged pair `C.reshape(dl, dr)`, but `rank ≤ bond × 4`. `tno_fast` QR-reduces each
tensor over the legs that stay with it (`A_x = Q_x R_x`). It then SVDs only `core = R_x·R_y[·G]`, which has
the same singular values, and maps U/V back with Q. Level 2, the default, also chooses between the 2 (gate) or
4 (unswap) leg assignments using singular values alone, then does one full split, and only when a change is committed.
The truncation rule, ranks and discarded weight are unchanged. Use it with `import tno_fast; tno_fast.install()` before
growing (it monkeypatches `tno.TNO.gate` and `tno.unswap_pass`).

Validation:
- `test_fast.py`: the full `test_tno.py` suite passes, including the hidden-relabelling mirror, unswap exactness, and the 3-CZ swaps.
  On 20 random 4q/10-gate circuits with unswap passes, the rank/accept sequences are identical to the original
  and the dense operators agree to 6e-15. This holds on both the VPS and the Mac.
- P9 c=50 (`bench_tnos.py`, cutoff 1e-3, local_cutoff 1e-8): the trajectory (lo, hi, side, elems, max_bond, nbonds,
  moved) is **identical to the original on all 20 common steps**. This was checked on the Mac and on the VPS (`compare_tnos.py`).

P9 swap-aware growth, Mac, time to reach each step (`maclogs/p9_tnos_{orig,fast1,fast,fast_long}.log`):
| step [lo,hi) | elems | orig | fast1 (QR only) | fast (QR + svals) |
|---|---|---|---|---|
| [42,57) | 8 496 | 3.9 s | 2.9 s | 2.5 s |
| [40,57) | 18 336 | 31.7 s | 3.9 s | 3.3 s |
| [38,57) | 34 024 | 156.8 s | 5.5 s | 4.6 s |
| [37,57) | 53 232 | 388.2 s | 6.8 s | 5.6 s |
| [36,57) | 82 976 | not reached in 900 s | 8.7 s | 7.2 s |
| [36,60) | 363 184 | – | 22.7 s | 18.8 s |
| [36,65) | 3 468 032 | – | – | 200.2 s (`fast_long`, peak RSS 993 MB) |

At [37,57) that is **69×** cumulative. The single step [38,57)→[37,57) took 231 s originally and 1.0 s with fast.
The original's peak RSS was 3.5 GB (the full SVDs). The whole fast run to 3.6e5 elements peaked at 0.25 GB.
On the VPS (busier, `logs/p9_tnos_*_vps.log`) the trajectory was identical and [38,57) took 18.1 s (fast) and 24–25 s (fast1, two runs), against 574 s for the original under cProfile (`logs/prof_tnos_p9_vps.log`).
After this change the time is spread across quimb `canonical_compress` (~50%: Python, QR, tensordot), core SVDs and
QRs (`logs/prof_tnos_fast_p9_vps.log`, a level-1 profile).

## 6. complex64 on the CPU as a separate data point
- tnoq (P4/P10 full solve): 0.91× and 1.0× (no gain). P9 tnoq growth to 1.2e6: **1.37×** (np128 17.5 / 16.2 s →
  np64 12.8 / 11.4 s). The peaks are the same and |Δp| ≤ 7.5e-8.
- Swap-aware route: **complex64 is not a drop-in replacement.** With local_cutoff 1e-8, float32 round-off (~1e-7 relative)
  hides the exact rank drops. The trajectory diverges at step 3 and finds 40–42 relabelled wires instead of 52
  (`fast64`, `orig64` logs; the orig64 run was ended externally by SIGTERM after 414 s and has no RESULT line).
  With local_cutoff 1e-5, complex64 reproduces the complex128 trajectory exactly over 25 steps and is 1.2× faster
  (26.4 → 21.9 s to 4.7e5 elements; `p9_tnos_fast{,64}_lc1e-5.log`). Note that lc=1e-5 gives a different trajectory from lc=1e-8.

## 7. Lean numpy path for tnoq (`tnoq_fast.py`, VPS)
quimb's virtual-tree compression already does "QR the two sides, then SVD the small `R_l R_r^T`", so the idea
behind `tno_fast` was already there. Its cost was Python object churn. `tnoq_fast.py` re-implements the same algorithm
with plain ndarrays and integer bond ids:
- gates are contracted into the site tensors at once, and multi-bonds are fused;
- each bond is compressed in turn: a tree span with r=3 and quimb's scoring, reduced factors absorbed from the leaves inwards,
  oblique projectors from the truncated SVD of `R_l R_r^T` with quimb's default `rsum2` cutoff;
- dim-1 bonds are squeezed and the norms equalised.

It's a drop-in `grow()` (and `solve_f.py`). Options:
- `DTYPE` (complex64)
- `BATCH_QR` (stack same-shape leaf QRs of a tree into one `np.linalg.qr` call)
- `QR_IMPL='lapack'` (call `?geqrf` directly, without numpy's per-call wrapper overhead: `logs/qr_micro.log`, 1.3–2.2× per small QR)

The VPS was shared (load 8–22 on 4 cores), so all runs use single-thread BLAS and report **process CPU time**.
Wall times are in the logs but aren't comparable.

P9 tnoq growth c=50 → 1.2e6 elements, 12 steps (`logs/fast_grow_P9_*_t1.log`, `fsum.py`). The **elems / max_bond / nbonds
trajectory is identical to quimb at every step for every variant**:
| variant | CPU s | vs quimb |
|---|---|---|
| quimb (tnoq) | 20.5, 19.0 | 1 |
| fast c128 | 14.9 | 1.3–1.4× |
| fast c128 + batch | 13.7 | 1.4–1.5× |
| fast c64 | 13.0 | 1.5–1.6× |
| fast c64 + batch | 12.4 | 1.5–1.65× |
| fast c128 + lapack | 11.1 | 1.7–1.85× |
| **fast c64 + batch + lapack** | **9.6** | **2.0–2.1×** |

Full solve_generic pipeline (`logs/fsolve_*.log`). Growth/scan uses tnoq_fast; the final exact contraction is unchanged:
| circuit | variant | CPU s | scan s (wall) | peak = quimb peak | p | \|Δp\| |
|---|---|---|---|---|---|---|
| P10 | quimb | 90.0 | 176 (load spike) | ref | 0.703448 | 0 |
| | fast c128 | 58.8 | 43.4 | yes | 0.703239 | 2.1e-4 |
| | fast c64 | 59.2 | 44.2 | yes | 0.703239 | 2.1e-4 |
| | fast c128 + batch | 59.6 | 44.6 | yes | 0.703239 | 2.1e-4 |
| | **fast c64 + batch + lapack** | **51.9** | 35.3 | yes | 0.703239 | 2.1e-4 |
| P4 | quimb | 77.9 | 67.0 | ref | 0.010455 | 0 |
| | fast c128 | 51.4 | 38.5 | yes | 0.010456 | 1.2e-6 |
| | fast c64 | 53.1 | 39.7 | yes | 0.010456 | 1.2e-6 |
| | fast c128 + batch | 53.3 | 40.4 | yes | 0.010456 | 1.2e-6 |
| | **fast c64 + batch + lapack** | **46.6** | 34.1 | yes | 0.010456 | 1.2e-6 |

All variants find the same peak, and |Δp| ≤ 2.1e-4 is inside the 1e-3 requirement. The p difference comes from the bond
compression order (by bond id, where quimb uses its index-map order), not from precision: c64 and c128 give the same p.
The whole pipeline is **1.7× less CPU**, and the scan/growth phase alone is ~2× faster (P4 scan 67 → 34 s).
What's left is the unchanged final contraction (~10–14 s) and, inside tnoq_fast, numpy tensordot, reshape copies and QR.
Batching barely helps (same-shape leaves are rare). complex64 helps only on the large P9 operators.

## 8. Bottom line and recommendations
1. Don't use MLX for this code. The work is dominated by the dispatch and overhead of hundreds of thousands of tiny
   ops, and by QR/SVD, which MLX can't run on the GPU. It is 4–16× slower measured end to end.
2. For P9 swap-aware growth, use `tno_fast.install()`. It gives identical results and is 20–70× faster at the point where the original
   stalled, and it reached 3.5e6 elements in 200 s within 1 GB.
3. For tnoq / solve_generic, use `tnoq_fast` (`solve_f.py`, with DTYPE=complex64, BATCH_QR, QR_IMPL='lapack'). It has an
   identical trajectory, the same peaks, and needs about half the growth time (1.7× less total CPU on P4/P10, 2× on P9 growth).
4. P4/P10 are still dominated by the centre scan (~70% of the wall time). The scan is independent per centre and would
   parallelise, but this was **not measured** here because both machines were saturated.
