# MLX port of middle-out TNO growth — running notes (2026-10-06)

Code copied from /tmp/peaked-generic (tnoq.py, tno.py, tnos.py, solve_generic.py, gparse.py, struct_probe.py), unmodified.
New: backend.py (np128 / np64 / mlx / mlxcpu array backends + quimb/autoray registrations for mlx),
tnob.py (tnoq with selectable backend + timers), solve_b.py (= solve_generic.py using tnob), bench.py, prof.py, prof_tnos.py.
Mac: /tmp/peaked-mac/mlx, venv ~/mlx-venv (numpy 2.5.3/Accelerate, scipy 1.18.1, quimb 1.11.2, cotengra 0.8.2,
autoray 0.11.0, numba 0.68.0, mlx 0.32.3). Mac was heavily loaded by other agents' runs during all
measurements (load average 16-23 on 8 cores) -> absolute times are noisy; backends were run back-to-back.

## MLX capabilities (probe_mlx.py, mlx 0.32.3, M1 Pro)
- GPU: complex64 matmul / tensordot / einsum / conj OK.
- linalg.qr, svd, eigh, cholesky: NOT supported on the GPU at all (any dtype) -> "pass a CPU stream".
- CPU stream: svd (complex64 OK, but only full_matrices - no economy SVD), eigh complex OK;
  qr and cholesky reject complex64 even on CPU.
=> in the MLX backend every decomposition must go GPU -> CPU (numpy/LAPACK) -> GPU.

## 1. Profile of the current code (cProfile)
### P4 full solve_generic (VPS, prof.py, logs/prof_P4_vps.log; 197 s under cProfile, VPS load ~8 on 4 cores)
- centre scan 147 s (75%), main growth 12 s, final contraction/marginals the rest. Growth = tnoq.compress =
  quimb tensor_network_ag_compress (local-late): 1150 calls, 138 s of 196 s.
- inside: virtual-tree gauging (_compute_tree_gauges 75 s, 50k calls), compute_reduced_factor/tensor_split 46 s
  (373k QR splits of tiny tensors), cotengra 'auto-hq' path search for the per-site contractions 27 s (14%).
- by leaf category: quimb Python 36%, autoray dispatch 16%, cotengra path finding 14%, numpy misc 5%,
  tensordot 3%, SVD 2%, eigh 0.5%.  => >90% interpreter / bookkeeping overhead, <10% linear algebra.
  W never exceeds ~2e4 elements (max bond 14). Nothing here is GPU-sized.
### P9 tnoq growth from c=50 to 1.2e6 elements (Mac, bench.py grow ... prof, logs/p9_grow_np128_prof.log)
- 12 layer-pair steps, 30.3 s under cProfile; last step (5e5 -> 1.2e6 elems, 267 bonds, max bond 16) 11 s.
- 100% in compress; _compute_tree_gauges 23.7 s; 65k QR (numba qr_stabilized, shows as tottime of
  autoray Composed.__call__) ~10.9 s = 36%; numpy tensordot 3.6 s = 12%; rest quimb/autoray Python (~50%).
### P9 swap-aware growth tnos.grow_adaptive(unswap=True) from c=50 (VPS, prof_tnos.py, logs/prof_tnos_p9_vps.log)
- stopped after 624 s at [38,57), 34k elements (steps 160-290 s each, growing).
- **602 s of 624 s (96.5%) in numpy.linalg.svd inside tno.TNO._split** (2681 calls, 0.225 s each):
  unswap_pass 419 s, TNO.gate 187 s. quimb canonical_compress only 18 s. tensordot 2.7 s.
  The SVDs are of the FULL merged-pair matrix C.reshape(dl, dr) although its rank is <= bond x 4.

## 2. MLX backend for the tnoq hot path (backend.py + tnob.py), validation P4/P10 (Mac, chain1.sh)
- Design: W tensors are mlx complex64 arrays; quimb/cotengra contractions dispatch through autoray to
  mx.tensordot/einsum on the GPU; quimb decomposition kernels (svd_truncated, qr_stabilized, eigh_truncated, ...)
  and autoray linalg.{qr,svd,eigh,inv,...} are registered for 'mlx' as numpy round trips (MLX has no GPU linalg).
  Final exact contraction always numpy complex128 (snapshots converted), so only the growth precision changes.
- Full solve_generic pipeline (bench.py solve), Mac, same session (load 10-25):
    P10: np128 63.0 s | np64 69.1 s | mlx 806.6 s | mlxcpu 95.2 s ; peak identical in all four (sha1 match,
         also = earlier VPS P10 peak), |dp| <= 1.8e-7 vs np128.
    P4 : np128 38.4 s | np64 38.2 s | mlx 616.5 s | mlxcpu 189.1 s ; peak identical in all four (= earlier
         parent/VPS P4 peak), |dp| <= 2.2e-8.
  => MLX GPU is 13-16x SLOWER on P4/P10 (tiny tensors: every op pays Metal dispatch + a GPU->CPU sync for each
     QR/SVD). complex64 on CPU: no gain (overhead-bound).
- P9 tnoq growth c=50 -> 1.2e6 elements (12 steps; bench.py grow):
    np128 17.5 s (r1) | np64 12.8 s (r1) | mlx 115.1 s | mlxcpu 54.0 s.  Identical elems/bond trajectory.

## 3. Swap-aware route (tno.py/tnos.py): QR-reduced splits (tno_fast.py)
- tno_fast.gate_fast / unswap_pass_fast: QR-reduce each tensor over the legs that stay with it, SVD only the
  small core (singular values identical to the full merged matrix). Level 2 (default): choose the leg
  assignment from singular values only (compute_uv=False), one full split for the chosen/committed one.
- test_fast.py: test_tno.py suite passes with the fast version; 20 random 4q/10-gate circuits with unswap passes:
  identical rank/accept sequences vs original, dense operators agree to 6e-15 (VPS and Mac).
- P9 c=50 adaptive+unswap (bench_tnos.py, cutoff 1e-3, local_cutoff 1e-8): fast reproduces the original
  trajectory step for step (lo, hi, side, elems, max_bond, nbonds, moved) on all common steps (VPS and Mac).
  Mac: original reaches [37,57) (53k elems) at 388 s and was stopped at 900 s inside the next step
  (peak RSS 3.5 GB from the full SVDs); fast reaches [36,60) (3.6e5 elems) in 18.8 s.
  VPS (noisy, load ~8 on 4 cores): orig under cProfile [38,57) at 574 s; fast1 25.3 / 24.3 s, fast 18.1 s (compare_tnos.py).
- complex64 + local_cutoff 1e-8 (fast64): trajectory diverges at step 3 (804 vs 756 elems, moved 0 vs 4; later
  moved 42 vs 52): float32 round-off (~1e-7 relative) exceeds the 1e-8 rank cutoff, so exact rank drops /
  relabellings are no longer detected. complex64 is NOT a drop-in for the swap-aware route.
- note: logs/prof_tnos_fast_p9_vps.log is a cProfile of the level-1 version (it motivated level 2).
- Mac (maclogs/): orig [37,57) at 388.2 s, stopped at 900 s, peak RSS 3.5 GB. fast1 [37,57) at 6.8 s, fast 5.6 s.
  fast_long (max_elems 3e6): [36,65) with 3.47e6 elements, max bond 16, 190 bonds at 200 s, RSS 993 MB,
  trajectory identical to fast/orig on all common steps.
- complex64 with local_cutoff 1e-5: fast64 == fast (complex128, same lc) on all 25 steps, 21.9 vs 26.4 s.
- orig64 run was SIGTERMed externally at ~414 s (not by me; no RESULT line). Its partial log shows the same divergence (moved 40).
- VPS: fast (level 2) [38,57) 18.1 s; fast1 runs 25.3 / 24.3 s; orig under cProfile 573.6 s.

## 4. Kernels (kernels.py, Mac): the MLX GPU floor is ~0.4 ms per call. It beats numpy complex64 only for matmul >= 512^2
   (1024^2: 7.5 vs 15.8 ms) and tensordot ~4M elements (5.6 vs 9.5 ms, 11.3 with transfer). Complex QR in numpy:
   c64 = c128 speed. SVD: MLX-cpu (c64, full matrices) is 2-3x faster than numpy's at n >= 256.

## Verdict
MLX: 13-16x slower on P4/P10, 4-7x slower on P9 tnoq growth; peaks identical, |dp| <= 2e-7. complex64 CPU: 1.0x
(P4/P10), 1.37x (P9 tnoq). It is unusable for the swap-aware route at local_cutoff 1e-8, and 1.2x at lc 1e-5.
Real win: tno_fast (QR-reduced splits), identical trajectory, 69x to [37,57), and the route now reaches
3.5e6 elements in 200 s / <1 GB. README.md has the tables.

## 5. tnoq_fast.py (lean numpy tnoq + virtual-tree compression), VPS, single-thread BLAS, CPU seconds
- quimb's compress_between virtual-tree already does QR-reduced factors + small SVD; its cost was object overhead.
- Trajectory (elems/max_bond/nbonds) identical to quimb on P10 grow from 110 (112 steps) and P9 to 1.2e6 (12 steps),
  for c128/c64/batch/lapack variants.
- P9 grow CPU: quimb 20.5/19.0 | fast 14.9 | +batch 13.7 | c64 13.0 | c64+batch 12.4 | +lapack 11.1 | c64+batch+lapack 9.6.
- Solves (CPU): P10 quimb 90.0 -> fast 58.8 (c64 59.2, batch 59.6, c64+batch+lapack 51.9); P4 quimb 77.9 -> 51.4
  (c64+batch+lapack 46.6). Same peaks; |dp| 2.1e-4 (P10), 1.2e-6 (P4), set by the bond order, not dtype.
- pitfall (again): `pkill -f bench_fast.py` from a shell whose command line contains that string kills the shell.
- After the parent's 21:12 "no more Mac jobs" note (not read in time), the queued chains3-5 ran on the Mac until ~22:30
  (fast1 23 s, fast_long 200 s, 2 x lc1e-5 ~25 s, kernels ~2 min). Nothing of mine is still running there.
