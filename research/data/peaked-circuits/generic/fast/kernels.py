"""Kernel micro-benchmarks on the Mac: complex matmul, tall-skinny QR, SVD for numpy c128 / numpy c64 /
mlx c64 GPU (compute only, and including numpy<->mlx transfer).  Median of R repeats (load on the machine)."""
import time, json, numpy as np
import mlx.core as mx

def med(f, r=7):
    f()
    ts = []
    for _ in range(r):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

rng = np.random.default_rng(0)
def cr(*s, dt=np.complex128):
    return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(dt)

out = []
print("matmul n x n (ms):  np128   np64   mlx-gpu   mlx-gpu+xfer   mlx-cpu")
for n in (16, 64, 128, 256, 512, 1024, 2048):
    A = cr(n, n); B = cr(n, n); A64, B64 = A.astype(np.complex64), B.astype(np.complex64)
    a, b = mx.array(A64), mx.array(B64); mx.eval(a, b)
    r = dict(op='matmul', n=n, np128=med(lambda: A @ B), np64=med(lambda: A64 @ B64),
             mlx_gpu=med(lambda: mx.eval(mx.matmul(a, b, stream=mx.gpu))),
             mlx_gpu_xfer=med(lambda: np.array(mx.matmul(mx.array(A64), mx.array(B64), stream=mx.gpu))),
             mlx_cpu=med(lambda: mx.eval(mx.matmul(a, b, stream=mx.cpu))))
    out.append(r); print(f"  {n:5d}  " + "  ".join(f"{1e3*r[k]:8.3f}" for k in ('np128', 'np64', 'mlx_gpu', 'mlx_gpu_xfer', 'mlx_cpu')), flush=True)
print("tensordot site-like: T(16,16,16,16,4) x R(16,16) over one bond (ms)")
for d in (8, 16, 32):
    T = cr(d, d, d, d, 4); R = cr(d, d); T64, R64 = T.astype(np.complex64), R.astype(np.complex64)
    t, rr = mx.array(T64), mx.array(R64)
    r = dict(op='tensordot', d=d, size=T.size, np128=med(lambda: np.tensordot(T, R, axes=([0], [0]))),
             np64=med(lambda: np.tensordot(T64, R64, axes=([0], [0]))),
             mlx_gpu=med(lambda: mx.eval(mx.tensordot(t, rr, axes=([0], [0]), stream=mx.gpu))),
             mlx_gpu_xfer=med(lambda: np.array(mx.tensordot(mx.array(T64), mx.array(R64), axes=([0], [0]), stream=mx.gpu))))
    out.append(r); print(f"  d={d:3d} size {T.size:8d} " + "  ".join(f"{1e3*r[k]:8.3f}" for k in ('np128', 'np64', 'mlx_gpu', 'mlx_gpu_xfer')), flush=True)
print("QR m x k (ms):  np128  np64 (numpy LAPACK; MLX has no complex QR)")
for m, k in ((64, 16), (256, 16), (4096, 16), (4096, 64), (65536, 16), (16384, 256)):
    A = cr(m, k); A64 = A.astype(np.complex64)
    r = dict(op='qr', m=m, k=k, np128=med(lambda: np.linalg.qr(A)), np64=med(lambda: np.linalg.qr(A64)))
    out.append(r); print(f"  {m:6d}x{k:<4d} {1e3*r['np128']:9.3f} {1e3*r['np64']:9.3f}", flush=True)
print("SVD n x n (ms):  np128  np64  mlx-cpu(c64, full)")
for n in (16, 64, 256, 1024):
    A = cr(n, n); A64 = A.astype(np.complex64); a = mx.array(A64)
    r = dict(op='svd', n=n, np128=med(lambda: np.linalg.svd(A, full_matrices=False)),
             np64=med(lambda: np.linalg.svd(A64, full_matrices=False)),
             mlx_cpu=med(lambda: mx.eval(*mx.linalg.svd(a, stream=mx.cpu)), r=3))
    out.append(r); print(f"  {n:5d} {1e3*r['np128']:9.3f} {1e3*r['np64']:9.3f} {1e3*r['mlx_cpu']:9.3f}", flush=True)
X = cr(100000, dt=np.complex64)
print("round trip numpy->mlx->numpy, 1e5 complex64 elements (ms):", round(1e3 * med(lambda: np.array(mx.array(X))), 3))
print("RESULT", json.dumps(out))
