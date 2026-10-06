"""R-only QR: numpy.linalg.qr(mode='r') vs direct LAPACK geqrf (tnoq_fast._qr_r), single-thread, CPU time."""
import numpy as np, time, tnoq_fast as T
rng = np.random.default_rng(0)
for shp in [(8, 4), (32, 8), (128, 16), (512, 16), (4096, 16)]:
    M = rng.standard_normal(shp) + 1j * rng.standard_normal(shp)
    T.QR_IMPL = 'numpy'; a = T._qr_r(M); T.QR_IMPL = 'lapack'; b = T._qr_r(M)
    ok = np.allclose(a.conj().T @ a, b.conj().T @ b)
    ts = []
    for impl in ('numpy', 'lapack'):
        T.QR_IMPL = impl; N = 300; t = time.process_time()
        for _ in range(N): T._qr_r(M)
        ts.append((time.process_time() - t) / N * 1e6)
    print(shp, 'R^H R equal', ok, 'CPU us/call numpy %.1f lapack %.1f' % tuple(ts), flush=True)
