import mlx.core as mx, numpy as np, time
print("mlx", mx.__version__, "default device", mx.default_device(), "metal", mx.metal.is_available())
A = (np.random.randn(256, 256) + 1j*np.random.randn(256, 256)).astype(np.complex64)
a = mx.array(A)
def tryop(name, f):
    for dev in (mx.gpu, mx.cpu):
        try:
            r = f(dev); mx.eval(r); print(f"  {name:12s} {str(dev):16s} OK")
        except Exception as e:
            print(f"  {name:12s} {str(dev):16s} FAIL: {str(e)[:110]}")
tryop("matmul c64", lambda d: mx.matmul(a, a, stream=d))
tryop("tensordot", lambda d: mx.tensordot(a.reshape(16,16,256), a.reshape(256,16,16), axes=1, stream=d))
tryop("einsum", lambda d: mx.einsum("ij,jk->ik", a, a, stream=d))
tryop("qr c64", lambda d: mx.linalg.qr(a, stream=d))
tryop("svd c64", lambda d: mx.linalg.svd(a, stream=d))
tryop("eigh c64", lambda d: mx.linalg.eigh(a @ mx.conj(a.T), stream=d))
tryop("qr f32", lambda d: mx.linalg.qr(mx.real(a), stream=d))
tryop("svd f32", lambda d: mx.linalg.svd(mx.real(a), stream=d))
tryop("cholesky c", lambda d: mx.linalg.cholesky(a @ mx.conj(a.T) + 256*mx.eye(256), stream=d))
tryop("conj", lambda d: mx.conj(a, stream=d))
