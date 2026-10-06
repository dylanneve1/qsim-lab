"""Array backends for the middle-out TNO growth.

  np128  : numpy complex128 (reference, = original code)
  np64   : numpy complex64 on CPU (Accelerate / OpenBLAS)
  mlx    : mlx complex64 arrays on the Metal GPU; contractions (tensordot/einsum via autoray+cotengra) run on
           the GPU; decompositions (QR/SVD/eigh, not available on the MLX GPU in mlx 0.32, complex QR not even on
           the MLX CPU) are routed to numpy/LAPACK on the CPU via registered quimb `compose` overrides.
  mlxcpu : same as mlx but mx.set_default_device(mx.cpu) (separates GPU effect from MLX dispatch overhead)
"""
import numpy as np

NAME = 'np128'
_mx = None


def setup(name):
    global NAME, _mx
    NAME = name
    if name.startswith('mlx'):
        import mlx.core as mx
        _mx = mx
        mx.set_default_device(mx.gpu if name == 'mlx' else mx.cpu)
        _register_mlx()


def dtype():
    return np.complex128 if NAME == 'np128' else np.complex64


def asb(x):
    """numpy array -> backend array"""
    x = np.asarray(x, dtype=dtype())
    if NAME.startswith('mlx'):
        return _mx.array(x)
    return x


def tonp(x):
    """backend array -> numpy (complex128 for downstream exact contraction)"""
    if _mx is not None and isinstance(x, _mx.array):
        return np.array(x).astype(np.complex128)
    return np.asarray(x).astype(np.complex128)


def to_numpy_tn(tn):
    tn = tn.copy()
    for t in tn:
        t.modify(data=tonp(t.data))
    return tn


_registered = [False]


def _register_mlx():
    if _registered[0]:
        return
    _registered[0] = True
    import autoray as ar
    import quimb.tensor.decomp as D
    mx = _mx

    def n2m(*xs):
        return tuple(None if x is None else (mx.array(x) if isinstance(x, np.ndarray) else x) for x in xs)

    def m2n(x):
        return np.array(x)

    # --- quimb decomposition kernels: run numpy/numba versions on CPU copies ---------------------------
    def _wrap(fn_numpy):
        def f(x, *args, **kwargs):
            out = fn_numpy(m2n(x), *args, **kwargs)
            if isinstance(out, tuple):
                return n2m(*out)
            return mx.array(out) if isinstance(out, np.ndarray) else out
        return f

    for nm in ('svd_truncated', 'eigh_truncated', 'qr_stabilized', 'lq_stabilized', 'polar_right', 'polar_left',
               'squared_op_to_reduced_factor', 'lu_truncated'):
        comp = getattr(D, nm, None)
        if comp is None:
            continue
        npfn = _numpy_impl(comp)
        if npfn is not None:
            comp.register('mlx')(_wrap(npfn))

    # --- autoray functions mlx lacks or gets wrong for complex --------------------------------------------
    def qr(x, mode='reduced'):
        q, r = np.linalg.qr(m2n(x))
        return mx.array(q), mx.array(r)

    def svd(x, full_matrices=False, compute_uv=True):
        if not compute_uv:
            return mx.array(np.linalg.svd(m2n(x), compute_uv=False))
        u, s, vh = np.linalg.svd(m2n(x), full_matrices=False)
        return mx.array(u), mx.array(s), mx.array(vh)

    def eigh(x):
        w, v = np.linalg.eigh(m2n(x))
        return mx.array(w), mx.array(v)

    ar.register_function('mlx', 'linalg.qr', qr)
    ar.register_function('mlx', 'linalg.svd', svd)
    ar.register_function('mlx', 'linalg.eigh', eigh)
    ar.register_function('mlx', 'linalg.inv', lambda x: mx.array(np.linalg.inv(m2n(x))))
    ar.register_function('mlx', 'linalg.pinv', lambda x, *a, **k: mx.array(np.linalg.pinv(m2n(x), *a, **k)))
    ar.register_function('mlx', 'linalg.solve', lambda a, b: mx.array(np.linalg.solve(m2n(a), m2n(b))))
    ar.register_function('mlx', 'linalg.norm', lambda x, *a, **k: float(np.linalg.norm(m2n(x), *a, **k)))
    ar.register_function('mlx', 'count_nonzero', lambda x: int(np.count_nonzero(m2n(x))))
    ar.register_function('mlx', 'to_numpy', m2n)


def _numpy_impl(comp):
    """fetch the numpy-registered implementation of a quimb/autoray `compose` function"""
    import autoray as ar
    try:
        return ar.get_lib_fn('numpy', comp._name)
    except Exception:
        return None
