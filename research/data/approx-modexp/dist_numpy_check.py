"""Independent numpy check of the Rust `distribution` (incl. the 2-D
Ekerå–Håstad path): the ideal masked distribution P(j, k) for
F = floor(f(a, b) / 2^t) mod T, f = g^a y^-b mod N, mask width W, computed
with numpy's FFT per output value V, against the `ideal_masked` column the
Rust driver writes (`dist ... dist_out=`).

python3 dist_numpy_check.py N=899 g=2 mode=eh f=8 mask=3 [w1=.. ...]
"""
from __future__ import annotations

import sys

import numpy as np

import gidney_env as ge


def main(argv: list[str]) -> int:
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    n_mod, g = int(kv["N"]), int(kv["g"])
    f, mask = int(kv["f"]), int(kv["mask"])
    n = n_mod.bit_length()
    t = max(0, n - f)
    T = n_mod >> t
    W = 1 << mask
    if kv.get("mode") == "eh":
        m = (n + 1) // 2
        s = int(kv.get("s", "1"))
        l = -(-m // s)
        ma, mb = m + l, l
        y = pow(g, (n_mod - 1) // 2, n_mod)
        yi = pow(y, -1, n_mod)
        a = np.arange(1 << ma)
        b = np.arange(1 << mb)
        ga = np.array([pow(g, int(x), n_mod) for x in a], dtype=object)
        yb = np.array([pow(yi, int(x), n_mod) for x in b], dtype=object)
        fv = np.array([[int(ga[i]) * int(yb[k]) % n_mod for i in range(1 << ma)] for k in range(1 << mb)])
    else:
        ma, mb = int(kv.get("m", 2 * n)), 0
        fv = np.array([[pow(g, i, n_mod) for i in range(1 << ma)]])
    F = (fv >> t) % T  # shape (2^mb, 2^ma): row b, column a
    P = np.zeros_like(F, dtype=float)
    for V in np.unique((F[..., None] + np.arange(W)) % T):
        ind = (((V - F) % T) < W).astype(float)
        P += np.abs(np.fft.fft2(ind)) ** 2
    P /= float(1 << (ma + mb)) ** 2 * W
    out = ge.HERE / "xcheck" / "dist_numpy_check.csv"
    out.parent.mkdir(exist_ok=True)
    rust_args = [f"{k}={v}" for k, v in kv.items()]
    ge.run_driver("dist", *rust_args, f"dist_out={out}", "unmasked=0")
    R = np.zeros((1 << mb) * (1 << ma))
    for line in out.read_text().splitlines()[1:]:
        j, _, ideal = line.split(",")
        R[int(j)] = float(ideal)
    R = R.reshape((1 << mb, 1 << ma))
    err = float(np.max(np.abs(R - P)))
    print(f"N={n_mod} mode={kv.get('mode', 'shor')} ma={ma} mb={mb} T={T} W={W}: max|P_rust - P_numpy| = {err:.3e} (sum {P.sum():.12f})")
    return 0 if err < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
