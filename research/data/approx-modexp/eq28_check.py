"""Gidney 2025, Eq. 28 states 1 - |<psi1|psi1~>|^2 <= eps/S. For a uniform
shift of every conditioned mask window by d = eps*N (window width W = S*N)
the overlap is exactly 1 - d/W, so the infidelity is 2 eps/S - (eps/S)^2,
up to twice the stated bound. Dense numpy check (no structure assumed)."""
import numpy as np

T, W, M = 1000, 100, 64
rng = np.random.default_rng(1)
F = rng.integers(0, T, M)
for d in (1, 5, 10, 25):
    psi = np.zeros((M, T))
    phi = np.zeros((M, T))
    for e in range(M):
        psi[e, (F[e] + np.arange(W)) % T] = 1
        phi[e, (F[e] + d + np.arange(W)) % T] = 1
    psi /= np.linalg.norm(psi)
    phi /= np.linalg.norm(phi)
    ov = abs(np.vdot(psi, phi))
    r = d / W
    print(f"shift d={d}: 1-|<psi|phi>|^2 = {1 - ov**2:.4f}  Eq.28 bound eps/S = {r:.4f}  2eps/S-(eps/S)^2 = {2 * r - r * r:.4f}")
