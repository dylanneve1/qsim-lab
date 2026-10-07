#!/usr/bin/env python3
"""Second diagnostic: stabilizer Renyi-2 entropy M2 (Pauli sampling) and a
restarted greedy Clifford disentangler, with kept weight at the middle cut.
usage: diag2.py [--restarts R] [--samples S] files..."""
import sys, time, numpy as np, stim
import diag

def wht(a):
    a = a.copy(); h = 1; N = len(a)
    while h < N:
        a = a.reshape(-1, 2, h); a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], 1).reshape(-1); h *= 2
    return a

def m2(T, S, rng):
    m = T.ndim; v = T.transpose(list(range(m))[::-1]).reshape(-1).astype(np.complex128)
    N = len(v); p = np.abs(v) ** 2
    F = wht(p); R = wht(F * F).real / N; R = np.clip(R, 0, None); R /= R.sum(); R[0] = max(R[0] - 1 / N, 0); R /= R.sum()
    # sample P != I from Xi_P = <P>^2/2^m; the identity's share (1/N) is added exactly
    acc = []
    idx = np.arange(N)
    for a in rng.choice(N, size=S, p=R):
        f = np.conj(v[idx ^ a]) * v
        G = np.abs(wht(f)) ** 2
        if a == 0: G[0] = 0
        b = rng.choice(N, p=G / G.sum())
        acc.append(G[b])
    acc = np.array(acc); w = 1 - 1 / N
    mu = 1 / N + w * acc.mean(); se = w * acc.std() / np.sqrt(S)
    return -np.log2(mu), se / mu / np.log(2), np.log2(N + 3) - 2

def kept(T, t, js=(0.5, 1, 2, 3)):
    M = T.reshape(2 ** t, -1); s = np.linalg.svd(M, compute_uv=False); p = np.sort(s ** 2)[::-1]; p /= p.sum()
    r = min(t, T.ndim - t)
    return [p[:int(round(2 ** (r - j)))].sum() for j in js]

def local_layer(T, rng):
    m = T.ndim; U1 = [stim.Tableau.random(1).to_unitary_matrix(endian='big') for _ in range(m)]
    for k in range(m):
        T = np.moveaxis(np.tensordot(U1[k], T, axes=([1], [k])), 0, k)
    return T

if __name__ == '__main__':
    args = sys.argv[1:]; R = 4; S = 400
    if '--restarts' in args: i = args.index('--restarts'); R = int(args[i + 1]); del args[i:i + 2]
    if '--samples' in args: i = args.index('--samples'); S = int(args[i + 1]); del args[i:i + 2]
    rng = np.random.default_rng(7)
    for f in args:
        t0 = time.time(); m, T = diag.load(f); mid = m // 2
        M2, M2se, haar = m2(T, S, rng)
        raw = diag.cut_stats(T); kr = kept(T, mid)
        best = None
        for r in range(R):
            T0 = T if r == 0 else local_layer(T, rng)
            Td, nap = diag.disentangle(T0, sweeps=8)
            st = diag.cut_stats(Td)
            if best is None or st[mid - 1, 0] < best[0][mid - 1, 0]: best = (st, Td, r)
        st, Td, r = best; kd = kept(Td, mid)
        print(f"SUMMARY2 {f} m={m} M2={M2:.2f}+-{M2se:.2f} (Haar {haar:.2f}) midS1 raw={raw[mid-1,0]:.3f} dis={st[mid-1,0]:.3f} "
              f"midS2 raw={raw[mid-1,1]:.3f} dis={st[mid-1,1]:.3f} sumS1 raw={raw[:,0].sum():.1f} dis={st[:,0].sum():.1f} best_restart={r} "
              f"kept(j=.5,1,2,3) raw={' '.join(f'{x:.3f}' for x in kr)} dis={' '.join(f'{x:.3f}' for x in kd)} {time.time()-t0:.0f}s", flush=True)
