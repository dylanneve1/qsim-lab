"""Validate tno_fast: (1) the original test_tno.py suite with the fast gate/unswap installed;
(2) original vs fast on random 6-qubit circuits: identical rank sequences, dense operators agree."""
import numpy as np, copy, runpy
import tno as TN
orig_gate, orig_unswap = TN.TNO.gate, TN.unswap_pass
import tno_fast
tno_fast.install()
print("=== test_tno.py with fast gate/unswap ===")
runpy.run_path('test_tno.py')
fast_gate, fast_unswap = TN.TNO.gate, TN.unswap_pass
def haar(d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))
def dense_from_tno(T, n):
    tids = list(T.T); A, la = T.T[tids[0]]
    for t in tids[1:]:
        A, la = T._contract(A, la, *T.T[t])
    order = [la.index(('o', q)) for q in range(n)] + [la.index(('i', q)) for q in range(n)]
    return np.transpose(A, order).reshape(2 ** n, 2 ** n)
rng = np.random.default_rng(11)
worst = 0.0; same = True
for trial in range(20):
    n = 4; gates = [(haar(4), *map(int, rng.choice(n, 2, replace=False))) for _ in range(10)]
    res = []
    for g_, u_ in ((orig_gate, orig_unswap), (fast_gate, fast_unswap)):
        TN.TNO.gate = g_
        T = TN.TNO(n, cutoff=1e-10); ranks = []
        c = 5; lo, hi = c, c
        while lo > 0 or hi < len(gates):
            if hi < len(gates): Gm, a, b = gates[hi]; ranks.append(T.gate(Gm, a, b, 'after')); hi += 1
            if lo > 0: Gm, a, b = gates[lo - 1]; ranks.append(T.gate(Gm, a, b, 'before')); lo -= 1
            if hi % 3 == 0: ranks.append(('u', u_(T)))
        res.append((ranks, dense_from_tno(T, n)))
    same &= res[0][0] == res[1][0]
    err = np.linalg.norm(res[0][1] - res[1][1]) / np.linalg.norm(res[0][1]); worst = max(worst, err)
print(f"random 4q/10 gates x20 (with unswap passes): identical rank/accept sequences: {same}; max rel diff of dense operators {worst:.2e}")
