"""Like scan_mpou.py but on a units file [(a,b,M)] and integer centre indices (index order growth)."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpou import MPOU
U = np.load(sys.argv[1], allow_pickle=True); lim = int(sys.argv[2]); T = float(sys.argv[3]); cs = [int(x) for x in sys.argv[4:]]
U = [(int(a), int(b), np.asarray(M, dtype=complex)) for a, b, M in U]
n = max(max(a, b) for a, b, _ in U) + 1
for c in cs:
    Lq = list(range(c - 1, -1, -1)); Rq = list(range(c, len(U)))
    W = MPOU(n, cutoff=1e-3, max_bond=4 * lim); t0 = time.time(); step = 0; flip = 0; hist = []
    while (Lq or Rq) and time.time() - t0 < T:
        side = 'dn' if (flip and Lq) or not Rq else 'up'; flip ^= 1
        k = (Lq if side == 'dn' else Rq).pop(0); W.absorb(*U[k], side); step += 1
        if W.stats()['max_bond'] > 4:
            for _ in range(3):
                if W.unswap_sweep() == 0: break
        mb = W.stats()['max_bond']; hist.append(mb)
        if mb > lim: break
    print(f'c={c} absorbed={step} bonds(first 40)={hist[:40]} final {W.stats()} t={time.time()-t0:.0f}s', flush=True)
