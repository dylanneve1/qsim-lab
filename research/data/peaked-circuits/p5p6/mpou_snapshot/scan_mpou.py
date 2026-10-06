"""Centre scan for the MPO+unswap grower: for each centre fraction, grow until max bond > LIM (after unswapping)
or T seconds; score = units absorbed.  usage: scan_mpou.py QASM LIM T f1 f2 ..."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gparse as G
from mpou import MPOU
qasm, lim, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); fr = [float(x) for x in sys.argv[4:]]
CZ = np.diag([1, 1, 1, -1]).astype(complex)
n, units, tail = G.parse(qasm)
U = [(a, b, CZ @ np.kron(Pa, Pb)) for a, b, Pa, Pb in units]
for f in fr:
    c = int(round(f * len(U))); Lq = list(range(c - 1, -1, -1)); Rq = list(range(c, len(U)))
    W = MPOU(n, cutoff=1e-3, max_bond=4 * lim); t0 = time.time(); step = 0; flip = 0; peak = 0
    while (Lq or Rq) and time.time() - t0 < T:
        if Lq and Rq:
            a, b = U[Lq[0]][:2]; dl = abs(W.pd[a] - W.pd[b]); a, b = U[Rq[0]][:2]; dr = abs(W.pu[a] - W.pu[b])
            side = 'dn' if (dl < dr or (dl == dr and flip)) else 'up'; flip ^= 1
        else: side = 'dn' if Lq else 'up'
        k = (Lq if side == 'dn' else Rq).pop(0); W.absorb(*U[k], side); step += 1
        if W.stats()['max_bond'] > 8:
            for _ in range(3):
                if W.unswap_sweep() == 0: break
        mb = W.stats()['max_bond']; peak = max(peak, mb)
        if mb > lim: break
    print(f'f={f:.3f} centre={c} absorbed={step} (left {c-len(Lq)}, right {len(U)-c-len(Rq)}) final {W.stats()} t={time.time()-t0:.0f}s', flush=True)
