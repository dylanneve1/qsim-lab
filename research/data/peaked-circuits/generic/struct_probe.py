"""Structure probe: anchors (exact inverse segment pairs) under plain and S-quotient fingerprints."""
import sys, collections, numpy as np, gparse as G
from itertools import product

def quat(V):
    V = V / np.sqrt(np.linalg.det(V))
    q = np.array([V[0, 0].real, V[0, 0].imag, V[0, 1].real, V[0, 1].imag])
    for x in q:
        if abs(x) > 1e-9:
            return -q if x < 0 else q
    return q
key = lambda V: tuple(np.round(quat(V), 6) + 0.0)
Sp = [np.diag([1, 1j ** k]) for k in range(4)]

def segments(n, units):
    last, segs = {}, []
    for k, (a, b, Pa, Pb) in enumerate(units):
        for q, P in ((a, Pa), (b, Pb)):
            if q in last: segs.append((q, last[q], k, P))
            last[q] = k
    return segs

def layers(n, units):
    t = [0] * n; L = []
    for a, b, *_ in units:
        l = max(t[a], t[b]); L.append(l); t[a] = t[b] = l + 1
    return L

def anchors(segs, quotient):
    if quotient:
        kf = lambda V: min(key(A @ V @ B) for A, B in product(Sp, Sp))
    else:
        kf = key
    K = [kf(s[3]) for s in segs]; Ki = [kf(np.linalg.inv(s[3])) for s in segs]
    cnt = collections.Counter(K)
    idx = {k: i for i, k in enumerate(K) if cnt[k] == 1}
    out = []
    for i in range(len(segs)):
        if cnt[K[i]] != 1: continue
        j = idx.get(Ki[i])
        if j is not None and j != i and segs[i][2] <= segs[j][1]:
            out.append((i, j))
    return out, cnt

def _main():
  for f in sys.argv[1:]:
      n, units, tail = G.parse(f)
      segs = segments(n, units); L = layers(n, units)
      print(f"== {f.split('/')[-1]}: n={n} units={len(units)} depth={max(L)+1} segments={len(segs)}")
      for qt in (False, True):
          anc, cnt = anchors(segs, qt)
          dup = sum(v for v in cnt.values() if v > 1)
          cen = [(segs[i][2] + segs[j][1]) / 2 for i, j in anc]
          cenL = [(L[segs[i][2]] + L[segs[j][1]]) / 2 for i, j in anc]
          print(f"  quotient={qt}: duplicated-fingerprint segments {dup}, anchors {len(anc)}")
          if anc:
              h = np.histogram(cenL, bins=20, range=(0, max(L)+1))[0]
              print("   centre(layer) histogram 20 bins:", h.tolist())
              same = sum(segs[i][0] == segs[j][0] for i, j in anc)
              print(f"   same-wire anchors {same}; centre layer range {min(cenL)}..{max(cenL)}")

if __name__=='__main__':
    _main()
