"""Width of intermediate cuts through the D=70 network with several worldlines in progress.
Cut = per qubit a done set that is a prefix (forward) or suffix (backward) of its event list."""
from chain import *
import itertools
n0, ops = parse()
n = 70
lines, be = chain(n, window(n0, ops, n, 0, 70))
cz = [[e for e in L if e[0] != 'G'] for L in lines]   # CZ events only (gates irrelevant for width)
def width(cfg):
    # cfg[j] = (dir, m): dir 'f' -> first m CZ events done; 'b' -> last m done
    done = {}
    wires = 0
    for j, (d, m) in enumerate(cfg):
        ev = cz[j]; N = len(ev)
        sel = ev[:m] if d == 'f' else ev[N-m:]
        for e in sel: done[(j, e[1])] = True
        if 0 < m < N:
            last = ev[m-1] if d == 'f' else ev[N-m]
            if last[0] == 'Z': wires += 1
            else: wires += ('tie', last[1], j)  and 0 or 0  # resolved below
    # crossed bonds
    crossed = set()
    for b, e in enumerate(be):
        if done.get((e, b), False) != done.get((e+1, b), False): crossed.add(b)
    w = len(crossed)
    for j, (d, m) in enumerate(cfg):
        ev = cz[j]; N = len(ev)
        if 0 < m < N:
            last = ev[m-1] if d == 'f' else ev[N-m]
            if last[0] == 'Z' or last[1] not in crossed: w += 1
    return w
N = len(cz[10])
import sys; J = int(sys.argv[1]) if len(sys.argv) > 1 else 30
base = lambda: [('f', len(cz[j])) for j in range(J)] + [('f', 0) for j in range(J, n)]
print('vertical cut', width(base()))
for dj in ('f', 'b'):
    best = {}
    for m in range(1, N):
        c = base(); c[J] = (dj, m); best.setdefault(width(c), []).append(m)
    print('one in progress, q%d dir %s:' % (J, dj), {k: len(v) for k, v in best.items()})
for k in (2,):
    hist = {}; hit = []
    step = 1 if k == 2 else 4
    for dirs in itertools.product('fb', repeat=k):
        for ms in itertools.product(range(1, N, step), repeat=k):
            c = base()
            for i in range(k): c[J+i] = (dirs[i], ms[i])
            w = width(c); hist[w] = hist.get(w, 0) + 1
            if w <= 35: hit.append((dirs, ms))
    print("k=%d min width %d" % (k, min(hist)), sorted(hist.items())[:4], "n35:", len(hit), hit[:6], flush=True)
