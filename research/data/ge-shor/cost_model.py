"""Toffoli cost models (research/shor/ge-shor.md section 5).

GE19: Gidney & Ekera 2019 (arXiv:1905.09749), per lookup-addition 2n + 2^(ce+cm)
Toffolis (their accounting), 2*(ne/ce)*(n/cm) lookup-additions, ne = 1.5n (EH).
Asymptotic headline: 0.3 n^3 + 0.0005 n^3 lg n.

ours(coset): our component costs (measured in the gate-level circuits):
  lookup over w = we+wm address bits, uncontrolled, temporary ANDs: 2^w - 2
  measurement-based unlookup: (2^(w-k)-1) + (2^k - k - 1), min over k
  plain (n+c)-bit Gidney addition: n + c - 1
  per window: 2 multiplies x ceil((n+c)/wm) lookup-additions
  windows: ceil(ne/we) per register (EH: m and 2m bits, m = ceil(n/2))
"""
import math

def ge19(n, ce, cm, ne=None):
    ne = 1.5 * n if ne is None else ne
    return 2 * math.ceil(ne / ce) * math.ceil(n / cm) * (2 * n + 2 ** (ce + cm))

def unlookup(w):
    return min((2 ** (w - k) - 1) + (2 ** k - k - 1) for k in range(0, min(w, 6) + 1))

def ours(n, we, wm, c, eh=True):
    m = math.ceil(n / 2)
    regs = [m, 2 * m] if eh else [2 * n]
    wins = sum(math.ceil(L / we) for L in regs)
    nr = n + c
    nw = math.ceil(nr / wm)
    w = we + wm
    per_mult = nw * (2 ** w - 2) + nw * (nr - 1) + nw * unlookup(w)
    return wins * 2 * per_mult

print("n   GE19-headline  GE19-model(best ce,cm)          ours-model coset c=2lg n+? (best we,wm)")
for n in [20, 24, 28, 31, 64, 128, 256, 512, 1024, 2048]:
    head = 0.3 * n ** 3 + 0.0005 * n ** 3 * math.log2(n)
    best = min((ge19(n, ce, cm), ce, cm) for ce in range(1, 9) for cm in range(1, 9))
    c = int(round(2 * math.log2(n))) + 4
    bo = min((ours(n, we, wm, c), we, wm) for we in range(1, 9) for wm in range(1, 9))
    print(f"{n:5d} {head:14.4g} {best[0]:14.4g} (ce={best[1]},cm={best[2]}) {bo[0]:14.4g} (c={c}, we={bo[1]}, wm={bo[2]})  ours/GE19-model={bo[0]/best[0]:.2f}")
