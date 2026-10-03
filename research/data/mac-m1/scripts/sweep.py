import sys, collections
D = collections.defaultdict(lambda: collections.defaultdict(list))
for fn in sys.argv[1:]:
    for line in open(fn):
        head, rest = line.split('|', 1); t, w = head.split()
        f = [x.strip() for x in rest.split('|')]
        D[(f[0], f[2], int(f[1]), int(t))][f[3]].append(float(f[4]))
for k in sorted(D):
    m = {c: min(v) for c, v in D[k].items()}
    best = min(m, key=m.get)
    print(f"### {k[0]} {k[1]} n={k[2]} t={k[3]}  (b256 = {m['b256']:.4f}s, best {best} {m['b256']/m[best]:.3f}x)")
    print(' '.join(f"{c}={m[c]:.4f}({m['b256']/m[c]:.3f}x,n{len(D[k][c])})" for c in m))
