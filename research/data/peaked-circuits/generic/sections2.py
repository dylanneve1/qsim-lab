"""Parameter-free generation sections: serial greedy layers; a section boundary is a serial-layer size jump
whose log-ratio lies in the upper Otsu class of all positive log-ratio jumps (Otsu = parameter-free 2-class split)."""
import sys, math, numpy as np
sys.path.insert(0, '/tmp/peaked-generic')

def serial_layers(units):
    layers, cur, used = [], [], set()
    for k, x in enumerate(units):
        if x[0] in used or x[1] in used:
            layers.append(cur); cur, used = [], set()
        cur.append(k); used |= {x[0], x[1]}
    layers.append(cur)
    return layers

def otsu(x):
    x = np.sort(np.asarray(x)); best = (-1, None)
    for i in range(1, len(x)):
        a, b = x[:i], x[i:]
        v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if v > best[0]: best = (v, (x[i - 1] + x[i]) / 2)
    return best[1]

def sections(units):
    layers = serial_layers(units)
    s = np.array([len(l) for l in layers], float)
    r = np.log(s[1:] / s[:-1])
    pos = r[r > 0]
    thr = otsu(pos)
    cuts = [i + 1 for i in range(len(r)) if r[i] > thr]
    secs, start = [], 0
    for c in cuts:
        secs.append((layers[start][0], layers[c - 1][-1])); start = c
    secs.append((layers[start][0], layers[-1][-1]))
    return secs, thr

if __name__ == "__main__":
    import solve_peaked as v1, gparse as G
    for f in sys.argv[1:]:
        n, units, tail = G.parse(f)
        secs, thr = sections(units)
        old = v1.sections(v1.parse(f)[1]) if 'P1' in f else None
        print(f.split('/')[-1], f"otsu log-ratio threshold {thr:.2f} (x{math.exp(thr):.1f})")
        print("   new:", secs); print("   old:", old, " identical:", secs == old)
