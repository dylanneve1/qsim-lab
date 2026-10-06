"""Parameter-free anchor blocks: sort anchors by mirror centre (ASAP-layer midpoint of the two segments) and
sweep; a new block starts exactly when an anchor contradicts the current block's involution
(w->a with a != f(w), or a wire already paired elsewhere). No thresholds on counts or layer sizes."""
import sys, collections, numpy as np, gparse as G, struct_probe as SP

def anchor_list(n, units, L):
    segs = SP.segments(n, units); anc, _ = SP.anchors(segs, False)
    out = []
    for i, j in anc:
        (w, kp_i, k_i, _), (v, kp_j, k_j, _) = segs[i], segs[j]
        out.append(dict(w=w, v=v, centre=(L[k_i] + L[kp_j]) / 2, k_end=k_i, k_start=kp_j))
    return sorted(out, key=lambda a: (a['centre'], a['w'], a['v']))

def sweep_blocks(n, anchors):
    blocks, cur, f = [], [], {}
    for a in anchors:
        w, v = a['w'], a['v']
        ok = f.get(w, v) == v and f.get(v, w) == w
        if not ok:
            blocks.append((cur, f)); cur, f = [], {}
        cur.append(a); f[w] = v; f[v] = w
    if cur: blocks.append((cur, f))
    return blocks

def complete(n, f):
    """complete a partial involution; unanchored wires: if exactly the fixed-point completion is forced, use it."""
    miss = [w for w in range(n) if w not in f]
    g = dict(f)
    for w in miss: g[w] = w        # unanchored wires default to fixed points (reported)
    return g, miss

if __name__ == "__main__":
    for path in sys.argv[1:]:
        n, units, tail = G.parse(path); L = SP.layers(n, units)
        A = anchor_list(n, units, L)
        B = sweep_blocks(n, A)
        print(f"== {path.split('/')[-1]}: {len(A)} anchors -> {len(B)} blocks")
        for b, f in B:
            cs = [a['centre'] for a in b]
            g, miss = complete(n, f)
            print(f"   block: {len(b)} anchors, centre layers {min(cs)}..{max(cs)} (median {np.median(cs)}), wires covered {len(f)}, unanchored {miss}, "
                  f"involution {all(g[g[w]]==w for w in range(n))}, moved {sum(g[w]!=w for w in range(n))}")
