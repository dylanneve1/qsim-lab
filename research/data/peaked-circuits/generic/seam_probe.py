"""Anchor geometry: per-wire CZ index of each anchor segment; seam sum m_i + m_j per unordered wire pair."""
import sys, collections, numpy as np, gparse as G, struct_probe as SP
for f in sys.argv[1:]:
    n, units, tail = G.parse(f); L = SP.layers(n, units)
    segs = SP.segments(n, units); anc, _ = SP.anchors(segs, False)
    pos = {}   # unit k on wire q -> per-wire CZ index (1-based)
    cnt = [0] * n
    for k, u in enumerate(units):
        for q in u[:2]:
            cnt[q] += 1; pos[k, q] = cnt[q]
    groups = collections.defaultdict(list)
    for i, j in anc:
        (w, kp_i, k_i, _), (v, kp_j, k_j, _) = segs[i], segs[j]
        mi, mj = pos[k_i, w], pos[k_j, v]          # segment index = index of the CZ that closes it
        groups[(min(w, v), max(w, v), mi + mj)].append((w, v, mi, mj, L[k_i], L[kp_j]))
    sizes = collections.Counter(len(g) for g in groups.values())
    print(f"== {f.split('/')[-1]}: anchors {len(anc)}, seam groups {len(groups)}, group-size histogram {sorted(sizes.items())}")
    perpair = collections.Counter((a, b) for a, b, s in groups)
    print("   seam groups per wire pair:", sorted(collections.Counter(perpair.values()).items()))
    big = sorted(groups.items(), key=lambda x: -len(x[1]))[:5]
    for (a, b, s), g in big:
        print(f"   pair ({a},{b}) seam-sum {s}: {len(g)} anchors, layers {sorted(set((x[4], x[5]) for x in g))[:6]}")
