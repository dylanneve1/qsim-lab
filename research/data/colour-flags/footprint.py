#!/usr/bin/env python3
"""Equal-footprint comparison: K-F's per-round p_L interpolated (log-linear in total qubits) between
adjacent K-F distances, at the qubit count of each flagged (HF) circuit.
usage: footprint.py <prefix> <p> <jsonl...>"""
import json, sys, math, collections
prefix, p = sys.argv[1], float(sys.argv[2])
Q_KF = {5: 28, 7: 55, 9: 91, 11: 136}
Q_HF = {5: 37, 7: 70, 9: 112}
agg, seen, R = collections.defaultdict(lambda: [0, 0]), set(), {}
for f in sys.argv[3:]:
    for l in open(f):
        j = json.loads(l)
        if not j["tag"].startswith(prefix) or abs(j["p"] - p) > 1e-12 or (j["tag"], j["seed"]) in seen:
            continue
        seen.add((j["tag"], j["seed"]))
        a = agg[j["tag"]]; a[0] += j["fails"]; a[1] += j["shots"]; R[j["tag"]] = j["rounds"]
pr = lambda P, r: (1 - max(0.0, 1 - 2 * P) ** (1 / r)) / 2
def rate(d, arm):
    t = f"{prefix}d{d}_p{p}_{arm}"
    if t not in agg or agg[t][0] == 0:
        return None
    f, n = agg[t]
    return pr(f / n, R[t]), f
print(f"p = {p*100:.1f}%  ({prefix})")
for d, q in Q_HF.items():
    h = rate(d, "hf")
    lo = max((dd for dd in Q_KF if Q_KF[dd] <= q), default=None)
    hi = min((dd for dd in Q_KF if Q_KF[dd] >= q), default=None)
    a, b = rate(lo, "kf") if lo else None, rate(hi, "kf") if hi else None
    if not (h and a and b):
        print(f"  HF d={d} ({q} q): missing data (HF {h}, KF d={lo} {a}, KF d={hi} {b})"); continue
    x = (q - Q_KF[lo]) / (Q_KF[hi] - Q_KF[lo])
    kf_q = math.exp((1 - x) * math.log(a[0]) + x * math.log(b[0]))
    sd = math.sqrt(1 / h[1] + ((1 - x) ** 2) / a[1] + (x ** 2) / b[1])
    r = h[0] / kf_q
    print(f"  HF d={d} ({q} qubits): {h[0]:.2e}; K-F at {q} qubits (d={lo}: {a[0]:.2e}, d={hi}: {b[0]:.2e}) "
          f"-> {kf_q:.2e}; ratio {r:.2f} [{r*math.exp(-1.96*sd):.2f}, {r*math.exp(1.96*sd):.2f}]; "
          f"vs K-F d={hi} ({Q_KF[hi]} q): {h[0]/b[0]:.2f}")
