"""Rigorous telescoping bound, decomposition v2:
  M = Outer[ Block_A , Block_B ],  Block_X = (transported boundary CZs) + hull_X.
  eps <= transport(T) + e(hull_A + T_A) + e(hull_B + T_B) + e(Outer without T, blocks -> perms).
T = boundary transposition-pair CZ units whose transport cost < TT (others stay in Outer).
usage: run_full2.py NAME KMAX TAU EXCL PART(A|B|O) [TT=0.05]"""
import sys, time, json, collections, numpy as np
sys.path.insert(0, '/tmp/peaked-bound')
from solve_peaked_v2 import Core
import solve_peaked as v1
from zipper2 import Zipper
from split_inner import inner_split
from transportlib import transport_costs
D = '/tmp/pk/research/data/peaked-circuits/'
name = sys.argv[1]; kmax = int(sys.argv[2]); tau = float(sys.argv[3])
EXCL = set(int(x) for x in sys.argv[4].split(',')) if sys.argv[4] else set()
part = sys.argv[5]; TT = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
maxsteps = int(sys.argv[7]) if len(sys.argv) > 7 else None
c = Core(D + f'peaked_circuit_{name}.qasm'); maps = dict(c.maps)
info = {}
for si, f in c.maps:
    inner, before, after, bad = inner_split(c, si)
    tc = transport_costs(c, si, f, inner, before, after)
    T = {k for k, v in tc.items() if v < TT}
    info[si] = dict(inner=inner, before=before, after=after, T=T, tcost=sum(tc[k] for k in T), tc=tc)
    print(f'sec{si}: hull {len(inner)}, boundary tCZ {len(tc)}, transported {len(T)} cost {info[si]["tcost"]:.4f}, kept-in-outer {len(tc)-len(T)}', flush=True)
def zip_run(gates, cut, label):
    t = time.time()
    z = Zipper(gates, cut, c.n, kmax=kmax, tau=tau, log=lambda s: print(f'[{label}] ' + s, flush=True)).run_paired(maxsteps=maxsteps)
    cs = np.array(z.costs); fc = np.array(z.forced_costs)
    r = dict(total=float(z.cost), peels=len(cs), forced_n=len(fc), forced_sum=float(fc.sum()),
             unforced_sum=float(z.cost - fc.sum()), maxk=z.maxk, remaining_V=len(z.V), remaining_W=len(z.W),
             nf_sum=float(np.sum(z.nfs)), nf_rss=float(np.sqrt(np.sum(np.square(z.nfs)))),
             rss=float(np.sqrt(np.sum(cs ** 2))), top=np.sort(cs)[::-1][:10].round(4).tolist(), secs=round(time.time() - t))
    print(f'[{label}] RESULT {json.dumps(r)}', flush=True)
    np.save(f'costs2_{name}_{label}_k{kmax}_t{tau}.npy', cs); np.save(f'nfs2_{name}_{label}_k{kmax}_t{tau}.npy', np.array(z.nfs))
    return z, r
out = {}
blocks = [si for si, _ in c.maps]
if part in ('A', 'B'):
    si = blocks[0] if part == 'A' else blocks[1]; f = maps[si]; I = info[si]
    lo, hi = c.secs[si]
    anc = v1.anchors(c.n, c.units); firsts = set()
    for a, b in anc:
        (w, k1, k2, _), (w2, k1b, k2b, _) = a, b
        if lo <= k1 and k2b <= hi: firsts |= {k1, k2}
    ks = set(I['inner']) - EXCL
    seq = collections.defaultdict(list)
    for k in sorted(ks):
        for q in c.units[k][:2]: seq[q].append(k)
    down = set(); st = [k for k in firsts if k in ks]
    while st:
        x = st.pop()
        if x in down: continue
        down.add(x)
        for q in c.units[x][:2]: st += [y for y in seq[q] if y < x]
    Tb = sorted(k for k in I['T'] if k in I['before']); Ta = sorted(k for k in I['T'] if k in I['after'])
    order = Tb + sorted(down) + sorted(ks - down) + Ta
    gates = [(c.units[k][:2], c.M[k]) for k in order]
    z, r = zip_run(gates, len(Tb) + len(down), f'H{si}')
    r['transport_cost'] = I['tcost']; r['Pi_equals_f'] = z.Pi == [f[q] for q in range(c.n)]
    r['Pi_mismatch'] = sum(z.Pi[q] != f[q] for q in range(c.n))
    print(f'[H{si}] Pi==f {r["Pi_equals_f"]} mismatch {r["Pi_mismatch"]}', flush=True)
    out[f'H{si}'] = r
if part == 'O':
    mp = list(range(c.n)); gates = []; cut = None
    for si in range(1, len(c.secs) - 1):
        lo, hi = c.secs[si]
        if si in maps:
            I = info[si]
            for k in range(lo, hi + 1):
                if k in I['before'] and k not in I['T'] and k not in EXCL:
                    a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
            mp = [mp[maps[si][q]] for q in range(c.n)]
            if cut is None: cut = len(gates)
            for k in range(lo, hi + 1):
                if k in I['after'] and k not in I['T'] and k not in EXCL:
                    a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
        else:
            for k in range(lo, hi + 1):
                if k in EXCL: continue
                a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
    print(f'outer: {len(gates)} gates, cut {cut}, final map == Core.L: {mp == c.L}', flush=True)
    z, r = zip_run(gates, cut, 'O')
    r['final_Pi_trivial'] = all(z.Pi[q] == q for q in range(c.n))
    out['O'] = r
json.dump(out, open(f'result2_{name}_{part}_k{kmax}_t{tau}.json', 'w'), indent=1)
print('DONE', json.dumps({k: v['total'] for k, v in out.items()}), flush=True)
