"""Rigorous telescoping bound for the middle of P11/P12.
M (middle, core-moved gates excluded) = Outer[ H_A , H_B ]  where H_X = inner hull of block X
(union of anchor-pair causal diamonds).  ||M - Pi|| <= e(H_A) + e(H_B) + e(Outer with H->perm),
each e(.) = sum of exact spectral-norm peel costs of a zipper.  Usage:
  run_full.py NAME KMAX TAU EXCL(comma list) [part: A|B|O|all]"""
import sys, time, json, collections, numpy as np
sys.path.insert(0, '/tmp/peaked-bound')
from solve_peaked_v2 import Core
from zipper2 import Zipper
from split_inner import inner_split
D = '/tmp/pk/research/data/peaked-circuits/'
name = sys.argv[1]; kmax = int(sys.argv[2]); tau = float(sys.argv[3])
EXCL = set(int(x) for x in sys.argv[4].split(',')) if sys.argv[4] else set()
part = sys.argv[5] if len(sys.argv) > 5 else 'all'
c = Core(D + f'peaked_circuit_{name}.qasm'); maps = dict(c.maps)
lo_mid, hi_mid = c.secs[1][0], c.secs[-2][1]
hulls = {}
for si, f in c.maps:
    inner, before, after, bad = inner_split(c, si)
    assert not bad
    hulls[si] = sorted(inner - EXCL)
    cov = set(q for k in inner for q in c.units[k][:2])
    print(f'hull sec{si}: {len(inner)} gates, covers {len(cov)} wires', flush=True)
out = {}
def zip_run(gates, cut, label):
    t = time.time()
    z = Zipper(gates, cut, c.n, kmax=kmax, tau=tau, log=lambda s: print(f'[{label}] ' + s, flush=True)).run_paired()
    cs = np.array(z.costs)
    r = dict(total=float(z.cost), peels=len(cs), nonzero=int((cs > 1e-9).sum()), forced=z.forced, maxk=z.maxk,
             small_sum=float(cs[cs < tau + 1e-12].sum()), big_sum=float(cs[cs > tau + 1e-12].sum()),
             top=np.sort(cs)[::-1][:12].round(4).tolist(), secs=round(time.time() - t),
             forced_sum=float(np.sum(z.forced_costs)), forced_n=len(z.forced_costs),
             unforced_sum=float(z.cost - np.sum(z.forced_costs)),
             nf_sum=float(np.sum(z.nfs)), nf_rss=float(np.sqrt(np.sum(np.square(z.nfs)))),
             unforced_rss=float(np.sqrt(max(0.0, np.sum(cs**2) - np.sum(np.array(z.forced_costs)**2)))))
    print(f'[{label}] RESULT {json.dumps(r)}', flush=True)
    np.save(f'costs_{name}_{label}_k{kmax}_t{tau}{"_droptcz" if len(sys.argv)>6 else ""}.npy', cs)
    return z, r
# ---- inner hulls
hullPi = {}
for si, f in c.maps:
    lab = f'H{si}'
    if part not in ('all', 'A' if si == c.maps[0][0] else 'B'): continue
    ks = hulls[si]
    # cut: downset of anchor 'first' members == use the serial index of the median anchor centre per wire? use downset of firsts
    import solve_peaked as v1
    anc = v1.anchors(c.n, c.units); firsts = set()
    lo, hi = c.secs[si]
    for a, b in anc:
        (w, k1, k2, _), (w2, k1b, k2b, _) = a, b
        if lo <= k1 and k2b <= hi: firsts |= {k1, k2}
    kset = set(ks); seq = collections.defaultdict(list)
    for k in ks:
        for q in c.units[k][:2]: seq[q].append(k)
    down = set(); st = [k for k in firsts if k in kset]
    while st:
        x = st.pop()
        if x in down: continue
        down.add(x)
        for q in c.units[x][:2]:
            st += [y for y in seq[q] if y < x]
    order = sorted(down) + sorted(kset - down)
    gates = [(c.units[k][:2], c.M[k]) for k in order]
    z, r = zip_run(gates, len(down), lab)
    hullPi[si] = z.Pi
    r['Pi_equals_f'] = z.Pi == [f[q] for q in range(c.n)]
    r['Pi_mismatch'] = sum(z.Pi[q] != f[q] for q in range(c.n))
    print(f'[{lab}] Pi==f {r["Pi_equals_f"]} mismatches {r["Pi_mismatch"]}; dangling {len(z.dangling)}', flush=True)
    out[lab] = r
# ---- outer
if part in ('all', 'O'):
    inner_all = set().union(*[set(h) for h in hulls.values()])
    mp = list(range(c.n)); gates = []; cut = None
    hull_end = {si: max(hulls[si]) for si in hulls}
    # walk middle gates in serial order; a hull is 'passed' once we reach the first gate that is
    # after it.  Since every wire crosses each hull, use per-section ordering: gates in a block
    # section split into before/after by inner_split.
    befores = {}; afters = {}
    DROP_TCZ = len(sys.argv) > 6 and sys.argv[6] == 'droptcz'
    for si, f in c.maps:
        inner, before, after, bad = inner_split(c, si); befores[si] = before; afters[si] = after
        if DROP_TCZ:   # DIAGNOSTIC ONLY: drop boundary CZ units on transposition pairs (changes the circuit!)
            tp = {frozenset((q, f[q])) for q in range(c.n)}
            dropped = {k for k in before | after if frozenset(c.units[k][:2]) in tp}
            befores[si] = before - dropped; afters[si] = after - dropped
            print(f'DIAGNOSTIC: dropped {len(dropped)} boundary transposition-pair CZ units of sec{si}', flush=True)
    for si in range(1, len(c.secs) - 1):
        lo, hi = c.secs[si]
        if si in maps:
            for k in range(lo, hi + 1):
                if k in befores[si] and k not in EXCL:
                    a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
            mp = [mp[maps[si][q]] for q in range(c.n)]
            if cut is None: cut = len(gates)
            for k in range(lo, hi + 1):
                if k in afters[si] and k not in EXCL:
                    a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
        else:
            for k in range(lo, hi + 1):
                if k in EXCL: continue
                a, b = c.units[k][:2]; gates.append(((mp[a], mp[b]), c.M[k]))
    print(f'outer: {len(gates)} gates, cut {cut}, final map == Core.L: {mp == c.L}', flush=True)
    z, r = zip_run(gates, cut, 'O')
    r['final_Pi_trivial'] = all(z.Pi[q] == q for q in range(c.n)); r['dangling'] = len(z.dangling)
    out['O'] = r
json.dump(out, open(f'result_{name}_{part}_k{kmax}_t{tau}{"_droptcz" if len(sys.argv)>6 else ""}.json', 'w'), indent=1)
print('DONE', json.dumps({k: v['total'] for k, v in out.items()}))
