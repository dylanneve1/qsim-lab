#!/usr/bin/env python3
"""Threshold-free P11 / P12 solver ("v2", max-peak boundary selection).

solve_peaked.py (v1) is kept unchanged as the historical reference: it guesses where the
reduced core ends with a weak-wire filter (|<Z>| < 0.6) and a 1.3x acceptance rule, and its
repair step is hash-seed dependent.  v2 keeps v1's structure inference (anchors, sections,
involutive wire maps) and replaces only the boundary choice:

    at every step evaluate EVERY frontier gate on both boundaries, add the one that gives the
    highest exact core peak probability, stop when nothing strictly improves.

No thresholds, a fixed total order on ties, no target string.  Justification: R |> P was
trained to maximise peak weight, so the correct core is the one that best reconstructs it.

The cached evaluator (path cache, per-wire marginal cache keyed by backward causal cone) and
the strict involution check follow Monit Sharma's extension of v1
(MonitSharma/peaked_circuits, classical_simulations/p11_p12, MIT).

Usage: solve_peaked_v2.py FILE.qasm [--max-steps N] [--start side:k ...] [--json OUT]
  --start forces the first accepted gate(s) (used for the branch-comparison audit).
"""
import argparse, collections, json, sys, time
import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
from cotengra.oe import PathOptimizer
import solve_peaked as v1


class CachedGreedy(PathOptimizer):
    def __init__(self):
        self.paths = {}

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        lab = {}
        f = lambda i: lab.setdefault(i, len(lab))
        key = (tuple(tuple(f(i) for i in t) for t in inputs), tuple(f(i) for i in output),
               tuple(int(size_dict[i]) for i in lab))
        if key not in self.paths:
            self.paths[key] = ctg.array_contract_path(inputs, output, size_dict, optimize='greedy', cache=False)
        return self.paths[key]


def infer_maps(n, anchors, secs):
    grouped = collections.defaultdict(list)
    for a, b in anchors:
        c = (a[2] + b[1]) / 2
        for si, (lo, hi) in enumerate(secs):
            if lo <= c <= hi:
                grouped[si].append((a[0], b[0]))
    out = []
    for si, pairs in sorted(grouped.items()):
        if len(pairs) < 50:
            continue
        m = {}
        for s, d in pairs:
            for a, b in ((s, d), (d, s)):
                if m.get(a, b) != b:
                    raise ValueError(f"conflicting anchors in section {si} for wire {a}")
                m[a] = b
        miss = sorted(set(range(n)) - m.keys())
        if len(miss) == 1:
            m[miss[0]] = miss[0]
        elif miss:
            raise ValueError(f"ambiguous unanchored wires in section {si}: {miss}")
        if set(m.values()) != set(range(n)) or any(m[m[q]] != q for q in range(n)):
            raise ValueError(f"wire map in section {si} is not an involution")
        out.append((si, m))
    if not out:
        raise ValueError("no usable anchor blocks")
    return out


class Core:
    def __init__(self, path):
        self.n, self.units = v1.parse(path)
        n, units = self.n, self.units
        self.M = [v1.unit_matrix(u) for u in units]
        self.secs = v1.sections(units)
        anc = v1.anchors(n, units)
        self.maps = infer_maps(n, anc, self.secs)
        L = list(range(n))
        for _, mp in self.maps:
            L = [L[mp[q]] for q in range(n)]
        self.L = L
        self.wires = {}
        for k, u in enumerate(units):
            self.wires['R', k] = tuple(u[:2])
            self.wires['P', k] = tuple(L[q] for q in u[:2])
        self.r0 = list(range(self.secs[0][0], self.secs[0][1] + 1))
        self.p0 = list(range(self.secs[-1][0], self.secs[-1][1] + 1))
        self.seq = collections.defaultdict(list)
        for k, u in enumerate(units):
            for q in u[:2]:
                self.seq[q].append(k)
        self.lo1, self.hi1 = self.secs[1]
        self.loL, self.hiL = self.secs[-2]
        self.opt = CachedGreedy()
        self.marg = {}
        self.Z = np.diag([1., -1.])

    def evaluate(self, er=frozenset(), ep=frozenset()):
        er, ep = frozenset(er), frozenset(ep)
        seq = [('R', k) for k in self.r0 + sorted(er)] + [('P', k) for k in sorted(ep) + self.p0]
        circ = qtn.Circuit(self.n)
        for key in seq:
            circ.apply_gate_raw(self.M[key[1]], self.wires[key])
        zs = np.empty(self.n)
        for q in range(self.n):
            cone, gates = {q}, []
            for key in reversed(seq):
                w = self.wires[key]
                if any(s in cone for s in w):
                    cone.update(w); gates.append(key)
            sig = (q, tuple(gates))
            if sig not in self.marg:
                self.marg[sig] = float(np.real(circ.local_expectation(self.Z, (q,), optimize=self.opt)))
            zs[q] = self.marg[sig]
        bits = ''.join('0' if z > 0 else '1' for z in zs)
        p = float(abs(circ.amplitude(bits, optimize=self.opt)) ** 2)
        return dict(p=p, bits=bits, er=er, ep=ep)

    def peak(self, ev):
        return ''.join(ev['bits'][self.L[w]] for w in range(self.n))

    def frontier(self, er, ep):
        out = set()
        for w in range(self.n):
            for k in self.seq[w]:
                if self.lo1 <= k <= self.hi1 and k not in er:
                    a, b = self.units[k][:2]; o = b if a == w else a
                    if all(kk in er for kk in self.seq[o] if self.lo1 <= kk < k):
                        out.add(('R', k))
                    break
            for k in reversed(self.seq[w]):
                if self.loL <= k <= self.hiL and k not in ep:
                    a, b = self.units[k][:2]; o = b if a == w else a
                    if all(kk in ep for kk in self.seq[o] if k < kk <= self.hiL):
                        out.add(('P', k))
                    break
        return sorted(out)


def trial(core, cur, side, k):
    return core.evaluate(cur['er'] | {k} if side == 'R' else cur['er'], cur['ep'] | {k} if side == 'P' else cur['ep'])


def solve(path, max_steps=8, start=(), verbose=True):
    t0 = time.time()
    core = Core(path)
    cur = core.evaluate()
    trace = [dict(step=0, p=cur['p'])]
    forced = list(start)
    for step in range(1, max_steps + 1):
        fr = core.frontier(cur['er'], cur['ep'])
        if forced:
            side, k = forced.pop(0)
            nxt = trial(core, cur, side, k)
            trace.append(dict(step=step, forced=[side, k], p=nxt['p']))
            cur = nxt
            continue
        trials = sorted(((trial(core, cur, s, k), s, k) for s, k in fr), key=lambda x: (-x[0]['p'], x[1], x[2]))
        top = [(round(t['p'], 6), s, k) for t, s, k in trials[:4]]
        trace.append(dict(step=step, frontier=len(fr), cur_p=cur['p'], top=top))
        if verbose:
            print(f"  step {step}: frontier {len(fr)}, p {cur['p']:.4f} -> top {top[:3]}", flush=True)
        if not trials or trials[0][0]['p'] <= cur['p']:
            break
        cur = trials[0][0]
    res = dict(peak=core.peak(cur), p=cur['p'], extra_r=sorted(cur['er']), extra_p=sorted(cur['ep']),
               seconds=round(time.time() - t0, 1), trace=trace)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--start", nargs="*", default=[]); ap.add_argument("--json")
    a = ap.parse_args()
    st = [(s.split(":")[0], int(s.split(":")[1])) for s in a.start]
    r = solve(a.qasm, a.max_steps, st)
    print(f"  PEAK (qubit 0 leftmost): {r['peak']}\n  core peak probability {r['p']:.4f}; "
          f"added R{r['extra_r']} P{r['extra_p']}; {r['seconds']}s", flush=True)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
