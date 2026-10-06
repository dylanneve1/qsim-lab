#!/usr/bin/env python3
"""Anchor route v3 (for circuits whose identity blocks keep exact inverse segments, e.g. P11/P12).

Changes against solve_peaked_v2.py (all structural thresholds removed):
  * front end: gparse (any QASM 2/3, any 1q gates, cz/cx/rzz...) instead of the strict 5-line parser;
  * blocks: anchors sorted by mirror centre, a new block starts exactly when an anchor contradicts the current
    block's involution (blocks.sweep_blocks) -- replaces "centre section" grouping, ">= 50 anchors", "two blocks";
  * generation sections: serial-layer size jumps split by Otsu's two-class rule (sections2.py) -- replaces the
    9/3 layer rule. Only R = first section and P = last section are used;
  * single-qubit gates are attached to the nearest 2q gate in file order (gparse.parse_nearest), which for
    P11/P12 reproduces the strict 5-line units exactly; the core is R ▷ π ▷ P built from those units;
  * boundary selection: v2's max-peak greedy over the full DAG frontier on both sides (no section restriction).
"""
import argparse, collections, json, resource, time
import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
from cotengra.oe import PathOptimizer
import gparse as G
import struct_probe as SP
import blocks as BL
import sections2 as S2

CZ = np.diag([1, 1, 1, -1]).astype(complex)


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


class Core:
    def __init__(self, path, log=print):
        n, units, tail = G.parse(path)          # pre-folded units: fingerprints / anchors / sections
        self.n, self.units, self.tail = n, units, tail
        n6, self.u6, self.lone = G.parse_nearest(path)   # same CZs, 1q gates attached to nearest 2q gate in file
        self.Lay = SP.layers(n, units)
        anchors = BL.anchor_list(n, units, self.Lay)
        blocks = BL.sweep_blocks(n, anchors)
        self.maps = []
        for b, f in blocks:
            g, miss = BL.complete(n, f)
            if any(g[g[w]] != w for w in range(n)):
                raise ValueError("block map is not an involution")
            self.maps.append(dict(anchors=len(b), centre=float(np.median([a['centre'] for a in b])), unanchored=miss, f=g))
        L = list(range(n))
        for m in self.maps:
            L = [L[m['f'][q]] for q in range(n)]
        self.L = L
        self.secs, self.otsu = S2.sections(units)
        self.r0 = set(range(self.secs[0][0], self.secs[0][1] + 1))
        self.p0 = set(range(self.secs[-1][0], self.secs[-1][1] + 1))
        self.seq = collections.defaultdict(list)
        for k, u in enumerate(units):
            for q in u[:2]:
                self.seq[q].append(k)
        self.M = [G.unit6_matrix(u) for u in self.u6]
        self.opt = CachedGreedy()
        self.marg = {}
        self.Z = np.diag([1., -1.])
        log(f"  structure: {len(units)} units, {len(anchors)} anchors -> {len(self.maps)} blocks "
            f"{[(m['anchors'], m['centre'], len(m['unanchored'])) for m in self.maps]}; {len(self.secs)} sections "
            f"(Otsu jump x{np.exp(self.otsu):.1f}); R={len(self.r0)} P={len(self.p0)} units")

    def ops(self, er, ep):
        """ordered op list [(matrix, wires, key)] for the core R ▷ π ▷ P with sets er/ep of extra units.
        Units carry the single-qubit gates attached to them in the file (pre and post)."""
        n, units, L = self.n, self.units, self.L
        R = sorted(self.r0 | er); P = sorted(self.p0 | ep)
        out = [(self.lone[w], (w,), ('Z', w)) for w in range(n) if not np.allclose(self.lone[w], np.eye(2))]
        out += [(self.M[k], tuple(units[k][:2]), ('R', k)) for k in R]
        out += [(self.M[k], tuple(L[q] for q in units[k][:2]), ('P', k)) for k in P]
        return out

    def evaluate(self, er=frozenset(), ep=frozenset()):
        er, ep = frozenset(er), frozenset(ep)
        ops = self.ops(er, ep)
        circ = qtn.Circuit(self.n)
        for Mx, wires, _ in ops:
            circ.apply_gate_raw(Mx, wires)
        zs = np.empty(self.n)
        for q in range(self.n):
            cone, gates = {q}, []
            for Mx, wires, key in reversed(ops):
                if any(s in cone for s in wires):
                    cone.update(wires); gates.append(key)
            sig = (q, tuple(gates))
            if sig not in self.marg:
                self.marg[sig] = float(np.real(circ.local_expectation(self.Z, (q,), optimize=self.opt)))
            zs[q] = self.marg[sig]
        bits = ''.join('0' if z > 0 else '1' for z in zs)
        p = float(abs(circ.amplitude(bits, optimize=self.opt)) ** 2)
        return dict(p=p, bits=bits, er=er, ep=ep, zs=zs)

    def frontier(self, er, ep):
        """DAG frontier: first non-core unit after R on each wire whose partner wire agrees; same before P."""
        R = self.r0 | er; P = self.p0 | ep
        out = set()
        for w in range(self.n):
            nx = [k for k in self.seq[w] if k not in R]
            if nx and nx[0] not in P:
                k = nx[0]; a, b = self.units[k][:2]; o = b if a == w else a
                if [kk for kk in self.seq[o] if kk not in R][0] == k:
                    out.add(('R', k))
            pv = [k for k in self.seq[w] if k not in P]
            if pv and pv[-1] not in R:
                k = pv[-1]; a, b = self.units[k][:2]; o = b if a == w else a
                if [kk for kk in self.seq[o] if kk not in P][-1] == k:
                    out.add(('P', k))
        return sorted(out)

    def peak(self, ev):
        return ''.join(ev['bits'][self.L[w]] for w in range(self.n))


def solve(path, max_steps=8, log=print):
    t0 = time.time()
    core = Core(path, log)
    cur = core.evaluate()
    log(f"  initial core: p {cur['p']:.4f}  ({time.time() - t0:.1f}s)")
    trace = [dict(step=0, p=cur['p'])]
    for step in range(1, max_steps + 1):
        fr = core.frontier(cur['er'], cur['ep'])
        trials = []
        for s, k in fr:
            ev = core.evaluate(cur['er'] | {k} if s == 'R' else cur['er'], cur['ep'] | {k} if s == 'P' else cur['ep'])
            trials.append((ev, s, k))
        trials.sort(key=lambda x: (-x[0]['p'], x[1], x[2]))
        top = [(round(t['p'], 6), s, k) for t, s, k in trials[:3]]
        trace.append(dict(step=step, frontier=len(fr), cur_p=cur['p'], top=top))
        log(f"  step {step}: frontier {len(fr)}, p {cur['p']:.4f} -> top {top}  ({time.time() - t0:.0f}s)")
        if not trials or trials[0][0]['p'] <= cur['p']:
            break
        cur = trials[0][0]
    return dict(file=path.split('/')[-1], peak=core.peak(cur), p=cur['p'], extra_r=sorted(cur['er']),
                extra_p=sorted(cur['ep']), blocks=[(m['anchors'], m['centre'], m['unanchored']) for m in core.maps],
                sections=core.secs, seconds=round(time.time() - t0, 1),
                rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, trace=trace,
                maps=[[m['f'][q] for q in range(core.n)] for m in core.maps])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("--max-steps", type=int, default=8); ap.add_argument("--json")
    a = ap.parse_args()
    print(a.qasm.split('/')[-1], flush=True)
    r = solve(a.qasm, a.max_steps, log=lambda s: print(s, flush=True))
    print(f"  PEAK (qubit 0 leftmost): {r['peak']}\n  core peak probability {r['p']:.4f}; added R{r['extra_r']} "
          f"P{r['extra_p']}; {r['seconds']}s; peak RSS {r['rss_mb']} MB", flush=True)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
