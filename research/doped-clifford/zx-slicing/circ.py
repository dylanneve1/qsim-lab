"""Circuit loading/truncation + raw-TN and ZX-TN builders for the IBM doped-Clifford circuit."""
import re, math, numpy as np
QASM = '/home/dylan/.talon/workspace/research/doped-clifford/nq70_depth70_checks27_doped.qasm'

def load(D=999, n=70, path=QASM):
    """Gate list truncated to brickwork CZ layers <= D (1q gates kept iff their qubit's true layer count <= D,
    i.e. the 1q layer right after layer D is kept), and to qubits < n (CZs crossing the cut dropped)."""
    gates, czl, tl = [], [0]*70, [0]*70
    for l in open(path):
        l = l.strip()
        m = re.match(r'cz q\[(\d+)\],q\[(\d+)\];', l)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            d = max(tl[a], tl[b]) + 1; tl[a] = tl[b] = d
            if d <= D and a < n and b < n: gates.append(('cz', a, b))
            continue
        m = re.match(r'(h|s|sx|sxdg|sdg|x|z|y) q\[(\d+)\];', l)
        if m:
            q = int(m.group(2))
            if tl[q] <= D and q < n: gates.append((m.group(1), q))
            continue
        m = re.match(r'rz\(pi/4\) q\[(\d+)\];', l)
        if m:
            q = int(m.group(1))
            if tl[q] <= D and q < n: gates.append(('t', q))
            continue
        assert l.startswith(('OPENQASM', 'include', 'qreg')) or not l, l
    return gates

S2 = 1/np.sqrt(2)
MATS = {'h': np.array([[1, 1], [1, -1]])*S2, 's': np.diag([1, 1j]), 'sdg': np.diag([1, -1j]),
        'sx': 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]]), 'sxdg': 0.5*np.array([[1-1j, 1+1j], [1+1j, 1-1j]]),
        't': np.diag([1, np.exp(1j*np.pi/4)]), 'x': np.array([[0, 1], [1, 0]]), 'z': np.diag([1, -1]),
        'y': np.array([[0, -1j], [1j, 0]])}

def to_qasm(gates, n):
    L = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];']
    for g in gates:
        if g[0] == 'cz': L.append(f'cz q[{g[1]}],q[{g[2]}];')
        elif g[0] == 't': L.append(f'rz(pi/4) q[{g[1]}];')   # differs from T by a global phase only
        else: L.append(f'{g[0]} q[{g[1]}];')
    return '\n'.join(L)

def statevector(gates, n):
    psi = np.zeros((2,)*n, complex); psi[(0,)*n] = 1
    for g in gates:
        if g[0] == 'cz':
            a, b = g[1], g[2]; idx = [slice(None)]*n; idx[a] = 1; idx[b] = 1; psi[tuple(idx)] *= -1
        else:
            psi = np.moveaxis(np.tensordot(MATS[g[0]], psi, axes=([1], [g[1]])), 0, g[1])
    return psi  # T as diag(1,e^{i pi/4}); exact amplitudes up to the rz global phase

def raw_tn(gates, n, x, split=False):
    """quimb amplitude TN <x|U|0> (T as diag(1, e^{i pi/4}) so it matches statevector()).
    split=True: CZ decomposed into two tensors joined by a rank-2 bond (the chain-sweep form)."""
    import quimb.tensor as qtn
    c = qtn.Circuit(n, gate_opts=dict(contract='split-gate') if split else None)
    for g in gates:
        if g[0] == 'cz': c.apply_gate('CZ', g[1], g[2])
        else: c.apply_gate_raw(MATS[g[0]], (g[1],))
    return c.amplitude_tn(''.join(map(str, x)))

def zx_graph(gates, n, x, reduce='full'):
    import pyzx as zx
    from fractions import Fraction as F
    c = zx.Circuit(n)
    for g in gates:
        k, q = g[0], g[1]
        if k == 'cz': c.add_gate('CZ', g[1], g[2])
        elif k == 'h': c.add_gate('HAD', q)
        elif k == 's': c.add_gate('ZPhase', q, F(1, 2))
        elif k == 'sdg': c.add_gate('ZPhase', q, F(3, 2))
        elif k == 't': c.add_gate('ZPhase', q, F(1, 4))
        elif k == 'z': c.add_gate('ZPhase', q, F(1))
        elif k == 'x': c.add_gate('XPhase', q, F(1))
        elif k == 'sx': c.add_gate('XPhase', q, F(1, 2))
        elif k == 'sxdg': c.add_gate('XPhase', q, F(3, 2))
        else: raise ValueError(k)
    g = c.to_graph()
    g.apply_state('0'*n); g.apply_effect(''.join(map(str, x)))
    before = (g.num_vertices(), g.num_edges())
    if reduce == 'full': zx.full_reduce(g)
    elif reduce == 'teleport': g = zx.teleport_reduce(g)
    elif reduce == 'clifford': zx.clifford_simp(g)
    elif reduce == 'none': pass
    return g, before

def zx_to_arrays(g):
    """Scalar ZX diagram -> (inputs, arrays, size_dict, scalar). Z spider = hyperindex; phase -> 1-index vector;
    Hadamard edge -> 2x2 H/sqrt2 matrix; simple edge between Z spiders -> merged index. X spiders get H on all legs."""
    import pyzx as zx
    from pyzx.utils import VertexType, EdgeType
    V = list(g.vertices()); par = {v: v for v in V}
    def f(v):
        while par[v] != v: par[v] = par[par[v]]; v = par[v]
        return v
    ty = {v: g.type(v) for v in V}
    for v in V: assert ty[v] in (VertexType.Z, VertexType.X), ty[v]
    # X spider: represent as Z spider whose every leg carries an extra H (H-edge toggles)
    inputs, arrays = [], []
    Hm = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    for e in g.edges():
        u, v = g.edge_st(e); et = g.edge_type(e)
        h = (et == EdgeType.HADAMARD) ^ (ty[u] == VertexType.X) ^ (ty[v] == VertexType.X)
        if u == v: raise ValueError('self loop')
        if not h: par[f(u)] = f(v)
    # need second pass after merges
    for e in g.edges():
        u, v = g.edge_st(e); et = g.edge_type(e)
        h = (et == EdgeType.HADAMARD) ^ (ty[u] == VertexType.X) ^ (ty[v] == VertexType.X)
        if h:
            a, b = f(u), f(v)
            if a == b:  # H self-loop on a spider: contributes diag(1/sqrt2 * (1, -1))... handle as vector
                inputs.append((f'v{a}',)); arrays.append(np.array([1, -1])/np.sqrt(2))
            else:
                inputs.append((f'v{a}', f'v{b}')); arrays.append(Hm)
    ph = {}
    for v in V:
        a = f(v); ph[a] = ph.get(a, 0) + float(g.phase(v))
    # an X spider's phase in the Z picture after H-conjugation is the same phase
    for a, p in ph.items():
        if abs((p % 2)) > 1e-12:
            inputs.append((f'v{a}',)); arrays.append(np.array([1, np.exp(1j*np.pi*p)]))
    used = {i for t in inputs for i in t}
    try: scalar = complex(g.scalar.to_number())
    except OverflowError: scalar = float('nan')  # scalar irrelevant for cost estimates
    for a in ph:  # isolated spiders: sum over index = 1 + e^{i p}
        if f'v{a}' not in used: scalar *= (1 + np.exp(1j*np.pi*ph[a]))
    return inputs, arrays, {i: 2 for i in used}, scalar
