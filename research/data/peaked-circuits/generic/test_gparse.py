"""Unit tests for gparse: parsed unit circuit == QASM circuit (state fidelity 1 up to global phase)."""
import sys, os, glob, random, math, numpy as np
import gparse as G

def fid(a, b): return abs(np.vdot(a, b))

def rand_state(n, rng):
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n); return v / np.linalg.norm(v)

def check(n, ops, rng, tag):
    units, tail = G.to_units(n, ops)
    psi = rand_state(n, rng)
    f = fid(G.ref_state(n, ops, psi), G.unit_state(n, units, tail, psi))
    u6, lone = G.to_units_nearest(n, ops)
    f6 = fid(G.ref_state(n, ops, psi), G.unit6_state(n, u6, lone, psi))
    ok = abs(f - 1) < 1e-10 and abs(f6 - 1) < 1e-10
    print(f"{tag}: n={n} ops={len(ops)} units={len(units)} fidelity={f:.15f} nearest-fold={f6:.15f} {'OK' if ok else 'FAIL'}")
    return ok

rng = np.random.default_rng(1)
allok = True
# 1) synthetic circuits with every supported gate type, written to QASM2 and QASM3 and re-parsed
one = ['u', 'u3', 'u2', 'u1', 'p', 'rz', 'rx', 'ry', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'id']
npar = {'u': 3, 'u3': 3, 'u2': 2, 'u1': 1, 'p': 1, 'rz': 1, 'rx': 1, 'ry': 1}
two = [('cz', 0), ('cx', 0), ('cy', 0), ('swap', 0), ('rzz', 1), ('cp', 1), ('crz', 1)]
for trial in range(6):
    n = 6; lines = []
    for _ in range(120):
        if rng.random() < 0.55:
            g = one[rng.integers(len(one))]; q = rng.integers(n)
            ps = ",".join(repr(float(x)) for x in rng.uniform(-math.pi, math.pi, npar.get(g, 0)))
            lines.append(f"{g}({ps}) q[{q}];" if ps else f"{g} q[{q}];")
        else:
            g, k = two[rng.integers(len(two))]; a, b = rng.choice(n, 2, replace=False)
            if g == 'rzz': ps = rng.choice(['-pi/2', 'pi/2', '3*pi/2', '-3*pi/2'])
            elif g == 'cp': ps = 'pi'
            elif g == 'crz': ps = rng.choice(['pi', '-pi'])
            else: ps = None
            lines.append(f"{g}({ps}) q[{a}],q[{b}];" if ps else f"{g} q[{a}],q[{b}];")
    for ver, hdr in (("2", f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\n'), ("3", f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[{n}] q;\n')):
        path = f"/tmp/peaked-generic/_t{trial}_{ver}.qasm"; open(path, "w").write(hdr + "\n".join(lines) + "\n")
        nn, ops = G.parse_ops(path); allok &= check(nn, ops, rng, f"synthetic#{trial} qasm{ver}"); os.remove(path)
# user gate definitions (iswap macro as in portal P8) against the exact iSWAP matrix
open("/tmp/peaked-generic/_tisw.qasm","w").write('OPENQASM 2.0;\ninclude "qelib1.inc";\ngate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }\nqreg q[2];\niswap q[0],q[1];\n')
nn, ops = G.parse_ops("/tmp/peaked-generic/_tisw.qasm")
ISW = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]])
U = np.column_stack([G.ref_state(2, ops, np.eye(4)[:, c].astype(complex)) for c in range(4)])
ok_isw = abs(abs(np.trace(U.conj().T @ ISW)) / 4 - 1) < 1e-12
print("iswap macro expands to", [o[0] for o in ops], "== iSWAP:", ok_isw); allok &= ok_isw
allok &= check(2, ops, rng, "iswap macro units")
import os; os.remove("/tmp/peaked-generic/_tisw.qasm")
# non-CZ-class must be rejected
try:
    G.canon_cz(G.twoq_matrix('rzz', [0.3])); print("non-CZ rzz accepted: FAIL"); allok = False
except ValueError: print("non-CZ-class rzz(0.3) rejected: OK")
# 2) the real circuits: random 10/12-qubit sub-circuits (all ops supported on the subset, in order)
files = sorted(glob.glob('/tmp/peaked-gen/portal/P8_*.qasm')) + sorted(glob.glob('/tmp/peaked-gen/peaked_circuit_*.qasm')) + sorted(glob.glob('/tmp/pk/research/data/peaked-circuits/peaked_circuit_P1*.qasm'))
for f in files:
    n, ops = G.parse_ops(f)
    for t in range(3):
        # pick a connected-ish subset: start from a random 2q gate and grow by interaction
        sub = set(rng.choice(n, 1).tolist())
        for name, p, qs in ops:
            if len(qs) == 2 and (qs[0] in sub or qs[1] in sub) and len(sub | set(qs)) <= 11:
                sub |= set(qs)
        sub = sorted(sub); idx = {q: i for i, q in enumerate(sub)}
        sops = [(nm, p, tuple(idx[q] for q in qs)) for nm, p, qs in ops if all(q in idx for q in qs)]
        allok &= check(len(sub), sops, rng, f"{f.split('/')[-1]} sub{t}")
    # full-file statistics
    nn, units, tail = G.parse(f)
    print(f"   full parse: n={nn} 2q units={len(units)} (2q ops in file {sum(len(q)==2 for _,_,q in ops)})")
print("ALL OK" if allok else "SOME FAILED")
