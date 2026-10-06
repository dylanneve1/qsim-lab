import sys, numpy as np, time
from fractions import Fraction
sys.path.insert(0,'/tmp/peaked-p5p6')
import pyzx as zx
from cstruct import load
def frac(x, tol):
    f = x/np.pi
    k = round(f*2)/2
    if abs(f-k) < tol/np.pi: return Fraction(int(round(k*2)), 2)
    return Fraction(f).limit_denominator(10**12)
def build(fn, tol):
    ops = load(fn); n = max(max(o[1]) for o in ops)+1
    c = zx.Circuit(n)
    for o in ops:
        if o[0]=='cz': c.add_gate('CZ', *o[1])
        else:
            q=o[1][0]; t,p,l=o[2]
            c.add_gate('ZPhase', q, frac(l-np.pi/2,tol)); c.add_gate('XPhase', q, frac(t,tol)); c.add_gate('ZPhase', q, frac(p+np.pi/2,tol))
    return n, c
if __name__=='__main__':
    fn, tol = sys.argv[1], float(sys.argv[2])
    n, c = build(fn, tol)
    print('circuit stats', c.stats(), flush=True)
    g = c.to_graph(); g.apply_state('0'*n)
    t=time.time(); zx.simplify.full_reduce(g); print('full_reduce', round(time.time()-t,1),'s', flush=True)
    nonc = sum(1 for v in g.vertices() if g.phase(v).denominator > 2 if hasattr(g.phase(v),'denominator'))
    print('vertices', g.num_vertices(), 'edges', g.num_edges(), 'non-Clifford phases', nonc, flush=True)
