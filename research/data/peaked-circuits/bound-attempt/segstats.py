"""Classify every inter-CZ single-qubit segment in each section by distance to the nearest
diagonal or anti-diagonal unitary (spectral norm, exact for 2x2)."""
import sys, numpy as np, collections, json
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
def dist_diag(S):
    # nearest diagonal unitary: diag(e^{ia}, e^{ib}) with phases of S's diagonal -> distance computed exactly
    Dg=np.diag(np.exp(1j*np.angle(np.diag(S)+1e-300)))
    Ad=np.array([[0,np.exp(1j*np.angle(S[0,1]+1e-300))],[np.exp(1j*np.angle(S[1,0]+1e-300)),0]])
    return np.linalg.norm(S-Dg,2), np.linalg.norm(S-Ad,2)
for name in ['P11_Hqap_98x1999','P12_Hqap_98x2457']:
    c=Core(D+f'peaked_circuit_{name}.qasm')
    seq=collections.defaultdict(list)
    for k,u in enumerate(c.units):
        for q in u[:2]: seq[q].append(k)
    def sec(k): return [i for i,(lo,hi) in enumerate(c.secs) if lo<=k<=hi][0]
    stats=collections.defaultdict(list)
    for w,ks in seq.items():
        for k1,k2 in zip(ks,ks[1:]):
            u1=c.units[k1]; u2=c.units[k2]
            post=u1[4] if u1[0]==w else u1[5]; pre=u2[2] if u2[0]==w else u2[3]
            S=pre@post; dd,da=dist_diag(S); d=min(dd,da)
            stats[(sec(k1),sec(k2))].append(d)
    print(name, 'sections', c.secs)
    for key in sorted(stats):
        a=np.array(stats[key])
        print(f'  secs {key}: n {len(a):5d} exact(<1e-9) {np.sum(a<1e-9):4d}  near(1e-9..0.05) {np.sum((a>=1e-9)&(a<0.05)):4d} sum_near {a[(a>=1e-9)&(a<0.05)].sum():.3f}  mid(0.05..0.3) {np.sum((a>=0.05)&(a<0.3)):4d}  generic(>=0.3) {np.sum(a>=0.3):4d}  med_near {np.median(a[(a>=1e-9)&(a<0.05)]) if np.any((a>=1e-9)&(a<0.05)) else 0:.4f}')
