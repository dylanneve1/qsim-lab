import sys, numpy as np
sys.argv=['x','/tmp/peaked-gen/portal/P5_granite_summit.qasm']
import gparse as G, centre
from mpou2 import SWAP
from scipy.optimize import linear_sum_assignment
CZ=np.diag([1,1,1,-1]).astype(complex)
n,units,tail=G.parse(sys.argv[1])
depth=[0]*n; ulay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); ulay.append(l); depth[a]=depth[b]=l+1
for cl,ch in [(36,48),(37,47),(36,47),(37,48),(38,46)]:
    ops=[(a,b,CZ@np.kron(Pa,Pb)) for k,(a,b,Pa,Pb) in enumerate(units) if cl<=ulay[k]<ch]
    groups=centre.components(n,[(a,b) for a,b,_ in ops])
    print('band',cl,ch,'groups',sorted(len(g) for g in groups))
    if max(len(g) for g in groups)>8: continue
    tot=[]
    for g in groups:
        if len(g)<2: continue
        U=centre.dense_group(g,[o for o in ops if o[0] in g]); m=len(g)
        S=centre.pauli_scores(U,m); r,c=linear_sum_assignment(-S); pi=dict(zip(r,c))
        T=U.reshape([2]*(2*m)); best=None
        import itertools
        for order in itertools.permutations(range(m)):
            ax=[]
            for i in order: ax+=[pi[i],m+i]
            Tp=np.transpose(T,ax).reshape(4**(m//2),-1)
            s=np.linalg.svd(Tp,compute_uv=False); w=s**2/np.sum(s**2)
            if best is None or w[0]>best[0]: best=(w[0],w[:4])
        tot.append(1-best[0])
    print('   middle-cut non-product weight per group:', np.round(tot,4), 'sum', round(sum(tot),4))
