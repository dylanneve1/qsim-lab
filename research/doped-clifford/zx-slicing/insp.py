import re, numpy as np, collections
from circ import *
D=40; n=70; gates=load(D,n); x=np.random.default_rng(7).integers(0,2,n)
tn=raw_tn(gates,n,x); print('before simplify', tn.num_tensors)
tn.full_simplify_(output_inds=[], atol=1e-12)
ts=list(tn)
nq=collections.Counter(len([g for g in t.tags if re.fullmatch(r'I\d+',g)]) for t in ts); print('qubit-tags per tensor', sorted(nq.items()))
print('ranks', sorted(collections.Counter(t.ndim for t in ts).items()))
print('hyper index degrees', sorted(collections.Counter(len(v) for v in tn.ind_map.values()).items()))
def qs(t): return [int(g[1:]) for g in t.tags if re.fullmatch(r'I\d+',g)]
for key in ('min','max','mean'):
    f={'min':min,'max':max,'mean':lambda l: sum(l)/len(l)}[key]
    side={t_id: f(qs(t)) for t_id,t in tn.tensor_map.items()}
    cut=[]
    for c in [10,20,35,50]:
        b=sum(1 for i,tids in tn.ind_map.items() if any(side[t]<=c+0.5 for t in tids) and any(side[t]>c+0.5 for t in tids))
        cut.append(b)
    print(key,'cut sizes at q=10,20,35,50:',cut)
