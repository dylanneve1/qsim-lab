"""Per burst epoch and ring edge: merged 2q block unitary (from first to last CZ on that pair within the epoch, incl. interior 1q
on both wires; aborted if another CZ touches either wire in between) -> Makhlin invariants. Compare bursts pairwise."""
import numpy as np, math
from parse import *
from segment import epochs
from weylcls import makhlin
G=load(); E=epochs()
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
CZ=np.diag([1,1,1,-1]).astype(complex)
# blocks: maximal runs of CZ on same pair with only 1q gates on those wires between
open_={}  # wire -> block
blocks=[]
for i,g in enumerate(G):
    if g[0]=='u3':
        q=g[1][0]; b=open_.get(q)
        if b is not None: b['pend'][q]=U3(*g[2])@b['pend'][q]
        continue
    a,c=g[1]; key=tuple(sorted((a,c)))
    b=open_.get(a)
    if b is not None and b is open_.get(c) and b['pair']==key:
        P=np.kron(b['pend'][key[0]],b['pend'][key[1]]); b['U']=CZ@P@b['U']; b['ncz']+=1; b['pend']={key[0]:np.eye(2),key[1]:np.eye(2)}
    else:
        for q in (a,c):
            ob=open_.get(q)
            if ob is not None:
                for w in ob['pair']: open_.pop(w,None)
        b={'pair':key,'U':CZ.copy(),'ncz':1,'pend':{key[0]:np.eye(2),key[1]:np.eye(2)},'ep':E[i]}
        blocks.append(b); open_[a]=b; open_[c]=b
print('blocks',len(blocks))
inv=[(b['ep'],b['pair'],b['ncz'],np.array(makhlin(b['U']))) for b in blocks]
gen=[x for x in inv if x[2]>=2 and abs(x[3][0]-1)>1e-3 and abs(x[3][0])>1e-3]
print('generic blocks',len(gen))
from collections import defaultdict, Counter
by=defaultdict(list)
for x in gen: by[x[1]].append(x)
M=Counter()
for pair,xs in by.items():
    for i in range(len(xs)):
        for j in range(i+1,len(xs)):
            if np.abs(xs[i][3]-xs[j][3]).max()<1e-3: M[(xs[i][0],xs[j][0])]+=1
print('same-pair invariant matches between epochs:',sorted(M.items(),key=lambda kv:-kv[1])[:30])
# twin pairs
M2=Counter()
for pair,xs in by.items():
    a,b=pair
    if ringof[a]!=0 or ringof[b]!=0: continue
    tp=tuple(sorted((twin[a],twin[b])))
    for x in xs:
        for y in by.get(tp,[]):
            if np.abs(x[3]-y[3]).max()<1e-3: M2[(x[0],y[0])]+=1
print('twin matches',sorted(M2.items()))
print('generic per epoch',sorted(Counter(x[0] for x in gen).items()))
