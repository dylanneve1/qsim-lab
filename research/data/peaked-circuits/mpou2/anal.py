import sys, numpy as np, collections
import gparse as G
qasm=sys.argv[1]
n,units,tail=G.parse(qasm)
K=len(units)
depth=[0]*n; lay=[]
for a,b,_,_ in units:
    l=max(depth[a],depth[b]); lay.append(l); depth[a]=depth[b]=l+1
D=max(lay)+1
print('n',n,'units',K,'depth',D)
# blocks: merge consecutive same-pair units (no other gate on a or b in between)
last={}  # wire -> block id
blocks=[]  # list of (a,b,[unit ids])
for k,(a,b,_,_) in enumerate(units):
    ba,bb=last.get(a),last.get(b)
    if ba is not None and ba==bb and set(blocks[ba][:2])=={a,b}:
        blocks[ba][2].append(k)
    else:
        blocks.append((a,b,[k])); last[a]=last[b]=len(blocks)-1
print('blocks',len(blocks), collections.Counter(len(b[2]) for b in blocks))
cnt=collections.Counter(lay)
print('units per layer', [cnt[l] for l in range(D)])
# where are 3-unit blocks (swap candidates)
b3=[b for b in blocks if len(b[2])>=3]
print('layers of >=3-unit blocks', sorted(collections.Counter(lay[b[2][0]] for b in b3).items()))
# file order of band units
lo,hi=int(sys.argv[2]),int(sys.argv[3])
band=[k for k in range(K) if lo<=lay[k]<hi]
print('band units',len(band),'file idx range',min(band),max(band),'quartiles',np.percentile(band,[0,25,50,75,100]))
# components
par=list(range(n))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for k in band: par[f(units[k][0])]=f(units[k][1])
comp=collections.Counter(f(q) for q in range(n)); print('components sizes',sorted(comp.values()))
