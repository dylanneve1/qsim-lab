"""For consecutive matched U-blocks on a wire (layers l, l') take the 1q segment between them (s) and the segment
between their V-partners on wire pi(a) (s'); report distance of s'.s from identity (up to phase)."""
import numpy as np, sys
sys.path.insert(0,'.')
from blocks2 import blocks_v2, u3m
from cstruct import load
fn=sys.argv[1]
ops=load(fn); n=max(max(o[1]) for o in ops)+1
X=np.load(fn.split('/')[-1]+'.bw.npy'); M=np.load(fn.split('/')[-1]+'.bw.npy.match.npy'); pi=np.load(fn.split('/')[-1]+'.bw.npy.pi.npy'); L=X[:,2].astype(int)
# per wire: sequence of (block id, segment before it) in time order using op stream
seq=[[] for _ in range(n)]; seg=[np.eye(2,dtype=complex) for _ in range(n)]
# block id per cz via same rule
owner=[None]*n; blk=[]; bid=[]
cur_blk=[None]*n
for o in ops:
    if o[0]=='u3':
        q=o[1][0]; seg[q]=u3m(*o[2])@seg[q]; continue
    a,b=o[1]; p=tuple(sorted((a,b)))
    if owner[a] is not None and owner[a]==owner[b] and blk[owner[a]]==p: k=owner[a]; inner=True
    else: blk.append(p); k=len(blk)-1; owner[a]=owner[b]=k; inner=False
    for q in (a,b):
        if not inner: seq[q].append((k,seg[q]))
        seg[q]=np.eye(2,dtype=complex)
FS={'inv':lambda A:A.conj().T,'T':lambda A:A.T,'conj':lambda A:A.conj(),'id':lambda A:A}
def dist(U):
    U=U/np.sqrt(np.linalg.det(U)); return min(np.linalg.norm(U-np.eye(2)),np.linalg.norm(U+np.eye(2)))
res=[]
for a in range(n):
    s=seq[a]; pa=int(pi[a])
    sp={k:i for i,(k,_) in enumerate(seq[pa])}
    for i in range(1,len(s)):
        k0,k1=s[i-1][0],s[i][0]
        j0,j1=M[k0],M[k1]
        if j0<0 or j1<0 or L[k1]>17 or L[k0]<2: continue
        if j0 not in sp or j1 not in sp: res.append((L[k0],'partner not on pi(a)')); continue
        i0,i1=sp[j0],sp[j1]
        if i0!=i1+1: res.append((L[k0],'not adjacent %d %d'%(i1,i0))); continue
        sv=s[i][1]; svp=seq[pa][i0][1]   # segment before partner of k0 (V order: j1 then j0)
        res.append((L[k0],{nm:dist(f(sv).conj().T@svp) for nm,f in FS.items()}))
num=[r for r in res if not isinstance(r[1],str)]
print('checked',len(res),'numeric',len(num))
for nm in FS:
    d=np.array([r[1][nm] for r in num]); print(nm,'quantiles',np.quantile(d,[0,.1,.5,.9,1]))
import collections; print(collections.Counter(r[1] for r in res if isinstance(r[1],str)).most_common(5))
