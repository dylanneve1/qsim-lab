import re, sys, math, collections, numpy as np
f=sys.argv[1]; src=open(f).read()
n=int(re.search(r'\[(\d+)\];',src).group(1))
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
def num(x): return float(eval(x,{'pi':math.pi}))
CZ=np.diag([1,1,1,-1]).astype(complex)
ops=[]
for l in src.splitlines():
    m=re.match(r'(u3|cz)(?:\(([^)]*)\))?\s+([^;]*);',l.strip())
    if not m: continue
    qs=[int(x) for x in re.findall(r'\[(\d+)\]',m.group(3))]
    ops.append(('u',qs[0],U3(*map(num,m.group(2).split(',')))) if m.group(1)=='u3' else ('cz',tuple(qs),None))
# build blocks: consecutive CZs on same pair (1q gates between them on those wires go inside)
wire=collections.defaultdict(list)   # per wire: list of events
blocks=[]  # each: pair, list of ops inside
open_blk={}  # pair -> block idx if still open
last2q={}
pend=collections.defaultdict(lambda: np.eye(2,dtype=complex))
for k,q,M in ops:
    if k=='u': pend[q]=M@pend[q]; continue
    a,b=q
    if last2q.get(a)==q and last2q.get(b)==q and q in open_blk:
        B=blocks[open_blk[q]]
        B['U']=CZ@np.kron(pend[a],pend[b])@B['U']; B['ncz']+=1
    else:
        blocks.append({'pair':q,'U':CZ.copy(),'ncz':1,'t':k}); open_blk[q]=len(blocks)-1
    pend[a]=np.eye(2,dtype=complex); pend[b]=np.eye(2,dtype=complex)
    last2q[a]=q; last2q[b]=q
# Makhlin invariants
Q=np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
def mak(U):
    U=U/np.linalg.det(U)**0.25
    UB=Q.conj().T@U@Q; m=UB.T@UB
    t=np.trace(m); return (t*t/16, (t*t-np.trace(m@m))/4)
inv=[]
for i,B in enumerate(blocks):
    g1,g2=mak(B['U']); inv.append((g1,g2))
big=[i for i,B in enumerate(blocks) if B['ncz']>=2]
print(f.split('/')[-1],'blocks',len(blocks),'multi-CZ blocks',len(big))
# match block i with j>i such that inv(j) == inverse-invariants of i: G1->conj(G1), G2 same
key=lambda g: (round(g[0].real,4),round(g[0].imag,4),round(g[1].real,4))
ikey=lambda g: (round(g[0].real,4),round(-g[0].imag,4),round(g[1].real,4))
by=collections.defaultdict(list)
for i in big: by[key(inv[i])].append(i)
pairs=[]
for i in big:
    for j in by.get(ikey(inv[i]),[]):
        if j>i: pairs.append((i,j))
print('inverse-invariant matches among multi-CZ blocks:', len(pairs), ' unique-key matches:', sum(1 for i,j in pairs if len(by[key(inv[j])])==1 and len(by[ikey(inv[i])])==1))
if pairs:
    mids=[(i+j)/2 for i,j in pairs]; h=np.histogram(mids,bins=20,range=(0,len(blocks)))[0]; print('mirror-centre histogram (block index):',h.tolist())
    same=sum(1 for i,j in pairs if blocks[i]['pair']==blocks[j]['pair']); print('same wire pair:',same)
