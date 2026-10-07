"""Heisenberg images of X_q, Z_q through the ops of one epoch (exact Pauli propagation on gate level, eps truncation).
usage: gapact.py EPOCH"""
import sys, numpy as np, collections, math
from parse import *
from segment import epochs, epochs3
import os
sys.path.insert(0,'/tmp/peaked-p5p6')
from segment import epochs4
G=load(); E={"3":epochs3,"4":epochs4}.get(os.environ.get("SEG"),epochs)()
PM=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])]
def U3(t,p,l): return np.array([[math.cos(t/2),-np.exp(1j*l)*math.sin(t/2)],[np.exp(1j*p)*math.sin(t/2),np.exp(1j*(p+l))*math.cos(t/2)]])
def ptm1(M):
    T=np.zeros((4,4))
    for i in range(4):
        A=M.conj().T@PM[i]@M
        for j in range(4): T[i,j]=np.real(np.trace(PM[j]@A))/2
    return T
# CZ PTM on 2 qubits
P2=[np.kron(PM[i],PM[j]) for i in range(4) for j in range(4)]
CZ=np.diag([1,1,1,-1]).astype(complex)
TCZ=np.zeros((16,16))
for i in range(16):
    A=CZ@P2[i]@CZ
    for j in range(16): TCZ[i,j]=np.real(np.trace(P2[j]@A))/4
def image(ops,q,p,eps=1e-6,maxterms=100000):
    cur={((q,p),):1.0}
    for g in reversed(ops):
        new=collections.defaultdict(float)
        if g[0]=='u3':
            a=g[1][0]; T=ptm1(U3(*g[2]))
            for key,c in cur.items():
                d=dict(key); pa=d.pop(a,0)
                if pa==0: new[key]+=c; continue
                for j in range(1,4):
                    if abs(T[pa,j])>1e-12:
                        dd=dict(d); dd[a]=j; new[tuple(sorted(dd.items()))]+=c*T[pa,j]
        else:
            a,b=g[1]
            for key,c in cur.items():
                d=dict(key); pa=d.pop(a,0); pb=d.pop(b,0)
                if pa==0 and pb==0: new[key]+=c; continue
                row=TCZ[pa*4+pb]; j=int(np.argmax(np.abs(row))); ja,jb=divmod(j,4); dd=dict(d)
                if ja: dd[a]=ja
                if jb: dd[b]=jb
                new[tuple(sorted(dd.items()))]+=c*row[j]
        cur={k:v for k,v in new.items() if abs(v)>eps}
        if len(cur)>maxterms: return None
    return cur
def analyse(e,verbose=True):
    ops=[G[i] for i in range(len(G)) if E[i]==e]
    nm=lambda q: 'ABC'[ringof[q]]+str(pos[q])
    res={}
    for q in range(62):
        out=[]
        for p in (1,3):
            im=image(ops,q,p)
            if im is None: out.append((None,0,0)); continue
            w=collections.defaultdict(float)
            for key,v in im.items():
                if len(key)==1: w[key[0][0]]+=v*v
            best=max(w,key=w.get) if w else None
            out.append((best,w.get(best,0),len(im)))
        res[q]=out
    return ops,res
if __name__=='__main__':
    e=int(sys.argv[1]); ops,res=analyse(e)
    nm=lambda q: 'ABC'[ringof[q]]+str(pos[q]) if q is not None else '-'
    ncz=sum(g[0]=='cz' for g in ops)
    print('epoch',e,'ops',len(ops),'cz',ncz)
    for q in range(62):
        (bx,wx,nx),(bz,wz,nz)=res[q]
        print(nm(q),'X->',nm(bx),round(wx,3),nx,' Z->',nm(bz),round(wz,3),nz)
