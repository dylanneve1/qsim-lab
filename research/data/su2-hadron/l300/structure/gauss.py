"""Free-fermion (U=0, exact) and time-dependent Hartree (TDH) simulation of the circuits.
Decodes the circuit into fused 1-/2-wire blocks on logical wires, verifies hop blocks are
number-conserving Gaussian, on-site blocks diagonal."""
import numpy as np, sys, os, json, argparse
from parse import load
from graph_util import logical_map
Hm=np.array([[1,1],[1,-1]])/np.sqrt(2); Xm=np.array([[0,1],[1,0]],dtype=complex)
def rz(t): return np.diag([np.exp(-1j*t/2),np.exp(1j*t/2)])
CX=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
def blocks(path):
    ops=load(path); k=0; init=[]
    while ops[k][0]=='x': init.append(ops[k][2][0]); k+=1
    m=logical_map(path); occ=[0]*120
    for q in init: occ[q]^=1
    lab=list(range(120)); out=[]; pend={}  # wire -> block dict
    nm=0
    def flush(b):
        out.append(b)
        for w in b['w']: pend.pop(w,None)
    for name,p,qs in ops[k:]:
        if name=='swap':
            a,b=qs; lab[a],lab[b]=lab[b],lab[a]; continue
        W=[lab[q] for q in qs]
        U=CX if name=='cx' else (Hm if name=='h' else Xm if name=='x' else rz(p))
        if name=='x' and W[0] in pend and len(pend[W[0]]['w'])==2 and m[pend[W[0]]['w'][0]][1]==m[pend[W[0]]['w'][1]][1]:
            flush(pend[W[0]])
        bl=[]; 
        for w in W:
            if w in pend and pend[w] not in bl: bl.append(pend[w])
        keep=[]
        for b in bl:
            if set(b['w'])<=set(W) or (len(W)==1 and len(b['w'])<=2): keep.append(b)
            else: flush(b)
        bl=keep
        # merge bl into one block on wires (ordered list)
        wl=sorted(set(W)|set(x for b in bl for x in b['w']))
        M=np.eye(2**len(wl),dtype=complex)
        for b in bl: M=embed(b['U'],[wl.index(x) for x in b['w']],len(wl))@M
        M=embed(U,[wl.index(x) for x in W],len(wl))@M
        nb={'w':wl,'U':M}
        for w in wl: pend[w]=nb
        if name=='rz' and abs(abs(p)-0.03)<1e-12:
            nm+=1
            if nm%120==0:
                for b in list({id(b):b for b in pend.values()}.values()): flush(b)
                out.append({'step':nm//120})
    return occ,m,out
def embed(op,legs,n):
    k=len(legs); op=op.reshape([2]*(2*k)); T=np.eye(2**n,dtype=complex).reshape([2]*(2*n))
    T=np.tensordot(op,T,axes=(list(range(k,2*k)),legs))
    rest=[a for a in range(2*n) if a not in legs]; order=[None]*(2*n)
    for j,l in enumerate(legs): order[l]=j
    for j,a in enumerate(rest): order[a]=k+j
    return np.transpose(T,order).reshape(2**n,2**n)
def run(circ,mode):
    D=os.environ.get('SU2_CIRCUITS','circuits')+'/'
    occ,m,out=blocks(D+f'x_100_{circ}.qasm')
    rho=[np.zeros((60,60),dtype=complex) for _ in range(2)]
    for w in range(120):
        s,l=m[w]; rho[l][s,s]=occ[w]
    rec=[]; maxoff=0; kinds={}
    for b in out:
        if 'step' in b:
            n=np.array([[rho[l][s,s].real for l in (0,1)] for s in range(60)])
            stag=sum((-1)**r*(n[r,0]+n[r,1]) for r in range(60))
            rec.append(dict(step=b['step'],Q=float(n.sum()),stag=float(stag),n=n.tolist())); continue
        w=b['w']; U=b['U']; sl=[m[x] for x in w]
        if len(w)==1:
            assert abs(U[0,1])<1e-12 and abs(U[1,0])<1e-12
            s,l=sl[0]; ph=U[1,1]/U[0,0]; rho[l][s,:]*=ph; rho[l][:,s]*=np.conj(ph); kinds['1']=kinds.get('1',0)+1
        else:
            (s1,l1),(s2,l2)=sl
            if l1==l2:   # hop on chain l between sites s1,s2 (adjacent)
                assert abs(s1-s2)==1
                off=max(abs(U[0,1:]).max(),abs(U[1:,0]).max(),abs(U[3,:3]).max(),abs(U[:3,3]).max()); maxoff=max(maxoff,off)
                V=U[1:3,1:3]/U[0,0]   # basis |01>=particle on w[1], |10>=particle on w[0]
                # single-particle basis order (w0, w1): |10>->idx0, |01>->idx1
                Vsp=np.array([[V[1,1],V[1,0]],[V[0,1],V[0,0]]])
                gerr=abs(U[3,3]/U[0,0]-np.linalg.det(Vsp)); maxoff=max(maxoff,gerr)
                idx=[s1,s2]; R=rho[l1]
                R[idx,:]=Vsp@R[idx,:]; R[:,idx]=R[:,idx]@Vsp.conj().T
                kinds['hop']=kinds.get('hop',0)+1
            else:        # on-site (i_r,o_r)
                assert s1==s2 and np.allclose(U,np.diag(np.diag(U)))
                ph=np.angle(np.diag(U)); # order |w0 w1>
                a0=ph[2]-ph[0]; a1=ph[1]-ph[0]; g=ph[3]-ph[2]-ph[1]+ph[0]
                g=(g+np.pi)%(2*np.pi)-np.pi
                kinds['onsite']=kinds.get('onsite',0)+1; kinds['gmax']=max(kinds.get('gmax',0),abs(g))
                n0=rho[l1][s1,s1].real; n1=rho[l2][s2,s2].real
                if mode=='free': g=0
                p0=np.exp(1j*(a0+g*n1)); p1=np.exp(1j*(a1+g*n0))
                for (l,s,p) in ((l1,s1,p0),(l2,s2,p1)):
                    rho[l][s,:]*=p; rho[l][:,s]*=np.conj(p)
    return rec,maxoff,kinds
if __name__=='__main__':
    res={}
    for mode in ('free','hartree'):
        for circ in ('SCV','meson'):
            rec,mo,kinds=run(circ,mode); res[(mode,circ)]=rec
            print(mode,circ,'gauss-check',f'{mo:.1e}',kinds)
    out={}
    for mode in ('free','hartree'):
        print(mode)
        for k in range(20):
            s=res[(mode,'SCV')][k]; me=res[(mode,'meson')][k]
            print(f"  step {k+1:2d} Q={s['Q']:.10f} stag_SCV={s['stag']:.6f} stag_meson={me['stag']:.6f} n_f={me['stag']-s['stag']:.6f}")
        out[mode]={c:res[(mode,c)] for c in ('SCV','meson')}
    json.dump(out,open('gauss.json','w'))
