"""Group gates into 2q blocks (maximal runs of CZs on the same pair with only 1q gates between on those wires),
compute block core unitary (from first CZ to last CZ incl. interior 1q gates) and Weyl coordinates."""
import numpy as np,sys,collections
from cstruct import load
def u3m(t,p,l):
    return np.array([[np.cos(t/2),-np.exp(1j*l)*np.sin(t/2)],[np.exp(1j*p)*np.sin(t/2),np.exp(1j*(p+l))*np.cos(t/2)]])
CZ=np.diag([1,1,1,-1]).astype(complex)
Bm=np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
def weyl(U):
    U=U/np.linalg.det(U)**0.25
    Up=Bm.conj().T@U@Bm
    ev=np.linalg.eigvals(Up.T@Up)
    th=np.angle(ev)/2
    # canonical coords via sorted
    c=np.sort(np.mod(th,np.pi))
    return c
def makhlin(U):
    U=U/np.linalg.det(U)**0.25
    Up=Bm.conj().T@U@Bm; m=Up.T@Up
    tr=np.trace(m); return (tr**2/16, (tr**2-np.trace(m@m))/4)
def blocks(ops,n):
    # each wire: pending 1q ops since last cz
    pend=[[] for _ in range(n)]
    owner=[None]*n  # open block index on wire
    B=[]
    for o in ops:
        if o[0]=='u3':
            q=o[1][0]; pend[q].append(o)
        else:
            a,b=o[1]; p=tuple(sorted((a,b)))
            if owner[a] is not None and owner[a]==owner[b] and B[owner[a]]['pair']==p:
                blk=B[owner[a]]
                blk['seq']+= [('mid',a,pend[a]),('mid',b,pend[b]),('cz',)]
                blk['ncz']+=1
            else:
                blk={'pair':p,'pre':{a:pend[a],b:pend[b]},'seq':[('cz',)],'ncz':1}
                B.append(blk); owner[a]=owner[b]=len(B)-1
            pend[a]=[];pend[b]=[]
        # a 1q op does not close a block; a cz on another pair does:
        if o[0]=='cz':
            pass
    return B
def blocks_v2(ops,n):
    # block closes on wire when that wire touches another cz
    owner=[None]*n; pend=[[] for _ in range(n)]; B=[]
    for o in ops:
        if o[0]=='u3': pend[o[1][0]].append(o); continue
        a,b=o[1]; p=tuple(sorted((a,b)))
        if owner[a] is not None and owner[a]==owner[b] and B[owner[a]]['pair']==p:
            blk=B[owner[a]]; blk['mid'].append((list(pend[a]),list(pend[b]),a,b)); blk['ncz']+=1
        else:
            for q in (a,b):
                if owner[q] is not None: B[owner[q]]['post'][q]=list(pend[q])
            B.append({'pair':p,'pre':{a:list(pend[a]),b:list(pend[b])},'mid':[],'ncz':1,'post':{}})
            owner[a]=owner[b]=len(B)-1
        pend[a]=[];pend[b]=[]
    return B
def core(blk):
    a,b=blk['pair']
    U=CZ.copy()
    for pa,pb,_,_ in blk['mid']:
        A=np.eye(2,dtype=complex)
        for o in (pa if True else []):
            A=u3m(*o[2])@A if o[1][0]==a else A
        Bq=np.eye(2,dtype=complex)
        for o in pa+pb:
            if o[1][0]==a: pass
        A=np.eye(2,dtype=complex); Bq=np.eye(2,dtype=complex)
        for o in pa+pb:
            M=u3m(*o[2])
            if o[1][0]==a: A=M@A
            else: Bq=M@Bq
        U=CZ@np.kron(A,Bq)@U   # wire a = first tensor factor (a<b)
    return U
if __name__=='__main__':
    fn=sys.argv[1]; ops=load(fn); n=max(max(o[1]) for o in ops)+1
    B=blocks_v2(ops,n)
    print('nblocks',len(B),collections.Counter(b['ncz'] for b in B))
    depth=[0]*n
    out=[]
    for b in B:
        a,c=b['pair']; l=max(depth[a],depth[c]); depth[a]=depth[c]=l+1
        w=weyl(core(b)); g=makhlin(core(b))
        out.append((a,c,l,b['ncz'],*w,g[0].real,g[0].imag,g[1].real))
    out=np.array(out); np.save(fn.split('/')[-1]+'.bw.npy',out)
    L=int(out[:,2].max())+1
    for l in range(L):
        r=out[out[:,2]==l]
        print(l,len(r),'ncz',collections.Counter(r[:,3].astype(int)),'G1re range %.3f..%.3f'%(r[:,7].min(),r[:,7].max()),'G2 %.3f..%.3f'%(r[:,9].min(),r[:,9].max()))
