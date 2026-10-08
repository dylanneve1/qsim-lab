"""Compile decoded ops into per-step rung-level brick structure:
 step = [layerA: {s: U16 on rungs (s,s+1)}], [layerB], diag: (60,4) phases per rung.
Rung basis x = 2*n_i + n_o ; pair basis 4*x_s + x_{s+1}."""
import numpy as np, pickle
def hop2(V):
    # 2-qubit unitary on (n_s, n_{s+1}) basis index 2*n_s+n_{s+1}; V acts on (amp particle-on-s, amp on-s+1)
    U=np.zeros((4,4),complex); U[0,0]=1; U[3,3]=np.linalg.det(V)
    U[2,2]=V[0,0]; U[2,1]=V[0,1]; U[1,2]=V[1,0]; U[1,1]=V[1,1]; return U
def compile_ops(circ,lam=1.0):
    n0,ops=pickle.load(open(f'/tmp/su2-254-opus/ops_{circ}.pkl','rb'))
    steps=[]; cur=None
    def new(): return dict(layers=[],diag=np.zeros((60,4)),hops={})
    cur=new(); phase='hop'
    for o in ops:
        if o[0]=='step':
            steps.append(cur); cur=new(); continue
        if o[0]=='h':
            _,l,s1,s2,V=o; a=min(s1,s2); Vab=V if s1<s2 else V[::-1,::-1]
            assert not cur['diag'].any()
            # start a new layer if bond parity changes
            if not cur['layers'] or (cur['layers'][-1]['par']!=a%2):
                cur['layers'].append(dict(par=a%2,g={}))
            cur['layers'][-1]['g'].setdefault(a,{})[l]=hop2(Vab)
        elif o[0]=='p':
            _,l,s,ph=o; x=np.arange(4); n=(x>>1)&1 if l==0 else x&1
            cur['diag'][s]+=ph*n
        else:
            _,s,g=o; x=np.arange(4); cur['diag'][s]+=lam*g*((x>>1)&1)*(x&1)
    # build U16 per bond
    I2=np.eye(4)
    for st in steps:
        for L in st['layers']:
            G={}
            for a,d in L['g'].items():
                U0=d.get(0,I2); U1=d.get(1,I2)
                # legs (ni_s, no_s, ni_s1, no_s1); U0 on (ni_s,ni_s1), U1 on (no_s,no_s1)
                T=np.einsum('ACac,BDbd->ABCDabcd',U0.reshape(2,2,2,2),U1.reshape(2,2,2,2)).reshape(16,16)
                G[a]=T
            L['G']=G
    return n0,steps
if __name__=='__main__':
    n0,steps=compile_ops('SCV')
    print(len(steps),[ (len(L['G']),L['par']) for L in steps[0]['layers']],[ (len(L['G']),L['par']) for L in steps[-1]['layers']])
    print(np.round(steps[0]['diag'][:3],4))
