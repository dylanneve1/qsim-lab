import numpy as np, quimb.tensor as qtn, math
from parse import parse
def mat(n,v):
    if n=='rx': c,s=math.cos(v/2),math.sin(v/2); return np.array([[c,-1j*s],[-1j*s,c]])
    if n=='rz': return np.diag([np.exp(-1j*v/2),np.exp(1j*v/2)])
    if n=='s': return np.diag([1,1j])
    if n=='sdg': return np.diag([1,-1j])
    if n=='sx': return 0.5*np.array([[1+1j,1-1j],[1-1j,1+1j]])
    if n=='sxdg': return 0.5*np.array([[1-1j,1+1j],[1+1j,1-1j]])
def build_ops(ops,O,dtype='complex128',simplify=True,z=None,seq='ADCRS'):
    """f*2^n = sum_{a,b} s_a s_b |<b|W|a>|^2 for gate list ops (W)."""
    CZ=np.diag([1,1,1,-1.]).reshape(2,2,2,2).astype(complex)
    qs=sorted({q for o in ops for q in o[1]}); n=len(qs)
    ts=[]; cnt=[0]
    def new(): cnt[0]+=1; return f'x{cnt[0]}'
    for side in ('k','b'):
        cur={q:f'a{q}' for q in qs}
        for nm,qq,v in ops:
            if nm=='cz':
                p,q=qq; na,nb=new(),new(); ts.append(qtn.Tensor(CZ,(na,nb,cur[p],cur[q]))); cur[p]=na; cur[q]=nb
            else:
                q=qq[0]; m=mat(nm,v)
                if side=='b': m=m.conj()
                nn=new(); ts.append(qtn.Tensor(m,(nn,cur[q]))); cur[q]=nn
        for q in qs: ts.append(qtn.Tensor(np.eye(2,dtype=complex),(cur[q],f'b{q}')))
    zz=np.array([1.,-1.],dtype=complex)
    if z is None:
        for q in O: ts.append(qtn.Tensor(zz,(f'a{q}',))); ts.append(qtn.Tensor(zz,(f'b{q}',)))
    else:  # fixed input basis state z (dict q->bit); returns sum_b s_b |<b|W|z>|^2
        for q in qs: ts.append(qtn.Tensor(np.eye(2,dtype=complex)[z[q]],(f'a{q}',)))
        for q in O: ts.append(qtn.Tensor(zz,(f'b{q}',)))
    tn=qtn.TensorNetwork(ts).astype(dtype)
    if simplify: tn=tn.full_simplify(output_inds=(),seq=seq)
    return tn,n
def build(qasm,obs,keep=None,dtype='complex128'):
    O=[int(x) for x in obs.split(',')]
    ops=[o for o in parse(qasm) if o[0]!='barrier']
    if keep: K=set(int(x) for x in keep.split(',')); ops=[o for o in ops if all(q in K for q in o[1])]
    return build_ops(ops,O,dtype)
