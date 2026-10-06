import sys, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
D='/tmp/pk/research/data/peaked-circuits/'
c=Core(D+'peaked_circuit_P11_Hqap_98x1999.qasm')
def unitary(ks, wires):
    n=len(wires); U=np.eye(2**n,dtype=complex); idx={w:i for i,w in enumerate(wires)}
    for k in sorted(ks):
        a,b=c.units[k][:2]; G=c.M[k].reshape(2,2,2,2)
        T=U.reshape((2,)*n+(2**n,))
        T=np.tensordot(G,T,axes=([2,3],[idx[a],idx[b]]))
        T=np.moveaxis(T,[0,1],[idx[a],idx[b]])
        U=T.reshape(2**n,2**n)
    return U
def kak_inv(U):
    # Makhlin invariants via magic basis
    B=np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
    Ub=B.conj().T@U@B; m=Ub.T@Ub; d=np.linalg.det(U)
    G1=np.trace(m)**2/(16*d); G2=(np.trace(m)**2-np.trace(m@m))/(4*d)
    return np.round(G1,4),np.round(G2,4)
SW=np.eye(4)[[0,2,1,3]]
print('CZ invariants',kak_inv(np.diag([1,1,1,-1]).astype(complex)),' identity',kak_inv(np.eye(4,dtype=complex)),' SWAP',kak_inv(SW.astype(complex)))
U=unitary([644,651,676,693,698,719],[8,10,72,75])
# split {8,10}|{72,75}
T=U.reshape(2,2,2,2,2,2,2,2).transpose(0,1,4,5,2,3,6,7).reshape(16,16)
s=np.linalg.svd(T,compute_uv=False); print('opSchmidt {8,10}|{72,75}', np.round(s**2/np.sum(s**2),4))
Uu,s,Vh=np.linalg.svd(T)
A=Uu[:,0].reshape(4,4)*np.sqrt(s[0]); B=Vh[0].reshape(4,4)*np.sqrt(s[0])
A/=abs(np.linalg.det(A))**0.25; B/=abs(np.linalg.det(B))**0.25
print('factor {8,10}', kak_inv(A), ' factor {72,75}', kak_inv(B))
print('SWAP*CZ', kak_inv(SW@np.diag([1,1,1,-1]).astype(complex)))
