import numpy as np
Bm=np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])/np.sqrt(2)
def weyl(U):
    U=U/np.linalg.det(U)**0.25
    Up=Bm.conj().T@U@Bm; m=Up.T@Up
    ev=np.linalg.eigvals(m); th=np.angle(ev)/2
    # canonical: c = sorted |.| reduced into Weyl chamber (approx; enough for classification)
    t=np.sort(np.mod(th,np.pi))[::-1]
    # coordinates via standard formula
    c=np.array([(t[0]+t[1])/2,(t[0]+t[2])/2,(t[1]+t[2])/2])
    c=np.mod(c+np.pi/4,np.pi/2)-np.pi/4
    c=np.sort(np.abs(c))[::-1]
    return c
def makhlin(U):
    U=U/np.linalg.det(U)**0.25
    Up=Bm.conj().T@U@Bm; m=Up.T@Up; tr=np.trace(m)
    return tr**2/16, (tr**2-np.trace(m@m))/4
def cls(U,tol=0.02):
    g1,g2=makhlin(U)
    if abs(g1-1)<tol and abs(g2-3)<tol: return 'loc'
    if abs(g1)<tol and abs(g2-1)<tol: return 'cnot'
    if abs(g1+1)<tol and abs(g2+3)<tol: return 'swap'
    if abs(g1)<tol and abs(g2+1)<tol: return 'iswap'
    return 'gen'
