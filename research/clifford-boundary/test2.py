import numpy as np, stim, diag, diag2
rng=np.random.default_rng(0)
for m in [8,10]:
    v=stim.Tableau.random(m).to_state_vector(endian='little').astype(complex)
    T=v.reshape([2]*m).transpose(list(range(m))[::-1]).copy(); print('stab',m,diag2.m2(T,200,rng))
    v=rng.normal(size=2**m)+1j*rng.normal(size=2**m); v/=np.linalg.norm(v)
    T=v.reshape([2]*m).transpose(list(range(m))[::-1]).copy(); print('haar',m,diag2.m2(T,400,rng))
    # one T on |+>: M2 = log2(4/3)=0.415
v=np.array([1,np.exp(1j*np.pi/4)])/np.sqrt(2); print('Tstate',diag2.m2(v.reshape(2),400,rng))
