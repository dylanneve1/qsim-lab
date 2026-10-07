import numpy as np, stim, sys
sys.path.insert(0,'.'); import diag
rng=np.random.default_rng(1)
m=10
for kind in ['stab','stab+T']:
    t=stim.Tableau.random(m); v=t.to_state_vector(endian='little')
    if kind=='stab+T':
        # 3 T gates on random qubits followed by a random Clifford
        for q in [1,4,7]:
            v=v.reshape(-1); idx=np.arange(2**m); v=v*np.where((idx>>q)&1, np.exp(1j*np.pi/4),1)
        t2=stim.Tableau.random(m); U=t2.to_unitary_matrix(endian='little'); v=U@v
    v=v/np.linalg.norm(v)
    T=v.reshape([2]*m).transpose(list(range(m))[::-1]).copy()
    raw=diag.cut_stats(T); Td,n=diag.disentangle(T,sweeps=8); dis=diag.cut_stats(Td)
    print(kind,'nullity_lb',diag.nullity_lb(T),'raw',raw[:,0].round(2),'dis',dis[:,0].round(3),n)
