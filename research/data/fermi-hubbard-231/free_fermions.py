import numpy as np
import scipy.linalg

def free_fermion_evolution(L, t_h, dt, steps):
    # H = -t_h \sum (c^\dagger_i c_{i+1} + h.c.)
    H = np.zeros((L, L))
    for i in range(L-1):
        H[i, i+1] = -t_h
        H[i+1, i] = -t_h
        
    # Initial state: Neel
    # dn, up, dn, up ...
    # For up spins: site 1, 3, 5, ... are 1. site 0, 2, 4 are 0.
    # For dn spins: site 0, 2, 4, ... are 1. site 1, 3, 5 are 0.
    
    C_up = np.zeros((L, L), dtype=complex)
    C_dn = np.zeros((L, L), dtype=complex)
    for i in range(L):
        if i % 2 == 1:
            C_up[i, i] = 1.0
        else:
            C_dn[i, i] = 1.0
            
    U_evol = scipy.linalg.expm(-1j * H * dt)
    U_dag = U_evol.T.conj()
    
    res_up = [np.diag(C_up).real.copy()]
    res_dn = [np.diag(C_dn).real.copy()]
    
    for _ in range(steps):
        C_up = U_evol @ C_up @ U_dag
        C_dn = U_evol @ C_dn @ U_dag
        res_up.append(np.diag(C_up).real.copy())
        res_dn.append(np.diag(C_dn).real.copy())
        
    return np.array(res_up), np.array(res_dn)

res_up, res_dn = free_fermion_evolution(30, 1.0, 0.2, 30)
print(res_up.shape)
