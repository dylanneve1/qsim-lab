import numpy as np
from scipy.linalg import expm

def get_H_bond(t_h, U):
    # Basis for one site: 0: empty, 1: up, 2: down, 3: up+down
    # Parity Z: diag(1, -1, -1, 1)
    Z = np.diag([1, -1, -1, 1])
    
    # Creation operators
    C_up = np.zeros((4,4))
    C_up[1,0] = 1; C_up[3,2] = 1
    
    # For down, we apply Z of up first
    Z_up = np.diag([1, -1, 1, -1])
    C_dn_raw = np.zeros((4,4))
    C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
    C_dn = Z_up @ C_dn_raw
    
    # H_bond
    H = np.zeros((16,16))
    
    # Hopping up
    hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
    # Hopping down
    hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)
    
    H += -t_h * (hop_up + hop_dn)
    
    # U term: applied half on left, half on right
    N_up = C_up.T @ C_up
    N_dn = C_dn.T @ C_dn
    N_updn = N_up @ N_dn
    H += (U/2) * (np.kron(N_updn, np.eye(4)) + np.kron(np.eye(4), N_updn))
    
    return H

H = get_H_bond(1.0, -2.0)
print("H shape:", H.shape)
print("H hermitian:", np.allclose(H, H.T.conj()))
