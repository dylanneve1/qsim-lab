import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm
import pickle
import time

def get_H_bond(t_h, U):
    Z = np.diag([1, -1, -1, 1])
    C_up = np.zeros((4,4))
    C_up[1,0] = 1; C_up[3,2] = 1
    Z_up = np.diag([1, -1, 1, -1])
    C_dn_raw = np.zeros((4,4))
    C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
    C_dn = Z_up @ C_dn_raw
    
    H = np.zeros((16,16))
    hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
    hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)
    H += -t_h * (hop_up + hop_dn)
    
    N_up = C_up.T @ C_up
    N_dn = C_dn.T @ C_dn
    N_updn = N_up @ N_dn
    H += (U/2) * (np.kron(N_updn, np.eye(4)) + np.kron(np.eye(4), N_updn))
    
    return H, N_up, N_dn

L = 30
U = -2.0
t_h = 1.0
dt = 0.2
steps = 30

H, N_up, N_dn = get_H_bond(t_h, U)
# Fix boundary U: add the remaining U/2 for sites 0 and L-1
# Wait, TEBD3 or TEBD2 handles boundaries automatically if we use LocalHam1D.
# In quimb, we can just define a LocalHam1D.
