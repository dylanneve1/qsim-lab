import numpy as np
import quimb.tensor as qtn

def get_H_bond(t_h, U):
    Z = np.diag([1, -1, -1, 1])
    C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
    C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
    Z_up = np.diag([1, -1, 1, -1])
    C_dn = Z_up @ C_dn_raw
    H = np.zeros((16,16), dtype=complex)
    hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
    hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)
    H += -t_h * (hop_up + hop_dn)
    N_up = C_up.T @ C_up
    N_dn = C_dn.T @ C_dn
    N_updn = N_up @ N_dn
    H += (U/2) * (np.kron(N_updn, np.eye(4)) + np.kron(np.eye(4), N_updn))
    return H, N_up, N_dn

L = 4
H_bond, N_up, N_dn = get_H_bond(1.0, -2.0)
ham = qtn.LocalHam1D(L, H2={(i, i+1): H_bond for i in range(L-1)})
psi0 = qtn.MPS_product_state([np.array([0,0,1,0], dtype=complex) if i%2==0 else np.array([0,1,0,0], dtype=complex) for i in range(L)])

tebd = qtn.TEBD(psi0, ham, dt=0.2, split_opts={'max_bond': 64, 'cutoff': 1e-8})
tebd.step()
print("err:", tebd.err)
print("max bond:", max(psi0.bond_sizes()))
print("new max bond:", max(tebd.pt.bond_sizes()))
