import quimb.tensor as qtn
import numpy as np

def get_H_bond():
    Z = np.diag([1, -1, -1, 1])
    C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
    C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
    Z_up = np.diag([1, -1, 1, -1])
    C_dn = Z_up @ C_dn_raw
    H = np.zeros((16,16), dtype=complex)
    hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
    hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)
    H += -1.0 * (hop_up + hop_dn)
    return H

L = 4
H2 = {(i, i+1): get_H_bond() for i in range(L-1)}
ham = qtn.LocalHam1D(L, H2=H2)

initial_states = [np.array([0,0,1,0], dtype=complex) if i % 2 == 0 else np.array([0,1,0,0], dtype=complex) for i in range(L)]
psi0 = qtn.MPS_product_state(initial_states)

tebd = qtn.TEBD(psi0, ham)
tebd.split_opts = {'max_bond': 10, 'cutoff': 1e-10}

for _ in range(5):
    tebd.update_to(tebd.t + 0.2, dt=0.1)
    norm = tebd.pt.H @ tebd.pt
    print("t=", tebd.t, "norm=", abs(norm))
