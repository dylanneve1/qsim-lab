import quimb.tensor as qtn
import numpy as np

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
    N_up = C_up @ C_up.T
    N_dn = C_dn @ C_dn.T
    N_updn = N_up @ N_dn
    H += (U/2) * (np.kron(N_updn, np.eye(4)) + np.kron(np.eye(4), N_updn))
    return H, N_up, N_dn

L = 4
U = -2.0
t_h = 1.0

H_bond, N_up, N_dn = get_H_bond(t_h, U)
H2 = { (i, i+1): H_bond for i in range(L-1) }
N_updn = N_up @ N_dn
H1 = { 0: (U/2) * N_updn, L-1: (U/2) * N_updn }
ham = qtn.LocalHam1D(L, H2=H2, H1=H1)

initial_states = [np.array([0,0,1,0], dtype=complex) if i % 2 == 0 else np.array([0,1,0,0], dtype=complex) for i in range(L)]
psi = qtn.MPS_product_state(initial_states)

tebd = qtn.TEBD(psi, ham)
tebd.split_opts = {'max_bond': 100, 'cutoff': 1e-10, 'renorm': 1}

for step in range(5):
    tebd.update_to(0.2 * (step + 1), order=2, dt=0.1)
    tebd.pt.normalize()
    dens = []
    for i in range(L):
        n_up = tebd.pt.compute_local_expectation({(i,): N_up}).real
        n_dn = tebd.pt.compute_local_expectation({(i,): N_dn}).real
        dens.append(n_up + n_dn)
    print(f"step {step+1}, densities = {dens}")

