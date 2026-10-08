import numpy as np
import quimb.tensor as qtn
import pickle

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

L = 30
U = -2.0
t_h = 1.0
dt = 0.2
steps = 30

H_bond, N_up, N_dn = get_H_bond(t_h, U)
H2 = { (i, i+1): H_bond for i in range(L-1) }
N_updn = N_up @ N_dn
H1 = { 0: (U/2) * N_updn, L-1: (U/2) * N_updn }

ham = qtn.LocalHam1D(L, H2=H2, H1=H1)

initial_states = [np.array([0,0,1,0], dtype=complex) if i % 2 == 0 else np.array([0,1,0,0], dtype=complex) for i in range(L)]
psi0 = qtn.MPS_product_state(initial_states)

chi_max = 256
# Add cutoff and max_bond to TEBD init
tebd = qtn.TEBD(psi0, ham, dt=dt, split_opts={'max_bond': chi_max, 'cutoff': 1e-8})

n_up_res = []
n_dn_res = []

def measure(psi):
    try:
        n_up = psi.compute_local_expectation({i: N_up for i in range(L)})
        n_dn = psi.compute_local_expectation({i: N_dn for i in range(L)})
        n_up_res.append([n_up[i].real for i in range(L)])
        n_dn_res.append([n_dn[i].real for i in range(L)])
    except:
        n_up = [np.real((psi.H @ psi.gate(N_up, i)).item()) for i in range(L)]
        n_dn = [np.real((psi.H @ psi.gate(N_dn, i)).item()) for i in range(L)]
        n_up_res.append(n_up)
        n_dn_res.append(n_dn)

measure(tebd.pt.copy())

for step in range(steps):
    tebd.step() # takes one step of size dt
    measure(tebd.pt.copy())
    print(f"Step {step+1} done.")

with open(f"/tmp/fh-231/tebd_res_chi{chi_max}.pkl", "wb") as f:
    pickle.dump({"up": np.array(n_up_res), "dn": np.array(n_dn_res)}, f)
