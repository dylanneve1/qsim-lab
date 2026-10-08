import numpy as np
import quimb.tensor as qtn
import pickle
import time
import os

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
    # C_up and C_dn are creation operators. So N = C @ C.T
    N_up = C_up @ C_up.T
    N_dn = C_dn @ C_dn.T
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

def run_chi(chi_max):
    print(f"Running chi={chi_max}...")
    psi0 = qtn.MPS_product_state(initial_states)
    
    # We will use dt=0.2 and order=2, meaning 2 sub-steps per Trotter step of 0.2
    tebd = qtn.TEBD(psi0, ham)
    tebd.split_opts = {'max_bond': chi_max, 'cutoff': 1e-10, 'renorm': 1}

    n_up_res = []
    n_dn_res = []
    nn_res = []
    

    def measure(psi):
        psi.normalize()
        # We only need site 15 for our analysis

        n_up = psi.compute_local_expectation({ (15,): N_up}).real
        n_dn = psi.compute_local_expectation({ (15,): N_dn}).real
        nn = psi.compute_local_expectation({ (15,): N_updn}).real
        n_up_res.append(n_up)
        n_dn_res.append(n_dn)
        nn_res.append(nn)

    measure(tebd.pt.copy())
    
    for step in range(steps):
        t0 = time.time()
        tebd.update_to(dt * (step + 1), order=2, dt=0.2) # 2 TEBD steps
        measure(tebd.pt.copy())
        print(f"chi={chi_max} Step {step+1} took {time.time()-t0:.2f}s, max_bond={max(tebd.pt.bond_sizes())}")

        res = []
        for step_idx in range(len(n_up_res)):
            res.append({
                'n_up': n_up_res[step_idx], # site 15
                'n_dn': n_dn_res[step_idx],
                'nn': nn_res[step_idx]
            })

        with open(f"/tmp/fh-231/tebd_res_chi{chi_max}.pkl", "wb") as f:
            pickle.dump(res, f)

for chi in [16, 32]:
    run_chi(chi)
