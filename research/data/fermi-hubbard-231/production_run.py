import sys
import numpy as np
import quimb.tensor as qtn
import pickle
import time

chi_max = int(sys.argv[1])

L = 30
U = -2.0
t_h = 1.0
dt = 0.2
steps = 30

def get_H_bond(t_h, U):
    Z = np.diag([1, -1, -1, 1])
    C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
    C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
    Z_up = np.diag([1, -1, 1, -1])
    C_dn = Z_up @ C_dn_raw
    H = np.zeros((16,16), dtype=complex)
    hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
    hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)
    
    H += t_h * (hop_up + hop_dn)
    
    N_up = C_up @ C_up.T
    N_dn = C_dn @ C_dn.T
    N_updn = N_up @ N_dn
    H += (U/2) * (np.kron(N_updn, np.eye(4)) + np.kron(np.eye(4), N_updn))
    return H, N_up, N_dn

H_bond, N_up, N_dn = get_H_bond(t_h, U)
H2 = { (i, i+1): H_bond for i in range(L-1) }
H1 = { 0: (U/2) * N_up @ N_dn, L-1: (U/2) * N_up @ N_dn }
ham = qtn.LocalHam1D(L, H2=H2, H1=H1)

initial_states = [np.array([0,0,1,0], dtype=complex) if i % 2 == 0 else np.array([0,1,0,0], dtype=complex) for i in range(L)]
psi0 = qtn.MPS_product_state(initial_states)

tebd = qtn.TEBD(psi0, ham)
tebd.split_opts = {'max_bond': chi_max, 'cutoff': 1e-10, 'renorm': 1}

n_up_res = []
n_dn_res = []
nn_res = []
err_res = []
ent_res = []
step_times = []

def measure(psi):
    psi.normalize()
    # site 14 and 15 are the center. The L=60 case had center 29.
    # Site 29 was |up>, so an odd site. Site 15 is odd. 
    site = 15
    n_up = psi.compute_local_expectation({ (site,): N_up}).real
    n_dn = psi.compute_local_expectation({ (site,): N_dn}).real
    nn = psi.compute_local_expectation({ (site,): N_up @ N_dn}).real
    n_up_res.append(n_up)
    n_dn_res.append(n_dn)
    nn_res.append(nn)

measure(tebd.pt.copy())

for step in range(steps):
    t0 = time.time()
    tebd.update_to(dt * (step + 1), order=2, dt=dt)
    t1 = time.time()
    measure(tebd.pt.copy())
    
    # tebd.err tracks the total truncated weight or similar, wait.
    # Quimb TEBD err is the sum of truncated weights. 
    # To get max discarded weight we might need to track it manually if tebd.err is cumulative.
    # Actually tebd.err is cumulative. So the discarded weight in this step is tebd.err - old_err.
    err_res.append(tebd.err)
    
    # max bond entropy
    # psi.entropy(i) gives entropy at bond i.
    entropies = [tebd.pt.entropy(i) for i in range(1, L)]
    ent_res.append(max(entropies))
    step_times.append(t1 - t0)
    
    print(f"chi={chi_max} Step {step+1} took {t1-t0:.2f}s, max_bond={max(tebd.pt.bond_sizes())}, err={tebd.err:.2e}, max_ent={max(entropies):.4f}")
    
    # Log every step so partial progress survives
    with open(f"/tmp/fh-231/tebd_res_chi{chi_max}.pkl", "wb") as f:
        pickle.dump({
            'n_up': n_up_res,
            'n_dn': n_dn_res,
            'nn': nn_res,
            'err': err_res,
            'ent': ent_res,
            'times': step_times
        }, f)

