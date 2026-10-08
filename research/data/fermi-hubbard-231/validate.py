import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import itertools
import quimb.tensor as qtn

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

for L in [4, 6]:
    print(f"\n--- Validation L={L} ---")
    def jw_basis(L): return list(itertools.product([0, 1], repeat=2*L))
    basis = jw_basis(L)
    basis_dict = {b: i for i, b in enumerate(basis)}
    dim = len(basis)

    def c_dag(i, spin, state):
        idx = 2*i + (0 if spin == 'up' else 1)
        if state[idx] == 1: return 0, None
        parity = sum(state[:idx])
        ns = list(state); ns[idx] = 1
        return (-1)**parity, tuple(ns)

    def c_ann(i, spin, state):
        idx = 2*i + (0 if spin == 'up' else 1)
        if state[idx] == 0: return 0, None
        parity = sum(state[:idx])
        ns = list(state); ns[idx] = 0
        return (-1)**parity, tuple(ns)

    def num(i, spin, state):
        idx = 2*i + (0 if spin == 'up' else 1)
        return state[idx], state

    H_even = sp.lil_matrix((dim, dim), dtype=complex)
    H_odd = sp.lil_matrix((dim, dim), dtype=complex)
    
    for i in range(L-1):
        H_bond_exact = sp.lil_matrix((dim, dim), dtype=complex)
        for spin in ['up', 'down']:
            for b in basis:
                s1, st1 = c_ann(i+1, spin, b)
                if s1 != 0:
                    s2, st2 = c_dag(i, spin, st1)
                    if s2 != 0: H_bond_exact[basis_dict[st2], basis_dict[b]] += -t_h * s1 * s2
                s1, st1 = c_ann(i, spin, b)
                if s1 != 0:
                    s2, st2 = c_dag(i+1, spin, st1)
                    if s2 != 0: H_bond_exact[basis_dict[st2], basis_dict[b]] += -t_h * s1 * s2
        
        for b in basis:
            nu0, _ = num(i, 'up', b)
            nd0, _ = num(i, 'down', b)
            nu1, _ = num(i+1, 'up', b)
            nd1, _ = num(i+1, 'down', b)
            H_bond_exact[basis_dict[b], basis_dict[b]] += (U/2) * (nu0 * nd0 + nu1 * nd1)
            
        if i == 0:
            for b in basis:
                nu, _ = num(0, 'up', b)
                nd, _ = num(0, 'down', b)
                H_bond_exact[basis_dict[b], basis_dict[b]] += (U/2) * nu * nd
        if i == L-2:
            for b in basis:
                nu, _ = num(L-1, 'up', b)
                nd, _ = num(L-1, 'down', b)
                H_bond_exact[basis_dict[b], basis_dict[b]] += (U/2) * nu * nd
                
        if i % 2 == 0: H_even += H_bond_exact
        else: H_odd += H_bond_exact

    U_even = expm(-1j * (dt/2) * H_even.tocsc())
    U_odd = expm(-1j * dt * H_odd.tocsc())
    U_step = U_even @ U_odd @ U_even
    
    H2 = { (i, i+1): H_bond for i in range(L-1) }
    H1 = { 0: (U/2) * N_up @ N_dn, L-1: (U/2) * N_up @ N_dn }
    ham = qtn.LocalHam1D(L, H2=H2, H1=H1)
    
    initial_states = [np.array([0,0,1,0], dtype=complex) if i % 2 == 0 else np.array([0,1,0,0], dtype=complex) for i in range(L)]
    psi0 = qtn.MPS_product_state(initial_states)
    tebd = qtn.TEBD(psi0, ham)
    tebd.split_opts = {'max_bond': 256, 'cutoff': 0.0, 'renorm': 1}
    
    init_state_jw = []
    for i in range(L):
        if i % 2 == 0: init_state_jw.extend([0, 1])
        else: init_state_jw.extend([1, 0])
    psi_exact = np.zeros(dim, dtype=complex)
    psi_exact[basis_dict[tuple(init_state_jw)]] = 1.0

    print("Checking evolution...")
    for step in range(1, steps+1):
        psi_exact = U_step @ psi_exact
        tebd.update_to(dt * step, order=2, dt=dt)
        tebd.pt.normalize()
        
        tot_up = 0
        tot_dn = 0
        tot_n = 0
        for i in range(L):
            n_up_ex = sum(num(i, 'up', b)[0] * abs(psi_exact[idx])**2 for idx, b in enumerate(basis))
            n_dn_ex = sum(num(i, 'down', b)[0] * abs(psi_exact[idx])**2 for idx, b in enumerate(basis))
            n_up_quimb = tebd.pt.compute_local_expectation({ (i,): N_up}).real
            n_dn_quimb = tebd.pt.compute_local_expectation({ (i,): N_dn}).real
            if abs(n_up_ex - n_up_quimb) > 1e-8:
                print(f"Mismatch at step {step}, site {i}, up: {n_up_ex} vs {n_up_quimb}")
                exit(1)
            tot_up += n_up_quimb
            tot_dn += n_dn_quimb
            tot_n += n_up_quimb + n_dn_quimb
        assert abs(tot_up - L/2) < 1e-8
        assert abs(tot_dn - L/2) < 1e-8
        assert abs(tot_n/L - 1) < 1e-8
    print("Exact validation passed!")
