import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import itertools
import quimb.tensor as qtn

L_list = [4, 6]
U = -2.0
t_h = 1.0
dt = 0.2

# We want H = -t_h (c_1^\dagger c_2 + h.c.)
def get_H_bond(t_h, U):
    Z = np.diag([1, -1, -1, 1])
    C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
    C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
    Z_up = np.diag([1, -1, 1, -1])
    C_dn = Z_up @ C_dn_raw
    H = np.zeros((16,16), dtype=complex)
    
    # Let's derive it directly
    # c_1^\dagger c_2 = C_up \otimes (Z C_up.T) ? NO.
    # JW string: c_1^\dagger = C_up \otimes I. c_2 = Z \otimes C_up.T
    # So c_1^\dagger c_2 = (C_up \otimes I) (Z \otimes C_up.T) = C_up Z \otimes C_up.T
    hop_up_correct = np.kron(C_up @ Z, C_up.T) + np.kron(Z @ C_up.T, C_up)
    
    # c_1dn^\dagger c_2dn = (C_dn \otimes I) (Z \otimes C_dn.T) = C_dn Z \otimes C_dn.T
    hop_dn_correct = np.kron(C_dn @ Z, C_dn.T) + np.kron(Z @ C_dn.T, C_dn)
    
    H += -t_h * (hop_up_correct + hop_dn_correct)
    
    N_up = C_up @ C_up.T
    N_dn = C_dn @ C_dn.T
    N_updn = N_up @ N_dn
    H += (U/2) * (np.kron(N_updn, np.eye(4)) + np.kron(np.eye(4), N_updn))
    return H, N_up, N_dn

H_bond, N_up, N_dn = get_H_bond(t_h, U)

for L in L_list:
    print(f"\n--- Testing L={L} ---")
    def jw_basis(L):
        return list(itertools.product([0, 1], repeat=2*L))

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
                # c_i^\dagger c_{i+1}
                s1, st1 = c_ann(i+1, spin, b)
                if s1 != 0:
                    s2, st2 = c_dag(i, spin, st1)
                    if s2 != 0:
                        val = -t_h * s1 * s2
                        H_bond_exact[basis_dict[st2], basis_dict[b]] += val
                # c_{i+1}^\dagger c_i
                s1, st1 = c_ann(i, spin, b)
                if s1 != 0:
                    s2, st2 = c_dag(i+1, spin, st1)
                    if s2 != 0:
                        val = -t_h * s1 * s2
                        H_bond_exact[basis_dict[st2], basis_dict[b]] += val
        
        # U terms
        for b in basis:
            nu0, _ = num(i, 'up', b)
            nd0, _ = num(i, 'down', b)
            nu1, _ = num(i+1, 'up', b)
            nd1, _ = num(i+1, 'down', b)
            H_bond_exact[basis_dict[b], basis_dict[b]] += (U/2) * (nu0 * nd0 + nu1 * nd1)
            
        if i % 2 == 0:
            H_even += H_bond_exact
        else:
            H_odd += H_bond_exact

    # Quimb's exact TEBD splits bonds into even and odd arrays.
    # We can ask quimb to print the exact dense unitary! Wait, let's just let quimb run.
    U_even = expm(-1j * dt * H_even.tocsc())
    U_odd = expm(-1j * dt * H_odd.tocsc())
    # Try different Trotter products to see which one quimb uses:
    U_step_1 = expm(-1j * (dt/2) * H_even.tocsc()) @ expm(-1j * dt * H_odd.tocsc()) @ expm(-1j * (dt/2) * H_even.tocsc())
    U_step_2 = expm(-1j * (dt/2) * H_odd.tocsc()) @ expm(-1j * dt * H_even.tocsc()) @ expm(-1j * (dt/2) * H_odd.tocsc())
    
    H2 = { (i, i+1): H_bond for i in range(L-1) }
    H1 = { 0: (U/2) * N_up @ N_dn, L-1: (U/2) * N_up @ N_dn }
    ham = qtn.LocalHam1D(L, H2=H2, H1=H1)
    
    initial_states = [np.array([0,0,1,0], dtype=complex) if i % 2 == 0 else np.array([0,1,0,0], dtype=complex) for i in range(L)]
    psi0 = qtn.MPS_product_state(initial_states)
    tebd = qtn.TEBD(psi0, ham)
    tebd.split_opts = {'max_bond': 256, 'cutoff': 1e-12, 'renorm': 1}
    
    init_state_jw = []
    for i in range(L):
        if i % 2 == 0:
            init_state_jw.extend([0, 1])
        else:
            init_state_jw.extend([1, 0])
            
    psi_exact_1 = np.zeros(dim, dtype=complex)
    psi_exact_1[basis_dict[tuple(init_state_jw)]] = 1.0
    psi_exact_2 = psi_exact_1.copy()

    psi_exact_1 = U_step_1 @ psi_exact_1
    psi_exact_2 = U_step_2 @ psi_exact_2
    
    tebd.update_to(dt, order=2, dt=dt)
    tebd.pt.normalize()
    
    n_up_exact_1 = sum(num(1, 'up', b)[0] * abs(psi_exact_1[i])**2 for i, b in enumerate(basis))
    n_up_exact_2 = sum(num(1, 'up', b)[0] * abs(psi_exact_2[i])**2 for i, b in enumerate(basis))
    n_up_quimb = tebd.pt.compute_local_expectation({ (1,): N_up}).real
    
    print(f"Quimb n_up: {n_up_quimb:.8f}")
    print(f"Exact_1 n_up (even-odd-even): {n_up_exact_1:.8f}, Diff: {abs(n_up_exact_1 - n_up_quimb):.2e}")
    print(f"Exact_2 n_up (odd-even-odd): {n_up_exact_2:.8f}, Diff: {abs(n_up_exact_2 - n_up_quimb):.2e}")

