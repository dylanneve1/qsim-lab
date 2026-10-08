import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import itertools
import quimb.tensor as qtn

L_list = [4, 6]
U = -2.0
t_h = 1.0
dt = 0.2

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
                # To match exactly what Quimb's get_H_bond does, 
                # wait! Quimb's hop_up is C_up.T @ Z ... which is actually +t_h * (hopping).
                # But let me just use the EXACT SAME matrix elements that `H_bond` has!
                # Wait, I can just construct H_even and H_odd by expanding `H_bond` directly!
                pass
                
        # Expanding H_bond:
        # H_bond is 16x16, acting on sites i, i+1.
        for b in basis:
            s_i = (b[2*i], b[2*i+1])
            s_j = (b[2*(i+1)], b[2*(i+1)+1])
            idx_in = s_i[0]*8 + s_i[1]*4 + s_j[0]*2 + s_j[1]
            
            for out_i_up in [0,1]:
                for out_i_dn in [0,1]:
                    for out_j_up in [0,1]:
                        for out_j_dn in [0,1]:
                            idx_out = out_i_up*8 + out_i_dn*4 + out_j_up*2 + out_j_dn
                            val = H_bond[idx_out, idx_in]
                            if val != 0:
                                b_out = list(b)
                                b_out[2*i] = out_i_up
                                b_out[2*i+1] = out_i_dn
                                b_out[2*(i+1)] = out_j_up
                                b_out[2*(i+1)+1] = out_j_dn
                                
                                # Apply parity! Wait. The local basis matrix H_bond doesn't know about 
                                # the rest of the chain! Does Quimb apply parity automatically?
                                # No! Quimb's LocalHam1D just applies the 16x16 matrix to sites i and i+1.
                                # BUT if it's fermions, the matrix must include the JW string.
                                # Our H_bond includes Z for the site 1. Does it need a Z for all sites < i?
                                # Yes, but a 2-body term c_i^\dagger c_{i+1} has Z_i. The strings from <i cancel!
                                # Because it's c_i^\dagger Z_{i-1}... Z_0 Z_0... Z_{i-1} c_{i+1} -> the Zs cancel!
                                # So NO parity string is needed from the rest of the chain!
                                # Thus, Quimb's application of H_bond directly to sites i and i+1 is EXACTLY correct for 1D nearest neighbor hopping.
                                H_bond_exact[basis_dict[tuple(b_out)], basis_dict[b]] += val
        
        # Add H1 to the bond matrix exactly as Quimb folds it.
        # Quimb LocalHam1D folds H1 into the first and last bond.
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
                
        if i % 2 == 0:
            H_even += H_bond_exact
        else:
            H_odd += H_bond_exact

    U_even = expm(-1j * dt * H_even.tocsc())
    U_odd = expm(-1j * dt * H_odd.tocsc())
    
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

    psi_exact_1 = U_step_1 @ psi_exact_1
    
    tebd.update_to(dt, order=2, dt=dt)
    tebd.pt.normalize()
    
    n_up_exact_1 = sum(num(1, 'up', b)[0] * abs(psi_exact_1[i])**2 for i, b in enumerate(basis))
    n_up_quimb = tebd.pt.compute_local_expectation({ (1,): N_up}).real
    
    print(f"Quimb n_up: {n_up_quimb:.8f}")
    print(f"Exact_1 n_up (even-odd-even): {n_up_exact_1:.8f}, Diff: {abs(n_up_exact_1 - n_up_quimb):.2e}")

