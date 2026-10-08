import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import itertools
import quimb.tensor as qtn

L_list = [4, 6]
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

    # Quimb's exact TEBD Hamiltonian splits into H_even and H_odd.
    # Actually, quimb Trotter step order 2:
    # exp(-dt/2 H_even) exp(-dt H_odd) exp(-dt/2 H_even) (if even-odd, or vice versa)
    # Let's construct the exact sparse matrices for even and odd bonds.
    H_even = sp.lil_matrix((dim, dim), dtype=complex)
    H_odd = sp.lil_matrix((dim, dim), dtype=complex)
    
    for i in range(L-1):
        H_bond_exact = sp.lil_matrix((dim, dim), dtype=complex)
        for spin in ['up', 'down']:
            for b in basis:
                s1, st1 = c_ann(i+1, spin, b)
                if s1 != 0:
                    s2, st2 = c_dag(i, spin, st1)
                    if s2 != 0:
                        # Wait! I found that the quimb code implements +t_h (c_1^\dagger c_2 + h.c.)!
                        # The exact script from earlier used -t_h. Let me match Quimb exactly first
                        # to see if my exact code and quimb match.
                        val = t_h * s1 * s2 # Because quimb effectively does +t_h. Let's test this!
                        H_bond_exact[basis_dict[st2], basis_dict[b]] += val
                
                s1, st1 = c_ann(i, spin, b)
                if s1 != 0:
                    s2, st2 = c_dag(i+1, spin, st1)
                    if s2 != 0:
                        val = t_h * s1 * s2
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

    U_even_half = expm(-1j * (dt/2) * H_even.tocsc())
    U_odd = expm(-1j * dt * H_odd.tocsc())
    
    # Quimb does: exp(-dt/2 H_even) exp(-dt H_odd) exp(-dt/2 H_even)
    # Wait, does it start with even or odd? LocalHam1D puts even first by default in TEBD.
    U_step = U_even_half @ U_odd @ U_even_half
    
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
            
    psi_exact = np.zeros(dim, dtype=complex)
    psi_exact[basis_dict[tuple(init_state_jw)]] = 1.0

    print("Checking step 1...")
    psi_exact = U_step @ psi_exact
    tebd.update_to(dt, order=2, dt=dt)
    tebd.pt.normalize()
    
    n_up_exact = sum(num(1, 'up', b)[0] * abs(psi_exact[i])**2 for i, b in enumerate(basis))
    n_up_quimb = tebd.pt.compute_local_expectation({ (1,): N_up}).real
    print(f"Exact n_up: {n_up_exact:.8f}")
    print(f"Quimb n_up: {n_up_quimb:.8f}")
    print(f"Diff: {abs(n_up_exact - n_up_quimb):.2e}")

