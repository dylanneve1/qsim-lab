import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import itertools
import quimb.tensor as qtn

L = 4
U = -2.0
t_h = 1.0
dt = 0.2

def jw_basis(L):
    return list(itertools.product([0, 1], repeat=2*L))

basis = jw_basis(L)
basis_dict = {b: i for i, b in enumerate(basis)}
dim = len(basis)

def c_dag(i, spin, state):
    idx = 2*i + (0 if spin == 'up' else 1)
    if state[idx] == 1:
        return 0, None
    parity = sum(state[:idx])
    new_state = list(state)
    new_state[idx] = 1
    return (-1)**parity, tuple(new_state)

def c_ann(i, spin, state):
    idx = 2*i + (0 if spin == 'up' else 1)
    if state[idx] == 0:
        return 0, None
    parity = sum(state[:idx])
    new_state = list(state)
    new_state[idx] = 0
    return (-1)**parity, tuple(new_state)

def num(i, spin, state):
    idx = 2*i + (0 if spin == 'up' else 1)
    return state[idx], state

H_hop = sp.lil_matrix((dim, dim), dtype=complex)
H_U = sp.lil_matrix((dim, dim), dtype=complex)

for i in range(L-1):
    for spin in ['up', 'down']:
        for b in basis:
            s1, st1 = c_ann(i+1, spin, b)
            if s1 != 0:
                s2, st2 = c_dag(i, spin, st1)
                if s2 != 0:
                    val = -t_h * s1 * s2
                    H_hop[basis_dict[st2], basis_dict[b]] += val
            
            s1, st1 = c_ann(i, spin, b)
            if s1 != 0:
                s2, st2 = c_dag(i+1, spin, st1)
                if s2 != 0:
                    val = -t_h * s1 * s2
                    H_hop[basis_dict[st2], basis_dict[b]] += val

for i in range(L):
    for b in basis:
        nu, _ = num(i, 'up', b)
        nd, _ = num(i, 'down', b)
        H_U[basis_dict[b], basis_dict[b]] += U * nu * nd

H_total = H_hop + H_U
U_exact = expm(-1j * dt * H_total.tocsc())

# Quimb
import run_tebd
psi0 = qtn.MPS_product_state(run_tebd.initial_states[:L])
tebd = qtn.TEBD(psi0, run_tebd.ham)
tebd.split_opts = {'max_bond': 64, 'cutoff': 1e-10, 'renorm': 1}

# Create exact initial state
init_b = tuple(run_tebd.initial_states[i][2] if i%2==0 else run_tebd.initial_states[i][1] for i in range(L))
# Wait, initial_states has shape (4,). [0,0,1,0] means down spin. So up=0, down=1.
# [0,1,0,0] means up spin. up=1, down=0.
init_state_jw = []
for i in range(L):
    if i % 2 == 0:
        init_state_jw.extend([0, 1])
    else:
        init_state_jw.extend([1, 0])
psi_exact = np.zeros(dim, dtype=complex)
psi_exact[basis_dict[tuple(init_state_jw)]] = 1.0

# Do 1 step
psi_exact = U_exact @ psi_exact

tebd.update_to(dt, order=2, dt=dt)
tebd.pt.normalize()

# Compare observables at site 1
n_up_exact = sum(num(1, 'up', b)[0] * abs(psi_exact[i])**2 for i, b in enumerate(basis))
n_up_quimb = tebd.pt.compute_local_expectation({ (1,): run_tebd.N_up}).real

print(f"Exact n_up: {n_up_exact:.8f}")
print(f"Quimb n_up: {n_up_quimb:.8f}")
print(f"Diff: {abs(n_up_exact - n_up_quimb):.2e}")

