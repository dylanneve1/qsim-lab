import sys, time
sys.path.insert(0, '/tmp/doped-zx')
import circ
import numpy as np
import quimb.tensor as qtn

def get_boundary_state(gates, n, e, x_bits, drop_t=False):
    """
    Construct the boundary tensor on the CZ bonds crossing edge (e, e+1)
    from contracting the left region (qubits 0..e).
    """
    # Filter gates: only keep gates acting on qubits <= e,
    # except CZs crossing (e, e+1) which get replaced by |b><b| on qubit e
    # and open bond index b_k.
    bond_count = 0
    t_count = 0
    
    # We build a Circuit or TN on qubits 0..e
    # Wire indices for qubits 0..e
    c = qtn.Circuit(e + 1)
    bond_tags = []
    
    for g in gates:
        if g[0] == 'cz':
            q1, q2 = sorted([g[1], g[2]])
            if q2 <= e:
                c.apply_gate('CZ', q1, q2)
            elif q1 <= e and q2 == e + 1:
                # This CZ crosses the cut!
                # It acts on qubit e as projector |b><b| where b is an open bond index
                # CZ = |0><0| (x) I + |1><1| (x) Z
                # On qubit e: if bond=0, |0><0|; if bond=1, |1><1|
                # In tensor form: a 3-index tensor on (in_wire, out_wire, bond)
                # T[in, out, bond] = 1 if in==out==bond else 0
                bond_name = f'bond_{bond_count}'
                bond_count += 1
                
                # Apply projector tensor on qubit e
                # We can do this by applying a custom gate or tensor
                # Equivalently: apply a 2-qubit gate where qubit e+1 is a dummy qubit, but simpler:
                # Use apply_gate_raw with an extra index
                proj_mat = np.zeros((2, 2, 2), dtype=complex) # (out, in, bond)
                proj_mat[0, 0, 0] = 1.0
                proj_mat[1, 1, 1] = 1.0
                
                # In quimb, we can add a tensor directly to the TN
                # Let's get the current wire index for qubit e
                wire = c.psi.site_ind(e)
                new_wire = f'w_{e}_{bond_count}'
                t = qtn.Tensor(data=proj_mat, inds=(new_wire, wire, bond_name), tags={f'CZ_BOND_{bond_count}'})
                c.psi.add_tensor(t)
                c.psi.reindex_({new_wire: wire}) # update phys ind
            else:
                pass # gate entirely in right region
        elif g[0] == 't':
            if g[1] <= e:
                t_count += 1
                if not drop_t:
                    c.apply_gate_raw(circ.MATS['t'], (g[1],))
        else:
            if g[1] <= e:
                c.apply_gate_raw(circ.MATS[g[0]], (g[1],))
                
    # Now project onto <x_0 ... x_e|
    tn = c.psi
    for q in range(e + 1):
        x_val = x_bits[q]
        proj = np.zeros(2, dtype=complex)
        proj[x_val] = 1.0
        tn.add_tensor(qtn.Tensor(data=proj, inds=(tn.site_ind(q),), tags={f'PROJ_{q}'}))
        
    # Contract all indices except bond_0 .. bond_{m-1}
    bond_inds = [f'bond_{k}' for k in range(bond_count)]
    # Contract
    res = tn.contract(output_inds=bond_inds)
    v = res.data.reshape(-1)
    return v, bond_count, t_count

def analyze_stabilizers(v, m):
    """
    Check exact Pauli stabilizers and nullity for an m-qubit state v.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return {'norm': 0, 'stabilizers': 0}
    v = v / norm
    
    # Fast Walsh-Hadamard transform based Pauli expectation computation
    # For small m (<= 12), we can compute all 4^m Pauli expectations:
    # Tr(rho P) for P = P_1 (x) ... (x) P_m
    # In statevector form: <v| P |v>
    # P in {I, X, Y, Z}
    # Density matrix rho = |v><v|
    # A Pauli P has <v|P|v> in [-1, 1] (for Hermitian Paulis)
    paulis = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex)
    ]
    
    # For m <= 10:
    if m <= 10:
        rho = np.outer(v, v.conj()).reshape([2, 2] * m)
        # Pauli basis expansion of rho:
        # rho = sum_P c_P P / 2^m, where c_P = <v|P|v>
        # Each single-qubit factor: Tr(rho_i P_a) / 2
        # We can transform rho along each pair of axes (in_k, out_k) using the 4x4 matrix
        # M_a,ij = P_a[i,j]^* = P_a[j,i]
        basis_transform = np.zeros((4, 2, 2), dtype=complex)
        for a in range(4):
            basis_transform[a] = paulis[a]
            
        C = rho
        for k in range(m):
            # C has shape (4^k, 2, 2, 2^(2*(m-k-1)))
            # Contract basis_transform with axes (2*k, 2*k+1)
            C = np.tensordot(basis_transform, C, axes=([1, 2], [2*k, 2*k+1]))
            # Move the new Pauli index (axis 0) to the end
            C = np.moveaxis(C, 0, -1)
            
        # C now has shape (4,)*m, real entries <v|P|v>
        exp_vals = C.real.reshape(-1)
        # Exactly +/- 1 stabilizers (excluding identity at index 0)
        # Check |<v|P|v>| >= 1 - 1e-6
        stabs = np.where(np.abs(exp_vals) >= 1.0 - 1e-6)[0]
        # Number of stabilizers (including identity):
        num_stabs = len(stabs)
        # Stabilizer dimension k = log2(num_stabs)
        stab_dim = int(round(np.log2(num_stabs))) if num_stabs > 0 else 0
        
        # Rényi-2 magic M2 = -log2( sum c_P^4 / 2^m )
        purity4 = np.sum(exp_vals**4) / (2**m)
        m2 = -np.log2(purity4)
        
        # Max non-identity expectation:
        non_id = np.abs(exp_vals[1:])
        max_non_id = np.max(non_id) if len(non_id) > 0 else 0
        
        return {
            'm': m,
            'stab_dim': stab_dim,
            'num_stabs': num_stabs,
            'm2': m2,
            'max_non_id': max_non_id
        }
    else:
        # For larger m (e.g. 12-16), sampling or WHT method
        p = np.abs(v)**2
        return {'m': m, 'note': 'larger m'}

if __name__ == '__main__':
    for D in [10, 16, 20]:
        gates = circ.load(D=D, n=20)
        e = 9 # cut at middle
        x = [0]*20
        # Undoped
        v_cliff, m, _ = get_boundary_state(gates, 20, e, x, drop_t=True)
        res_cliff = analyze_stabilizers(v_cliff, m)
        # Doped
        v_doped, m, tc = get_boundary_state(gates, 20, e, x, drop_t=False)
        res_doped = analyze_stabilizers(v_doped, m)
        print(f"D={D:2d} m={m:2d} (bonds at cut e={e}):")
        print(f"  UNDOPED: stab_dim={res_cliff['stab_dim']}/{m}, M2={res_cliff['m2']:.4f}, max_Pauli={res_cliff['max_non_id']:.4f}")
        print(f"  DOPED (T={tc:2d}): stab_dim={res_doped['stab_dim']}/{m}, M2={res_doped['m2']:.4f}, max_Pauli={res_doped['max_non_id']:.4f}")
