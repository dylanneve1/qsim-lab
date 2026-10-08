import numpy as np

Z = np.diag([1, -1, -1, 1])
C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
Z_up = np.diag([1, -1, 1, -1])
C_dn = Z_up @ C_dn_raw

hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)

print("hop_up hermitian:", np.allclose(hop_up, hop_up.conj().T))
print("hop_dn hermitian:", np.allclose(hop_dn, hop_dn.conj().T))

# Check eigenvalues of hopping
print("eig hop_up:", np.linalg.eigvalsh(hop_up))
print("eig hop_dn:", np.linalg.eigvalsh(hop_dn))
