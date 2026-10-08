import numpy as np

Z = np.diag([1, -1, -1, 1])
C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
Z_up = np.diag([1, -1, 1, -1])
C_dn = Z_up @ C_dn_raw

hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)

N_up1 = np.kron(C_up @ C_up.T, np.eye(4))
N_up2 = np.kron(np.eye(4), C_up @ C_up.T)

N_tot_up = N_up1 + N_up2

comm_up = hop_up @ N_tot_up - N_tot_up @ hop_up
print("Commutator of hop_up with total up number:", np.linalg.norm(comm_up))

N_dn1 = np.kron(C_dn @ C_dn.T, np.eye(4))
N_dn2 = np.kron(np.eye(4), C_dn @ C_dn.T)
N_tot_dn = N_dn1 + N_dn2

comm_dn = hop_dn @ N_tot_dn - N_tot_dn @ hop_dn
print("Commutator of hop_dn with total dn number:", np.linalg.norm(comm_dn))

