import numpy as np

Z = np.diag([1, -1, -1, 1])
C_up = np.zeros((4,4)); C_up[1,0] = 1; C_up[3,2] = 1
C_dn_raw = np.zeros((4,4)); C_dn_raw[2,0] = 1; C_dn_raw[3,1] = 1
Z_up = np.diag([1, -1, 1, -1])
C_dn = Z_up @ C_dn_raw

hop_up = np.kron(C_up.T @ Z, C_up) + np.kron(Z @ C_up, C_up.T)
hop_dn = np.kron(C_dn.T @ Z, C_dn) + np.kron(Z @ C_dn, C_dn.T)

# In the exact basis, does hop_up have the same sign as hop_dn?
# Let's check matrix element for |up, 0> -> |0, up>
# state |up, 0> is index 1*4 + 0 = 4.
# state |0, up> is index 0*4 + 1 = 1.
print("hop_up |up,0> -> |0,up>:", hop_up[1, 4])

# state |dn, 0> is index 2*4 + 0 = 8.
# state |0, dn> is index 0*4 + 2 = 2.
print("hop_dn |dn,0> -> |0,dn>:", hop_dn[2, 8])

