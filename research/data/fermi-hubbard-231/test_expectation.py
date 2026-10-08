import quimb.tensor as qtn
import numpy as np

psi = qtn.MPS_product_state([np.array([0,1,0,0], dtype=complex) for _ in range(4)])
N_up = np.zeros((4,4)); N_up[1,1]=1; N_up[3,3]=1
res1 = psi.compute_local_expectation({(1,): N_up})
print(res1)
