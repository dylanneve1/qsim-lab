import pickle
import numpy as np

with open('tebd_res_chi64.pkl', 'rb') as f:
    data = pickle.load(f)

for step, d in enumerate(data):
    print(f"step {step}, n_up={d['n_up']}, n_dn={d['n_dn']}, sum={d['n_up']+d['n_dn']}")
