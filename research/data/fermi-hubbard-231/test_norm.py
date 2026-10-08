import pickle
import numpy as np

with open('tebd_res_chi64.pkl', 'rb') as f:
    data = pickle.load(f)

for step, d in enumerate(data):
    pass
    # We didn't save the norm, but we can see the particle number sum drops.
