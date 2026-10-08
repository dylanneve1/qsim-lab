import pickle
import numpy as np

with open("/tmp/fh-231/tdvp_data_1_pt.pkl", "rb") as f:
    tdvp_1 = pickle.load(f)

with open("/tmp/fh-231/mm_and_dr_data_1_pt.pkl", "rb") as f:
    hw_1 = pickle.load(f)

with open("/tmp/fh-231/tdvp_data_2_pt.pkl", "rb") as f:
    tdvp_2 = pickle.load(f)

with open("/tmp/fh-231/mm_and_dr_data_2_pt.pkl", "rb") as f:
    hw_2 = pickle.load(f)

site = 29
print(f"Center site: {site}")
for t_idx in [5, 10, 15, 20, 25, 30]:
    t = t_idx * 0.2
    print(f"t={t:.1f} (index {t_idx}):")
    if t_idx < len(tdvp_1[(site, 'up')]):
        print(f"  TDVP: n_up={tdvp_1[(site, 'up')][t_idx]:.6f}, n_dn={tdvp_1[(site, 'down')][t_idx]:.6f}, n_up_dn={tdvp_2[(site, 'down', site, 'up')][t_idx]:.6f}")
    else:
        print(f"  TDVP: Missing")
    
    if t_idx < len(hw_1[(site, 'up')]):
        print(f"  HW:   n_up={hw_1[(site, 'up')][t_idx]:.6f}, n_dn={hw_1[(site, 'down')][t_idx]:.6f}, n_up_dn={hw_2[(site, 'down', site, 'up')][t_idx]:.6f}")
    else:
        print(f"  HW:   Missing")
