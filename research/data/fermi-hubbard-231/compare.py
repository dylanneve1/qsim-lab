import pickle
import numpy as np
import glob
import os

chi_list = [64, 128, 256, 512]
tebd_data = {}
for chi in chi_list:
    if os.path.exists(f"/tmp/fh-231/tebd_res_chi{chi}.pkl"):
        with open(f"/tmp/fh-231/tebd_res_chi{chi}.pkl", "rb") as f:
            tebd_data[chi] = pickle.load(f)

with open("/tmp/fh-231/tdvp_data_1_pt.pkl", "rb") as f:
    tdvp_1 = pickle.load(f)
with open("/tmp/fh-231/tdvp_data_2_pt.pkl", "rb") as f:
    tdvp_2 = pickle.load(f)
with open("/tmp/fh-231/mm_and_dr_data_1_pt.pkl", "rb") as f:
    hw_1 = pickle.load(f)
with open("/tmp/fh-231/mm_and_dr_data_2_pt.pkl", "rb") as f:
    hw_2 = pickle.load(f)

t_steps = [5, 10, 15, 20, 25, 30] # t=1,2,3,4,5,6
site_tdvp = 29
site_tebd = 15

observables = ['n_up', 'n_dn', 'nn']
labels = ['<n_{c,up}>', '<n_{c,dn}>', '<n_{c,up} n_{c,dn}>']

print("# Comparison Table")
for t_idx in t_steps:
    t = t_idx * 0.2
    print(f"\n## t = {t:.1f} (Step {t_idx})")
    
    # TDVP and HW values
    v_tdvp = {}
    if t_idx < 30:
        v_tdvp['n_up'] = tdvp_1[(site_tdvp, 'up')][t_idx]
        v_tdvp['n_dn'] = tdvp_1[(site_tdvp, 'down')][t_idx]
        v_tdvp['nn'] = tdvp_2[(site_tdvp, 'down', site_tdvp, 'up')][t_idx]
    else:
        v_tdvp['n_up'] = np.nan
        v_tdvp['n_dn'] = np.nan
        v_tdvp['nn'] = np.nan
        
    v_hw = {}
    if t_idx < len(hw_1[(site_tdvp, 'up')]):
        v_hw['n_up'] = hw_1[(site_tdvp, 'up')][t_idx]
        v_hw['n_dn'] = hw_1[(site_tdvp, 'down')][t_idx]
        v_hw['nn'] = hw_2[(site_tdvp, 'down', site_tdvp, 'up')][t_idx]
    else:
        v_hw['n_up'] = np.nan
        v_hw['n_dn'] = np.nan
        v_hw['nn'] = np.nan
        
    print("| Observable | " + " | ".join([f"TEBD(chi={c})" for c in chi_list]) + " | TDVP | Hardware | Error Bar |")
    print("|---|-" + "-|-".join([""]*len(chi_list)) + "-|---|---|---|")
    
    for i, obs in enumerate(observables):
        row = f"| {labels[i]} | "
        tebd_vals = []
        for chi in chi_list:
            if chi in tebd_data and t_idx <= len(tebd_data[chi][obs]):
                val = tebd_data[chi][obs][t_idx] # tebd_res includes initial state? No, step 0 is t=0!
                # Wait, my tebd_res_chi.pkl stores step=0...30
                # length of array is 31 (initial + 30 steps)
                # t_idx is exactly the index!
                val = tebd_data[chi][obs][t_idx]
                row += f"{val:.6f} | "
                tebd_vals.append(val)
            else:
                row += "N/A | "
        
        row += f"{v_tdvp[obs]:.6f} | {v_hw[obs]:.6f} | "
        
        # calculate error bar
        err = 0.0
        if len(tebd_vals) >= 2:
            spread = abs(tebd_vals[-1] - tebd_vals[-2])
            if not np.isnan(v_tdvp[obs]):
                trotter_err = abs(tebd_vals[-1] - v_tdvp[obs])
            else:
                trotter_err = 0.0
            err = spread + trotter_err
            row += f"{err:.6e} |"
        else:
            row += "N/A |"
            
        print(row)
