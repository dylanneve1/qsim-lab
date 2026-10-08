import pickle

with open("/tmp/fh-231/tdvp_data_1_pt.pkl", "rb") as f:
    tdvp_1 = pickle.load(f)

with open("/tmp/fh-231/mm_and_dr_data_1_pt.pkl", "rb") as f:
    hw_1 = pickle.load(f)

with open("/tmp/fh-231/tdvp_data_2_pt.pkl", "rb") as f:
    tdvp_2 = pickle.load(f)

with open("/tmp/fh-231/mm_and_dr_data_2_pt.pkl", "rb") as f:
    hw_2 = pickle.load(f)

for step in [8, 5, 10, 15, 20, 25, 30]:
    print(f"\nStep {step} (t={step*0.2:.1f}):")
    for site in [29, 30]:
        n_up_tdvp = tdvp_1[(site, 'up')][step]
        n_dn_tdvp = tdvp_1[(site, 'down')][step]
        n_up_dn_tdvp = tdvp_2[(site, 'down', site, 'up')][step]
        print(f"  Site {site}: <n_up>={n_up_tdvp:.6f}, <n_dn>={n_dn_tdvp:.6f}, <n_up n_dn>={n_up_dn_tdvp:.6f}")
