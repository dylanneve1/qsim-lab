import pickle

with open("/tmp/fh-231/tdvp_data_2_pt.pkl", "rb") as f:
    tdvp_2 = pickle.load(f)

same_site_keys = [k for k in tdvp_2.keys() if k[0] == k[2]]
print("Same site keys example:", same_site_keys[:5])
