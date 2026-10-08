with open('run_tebd.py', 'r') as f:
    content = f.read()

# Instead of dumping at the end, dump inside the loop
old_loop = """
    for step in range(steps):
        t0 = time.time()
        tebd.update_to(dt * (step + 1), order=2, dt=0.2) # 2 TEBD steps
        measure(tebd.pt.copy())
        print(f"chi={chi_max} Step {step+1} took {time.time()-t0:.2f}s, max_bond={max(tebd.pt.bond_sizes())}")

    # For the agent to be able to parse: store the dictionaries where we can retrieve easily
    res = []
    for step_idx in range(len(n_up_res)):
        res.append({
            'n_up': n_up_res[step_idx], # site 15
            'n_dn': n_dn_res[step_idx],
            'nn': nn_res[step_idx]
        })

    with open(f"/tmp/fh-231/tebd_res_chi{chi_max}.pkl", "wb") as f:
        pickle.dump(res, f)
"""

new_loop = """
    for step in range(steps):
        t0 = time.time()
        tebd.update_to(dt * (step + 1), order=2, dt=0.2) # 2 TEBD steps
        measure(tebd.pt.copy())
        print(f"chi={chi_max} Step {step+1} took {time.time()-t0:.2f}s, max_bond={max(tebd.pt.bond_sizes())}")

        res = []
        for step_idx in range(len(n_up_res)):
            res.append({
                'n_up': n_up_res[step_idx], # site 15
                'n_dn': n_dn_res[step_idx],
                'nn': nn_res[step_idx]
            })

        with open(f"/tmp/fh-231/tebd_res_chi{chi_max}.pkl", "wb") as f:
            pickle.dump(res, f)
"""

content = content.replace(old_loop, new_loop)
with open('run_tebd.py', 'w') as f:
    f.write(content)
