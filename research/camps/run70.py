import sys, time, json, resource, numpy as np
from camps import *
mode, d, chi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ops = load_circuit(n=70, d=d)
nt = sum(1 for o in ops if o[0].startswith('rz'))
t0 = time.time()
if mode == 'mps':
    m = run_mps(ops, 70, chi); st = None
else:
    st = run_camps(ops, 70, chi, disentangle=(mode == 'camps'), ofd=(mode != 'camps-noOFD'))
    m = st.mps
# fidelity loss by CZ-layer
byl = {}
for tag, w in m.disc:
    byl[tag] = byl.get(tag, 0.0) + (-np.log1p(-w))
ent = m.entropies()
res = dict(mode=mode, d=d, chi=chi, T=nt, fid=m.fid, logfid=float(np.log(max(m.fid,1e-300))) if m.fid > 0 else None,
           sumloss=sum(byl.values()), maxbond=m.max_bond(), time=time.time() - t0,
           rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           Smax=max(ent), Smid=ent[34], loss_by_layer={int(k): round(v, 5) for k, v in sorted(byl.items())})
if st is not None:
    res.update(ofd=st.stats['ofd'], mpo=st.stats['mpo'], trivial=st.stats['trivial'], dis=st.stats['dis_applied'],
               support_med=float(np.median([s for s in st.stats['support']])) if st.stats['support'] else 0)
print(json.dumps(res), flush=True)
