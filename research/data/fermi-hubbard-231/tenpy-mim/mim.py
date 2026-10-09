"""Meet in the middle: <psi(k0)| O_H(layers k0+1..k) |psi(k0)>.  usage: mim.py obs k0 k chiPsi chiO [c=29] [L=60]"""
import sys, os, json, time, pickle, resource, numpy as np
sys.path.insert(0, '/tmp/fh-231-t')
import tebd2 as tf, heis2 as hs, sandwich as sw

obs = sys.argv[1]; k0 = int(sys.argv[2]); k = int(sys.argv[3]); chiP = int(sys.argv[4]); chiO = int(sys.argv[5])
c = int(sys.argv[6]) if len(sys.argv) > 6 else 29; L = int(sys.argv[7]) if len(sys.argv) > 7 else 60
os.makedirs('/tmp/fh-231-t/mim', exist_ok=True)
cache = f'/tmp/fh-231-t/mim/psi_L{L}_k{k0}_chi{chiP}.pkl'
t0 = time.time()
if os.path.exists(cache):
    psi, epsP = pickle.load(open(cache, 'rb'))
else:
    rec, kept = tf.run(L, chiP, k0, svd_min=1e-10, quiet=True, outdir='/tmp/fh-231-t/mim', tag=f'_k0{k0}', keep=(k0,))
    psi = kept[k0]; epsP = rec[-1]['terr']
    pickle.dump((psi, epsP), open(cache, 'wb'))
tF = time.time() - t0
if k == k0:
    val = float(np.real(psi.expectation_value({'nu': 'Nu', 'nd': 'Nd', 'dd': 'Nud'}[obs])[c])); epsO = 0.; tH = tS = 0.; chimO = 0
else:
    t1 = time.time()
    r, Om, ln = hs.run(L, c, obs, k, chiO, svd_min=1e-10, quiet=True, k0=k0, ret=True)
    tH = time.time() - t1; epsO = r[-1]['eps']; chimO = r[-1]['chimax']
    t2 = time.time(); val = sw.value(psi, Om, ln); tS = time.time() - t2
res = dict(obs=obs, k0=k0, k=k, chiPsi=chiP, chiO=chiO, val=val, epsPsi=epsP, epsO=epsO, chimaxO=chimO, tF=tF, tH=tH, tS=tS, rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
print(json.dumps(res), flush=True)
json.dump(res, open(f'/tmp/fh-231-t/mim/mim_{obs}_k0{k0}_k{k}_P{chiP}_O{chiO}.json', 'w'))
