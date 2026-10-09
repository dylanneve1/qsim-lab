import sys, numpy as np, json
sys.path.insert(0,'/tmp/fh-231-t')
import exact_circuit as ec, tebd_fh as tf
import os
out={}
for L in (4,6,8):
    lit=ec.literal_circuit(L,30); dire=ec.direct_pauli(L,30)
    rec=tf.run(L,512,30,svd_min=1e-16,outdir='/tmp/fh-231-t/val',tag='_val',quiet=True)
    dm=0; dc=0; dN=0
    for k,r in enumerate(rec):
        nu,nd,dd=lit[k]
        d=max(np.abs(np.array(r['nu'])-nu).max(),np.abs(np.array(r['nd'])-nd).max(),np.abs(np.array(r['dd'])-dd).max())
        dm=max(dm,d)
        dN=max(dN,abs(sum(r['nu'])-L/2),abs(sum(r['nd'])-L/2))
    print(f'L={L}: max |TEBD - literal fSWAP circuit| over 30 steps, all sites, nu/nd/dd = {dm:.2e};  max|N_up-L/2|,|N_dn-L/2| dev = {dN:.2e}; terr={rec[-1]["terr"]:.1e}; chimax={max(r["chimax"] for r in rec)}',flush=True)
    out[L]=dm
json.dump(out,open('/tmp/fh-231-t/val/validate.json','w'))
