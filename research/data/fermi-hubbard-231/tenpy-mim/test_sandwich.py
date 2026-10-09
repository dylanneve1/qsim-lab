import sys, numpy as np
sys.path.insert(0,'/tmp/fh-231-t')
import exact_circuit as ec, tebd2 as tf, heis2 as hs, sandwich as sw
L,c=6,2; lit=ec.literal_circuit(L,30)
k0s=(3,5)
for k0 in k0s:
    rec,kept=tf.run(L,512,k0,svd_min=1e-14,quiet=True,outdir='/tmp/fh-231-t/val',tag='_sw',keep=(k0,))
    psi=kept[k0]
    for obs,ix in (('nu',0),('dd',2)):
        for k in (k0,k0+1,k0+3,k0+6):
            if k>30: continue
            if k==k0:
                # no heisenberg layers: identity conjugation -> just evaluate observable on psi
                ex=lit[k-1][ix][c]; v=rec[-1][{'nu':'nu','dd':'dd'}[obs]][c]; print(f'k0={k0} k={k} {obs}: forward only diff {abs(v-ex):.2e}'); continue
            r,Om,ln=hs.run(L,c,obs,k,3000,svd_min=1e-14,quiet=True,k0=k0,ret=True)
            v=sw.value(psi,Om,ln); ex=lit[k-1][ix][c]
            print(f'k0={k0} k={k} {obs}: sandwich {v:.10f} exact {ex:.10f} diff {abs(v-ex):.2e}  chiO={r[-1]["chimax"]}',flush=True)
