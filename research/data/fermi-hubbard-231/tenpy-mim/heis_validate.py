import sys, numpy as np
sys.path.insert(0,'/tmp/fh-231-t')
import exact_circuit as ec, heis
for L,c,chi,ks in ((4,1,256,(1,2,3,7,12,30)),(6,2,600,(1,2,5,12,20))):
    lit=ec.literal_circuit(L,30)
    for obs,ix in (('nu',0),('nd',1),('dd',2)):
        worst=0
        for k in ks:
            rec=heis.run(L,c,obs,k,chi,svd_min=1e-14,quiet=True)
            v=rec[-1]['val']; ex=lit[k-1][ix][c]
            worst=max(worst,abs(v-ex))
        print(f'L={L} c={c} chi<={chi} {obs}: max |Heisenberg-MPO - exact literal circuit| over k in {ks}: {worst:.2e}; chimax {rec[-1]["chimax"]} Smax {rec[-1]["Smax"]:.2f} eps {rec[-1]["eps"]:.1e}',flush=True)
