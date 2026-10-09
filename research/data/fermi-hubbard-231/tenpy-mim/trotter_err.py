import sys, numpy as np, json
sys.path.insert(0,'/tmp/fh-231-t')
import exact_circuit as ec
L=int(sys.argv[1]); n=30
times=[0.2*k for k in range(1,n+1)]
tr=ec.literal_circuit(L,n)
co=ec.continuous(L,times)
res={}
for site in (L//2-1, L//2):
    res[site]={'tr_nu':[float(x[0][site]) for x in tr],'co_nu':[float(x[0][site]) for x in co],
               'tr_dd':[float(x[2][site]) for x in tr],'co_dd':[float(x[2][site]) for x in co]}
json.dump(res,open(f'/tmp/fh-231-t/val/trotter_vs_cont_L{L}.json','w'))
for site in res:
    r=res[site]
    for k in (5,10,15,20,25,30):
        print(L,site,k, 'nu: trotter %.5f cont %.5f diff %+.5f | dd: trotter %.5f cont %.5f diff %+.5f'%(r['tr_nu'][k-1],r['co_nu'][k-1],r['tr_nu'][k-1]-r['co_nu'][k-1],r['tr_dd'][k-1],r['co_dd'][k-1],r['tr_dd'][k-1]-r['co_dd'][k-1]))
