import json,numpy as np
print('symmetry diagnostics (exact dynamics obeys n_up(i)=n_dn(59-i) and n_up(i)+n_dn(i)=1 at half filling; violations measure truncation damage)')
print('| chi | t | max_i |n_up(i)-n_dn(59-i)| | |n_up+n_dn-1| at site 29 |')
print('|---|---|---|---|')
for c in (512,1024,2048,3072):
    r=json.load(open(f'runs/tebd_L60_chi{c}.json'))['rec']
    for t in range(1,7):
        if len(r)<5*t: continue
        x=r[5*t-1]; nu=np.array(x['nu']); nd=np.array(x['nd'])
        print(f'| {c} | {t} | {np.abs(nu-nd[::-1]).max():.1e} | {abs(nu[29]+nd[29]-1):.1e} |')
