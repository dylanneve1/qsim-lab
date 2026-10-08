import json,numpy as np
for L in (12,14):
    lo=30-L//2
    for c in ('SCV','meson'):
        try: w=np.load(f'win_L{L}_{c}.npy')
        except: continue
        d=json.load(open(f'tebd_{c}_chi2048_lam1.0_0_60.json'))['rec']
        out=[]
        for k in range(1,9):
            t=np.array(d[k-1]['ni'])+np.array(d[k-1]['no'])
            ctr=range(30-2,30+2)   # central 4 rungs 28..31
            out.append(max(abs(t[r]-w[k-1,r-lo].sum()) for r in ctr))
        print(f'L={L} {c}: max|TEBD(full,chi2048) - exact window| on rungs 28-31, steps 1-8:',' '.join(f'{x:.1e}' for x in out))
