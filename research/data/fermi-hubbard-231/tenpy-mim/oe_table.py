import json
rows={}
for chi in (128,256,512):
    rows[chi]=json.load(open(f'heis/heis_L60_c29_nu_k30_chi{chi}.json'))['rec']
f=json.load(open('runs/tebd_L60_chi3072.json'))['rec']
f2=json.load(open('runs/tebd_L60_chi2048.json'))['rec']
print('| j (layers = time t=0.2 j) | S_op(chi=128) | S_op(chi=256) | S_op(chi=512) | cum. discarded wt chi=256 / 512 | forward-state S (chi=3072 for j<=20, 2048 beyond) |')
print('|---|---|---|---|---|---|')
for j in (3,5,6,9,10,12,15,18,20,21,24,27,30):
    fs=(f[j-1]['Smax'] if j<=20 else f2[j-1]['Smax'])
    print(f"| {j} (t={0.2*j:.1f}) | {rows[128][j-1]['Smax']:.3f} | {rows[256][j-1]['Smax']:.3f} | {rows[512][j-1]['Smax']:.3f} | {rows[256][j-1]['eps']:.1e} / {rows[512][j-1]['eps']:.1e} | {fs:.2f} |")
