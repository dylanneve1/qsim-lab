import json,glob,numpy as np,os
G=json.load(open('/tmp/su2-254-win/work/gauss.json'))['free']
free={c:np.array([r['stag'] for r in G[c]]) for c in ('SCV','meson')}
freen={c:np.array([np.array(r['n']).sum(1) for r in G[c]]) for c in ('SCV','meson')}  # (20,60)
sg=(-1.0)**np.arange(60)
def load(c,chi,lam):
    f=f'tebd_{c}_chi{chi}_lam{lam:.1f}_0_60.json'
    if not os.path.exists(f): return None
    d=json.load(open(f)); R=d['rec']
    n=np.array([np.array(r['ni'])+np.array(r['no']) for r in R]); return n,np.array([r['terr'] for r in R]),np.array([r['Smax'] for r in R])
chis=sorted({int(f.split('chi')[1].split('_')[0]) for f in glob.glob('tebd_*_0_60.json')})
print('chis',chis)
for lam in (0.0,1.0):
    print(f'=== lam={lam}')
    for chi in chis:
        a=load('SCV',chi,lam); b=load('meson',chi,lam)
        if a is None: continue
        k=len(a[0]); sS=a[0]@sg
        line=f'chi={chi:5d} SCV  stag: '+' '.join(f'{x:9.5f}' for x in sS[9::2])
        if lam==0: line+='\n            |err| vs exact free: '+' '.join(f'{x:9.1e}' for x in np.abs(sS-free["SCV"][:k])[9::2])
        print(line, ' terr(20)=%.1e'%a[1][-1])
        if b is not None:
            kk=min(k,len(b[0])); nf=(b[0][:kk]-a[0][:kk])@sg
            line=f'           n_f : '+' '.join(f'{x:9.5f}' for x in nf[9::2])
            if lam==0: line+='\n            |err| vs exact free: '+' '.join(f'{x:9.1e}' for x in np.abs(nf-(free["meson"]-free["SCV"])[:kk])[9::2])
            print(line)
print('steps shown: 10,12,...,20')
