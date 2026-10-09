import json, pickle, glob, re, numpy as np
D='/tmp/fh-231/'
t1=pickle.load(open(D+'tdvp_data_1_pt.pkl','rb')); t2=pickle.load(open(D+'tdvp_data_2_pt.pkl','rb'))
hw1={n:pickle.load(open(D+f'{n}_data_1_pt.pkl','rb')) for n in ('raw_hw','mm','mm_and_dr')}
hw2={n:pickle.load(open(D+f'{n}_data_2_pt.pkl','rb')) for n in ('raw_hw','mm','mm_and_dr')}
runs={}
for f in glob.glob('/tmp/fh-231-t/runs/tebd_L60_chi*.json'):
    m=re.search(r'chi(\d+)\.json',f); runs[int(m.group(1))]=json.load(open(f))['rec']
chis=sorted(runs); top=chis[-1]; prev=chis[-2]
ntop=len(runs[top]); L=60
out=[]
P=out.append
def tdv(site,j,k):
    if k>=30: return np.nan
    if j==0: return t1[(site,'up')][k]
    if j==1: return t1[(site,'down')][k]
    return t2[(site,'down',site,'up')][k]
def hwv(n,site,j,k):
    if j==0: return hw1[n][(site,'up')][k]
    if j==1: return hw1[n][(site,'down')][k]
    return hw2[n][(site,'down',site,'up')][k]
def te(c,site,j,k):
    if k>len(runs[c]): return np.nan
    r=runs[c][k-1]; return [r['nu'],r['nd'],r['dd']][j][site]
names=['<n_up>','<n_dn>','<n_up n_dn>']
P(f'TEBD chi values available: {chis}; steps completed: { {c:len(runs[c]) for c in chis} }\n')
for site in (29,30):
    P(f'## Centre site i={site} ({"initially up" if site%2 else "initially down"})\n')
    for j in range(3):
        P(f'### {names[j]}, site {site}\n')
        P('| t (step) | '+' | '.join(f'chi={c}' for c in chis)+f' | err bar = |best chi - next chi| | TDVP paper (cont. time) | HW mm+dr | HW mm | HW raw | HW(mm+dr)-TEBD(best) | TDVP-TEBD(best) |')
        P('|'+'---|'*(len(chis)+8))
        for t in range(1,7):
            k=5*t; v=[te(c,site,j,k) for c in chis]; av=[c for c in chis if len(runs[c])>=k]; best=te(av[-1],site,j,k); eb=abs(best-te(av[-2],site,j,k))
            td=tdv(site,j,k); h=hwv('mm_and_dr',site,j,k)
            P(f'| {t} ({k}) | '+' | '.join(('—' if np.isnan(x) else f'{x:.6f}') for x in v)+f' | {eb:.1e} (chi {av[-2]}→{av[-1]}) | {td:.6f} | {h:.6f} | {hwv("mm",site,j,k):.6f} | {hwv("raw_hw",site,j,k):.6f} | {h-best:+.4f} | {td-best:+.4f} |')
        P('')
# all-site RMSE and convergence
P('## All-site 1-point observables (120 values n_{i,sigma}), per step\n')
P('conv = max/rms over all 120 observables of |best chi - next chi| (best chi = 3072 for t<=4, 2048 for t=5,6); RMSE vs best TEBD of hardware (mm+dr, mm, raw) and TDVP.\n')
P('| t (step) | conv max | conv rms | RMSE HW mm+dr | RMSE HW mm | RMSE HW raw | RMSE TDVP |')
P('|---|---|---|---|---|---|---|')
for t in range(1,7):
    k=5*t
    av=[c for c in chis if len(runs[c])>=k]; a=runs[av[-1]][k-1]; b=runs[av[-2]][k-1]
    A=np.array(a['nu']+a['nd']); B=np.array(b['nu']+b['nd'])
    keys=[(i,'up') for i in range(L)]+[(i,'down') for i in range(L)]
    def rm(src): return float(np.sqrt(np.mean((np.array([src[kk][k] for kk in keys])-A)**2)))
    td=rm(t1) if k<30 else np.nan
    P(f'| {t} ({k}) | {np.abs(A-B).max():.1e} | {np.sqrt(np.mean((A-B)**2)):.1e} | {rm(hw1["mm_and_dr"]):.4f} | {rm(hw1["mm"]):.4f} | {rm(hw1["raw_hw"]):.4f} | {td:.4f} |')
P('\n## Per-chi run statistics (centre-bond entropy max, max bond dim, sum of discarded weights, wall, RSS)\n')
P('| chi | steps | Smax@t=1..6 (nats) | sum discarded weight @t=1..6 | wall (s) | peak RSS GB |')
P('|---|---|---|---|---|---|')
for c in chis:
    r=runs[c]
    P(f"| {c} | {len(r)} | {[round(r[5*t-1]['Smax'],2) for t in range(1,7) if len(r)>=5*t]} | {[f'{r[5*t-1]['terr']:.1e}' for t in range(1,7) if len(r)>=5*t]} | {r[-1]['wall']:.0f} | {r[-1]['rss']:.2f} |")
open('/tmp/fh-231-t/results_tables.md','w').write('\n'.join(out)); print('\n'.join(out)[:6000])
