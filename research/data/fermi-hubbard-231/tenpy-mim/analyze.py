import json, pickle, glob, sys, numpy as np, re
D='/tmp/fh-231/'
tdvp1=pickle.load(open(D+'tdvp_data_1_pt.pkl','rb')); tdvp2=pickle.load(open(D+'tdvp_data_2_pt.pkl','rb'))
hw={n:(pickle.load(open(D+f'{n}_data_1_pt.pkl','rb')),pickle.load(open(D+f'{n}_data_2_pt.pkl','rb'))) for n in ('raw_hw','mm','mm_and_dr')}
runs={}
for f in glob.glob('/tmp/fh-231-t/runs/tebd_L60_chi*.json'):
    m=re.search(r'chi(\d+)\.json',f)
    if m: runs[int(m.group(1))]=json.load(open(f))['rec']
chis=sorted(runs)
SITE=int(sys.argv[1]) if len(sys.argv)>1 else 29
def tdvp(k):
    if k>=30: return (np.nan,)*3
    return (tdvp1[(SITE,'up')][k],tdvp1[(SITE,'down')][k],tdvp2[(SITE,'down',SITE,'up')][k])
def hwv(n,k):
    a,b=hw[n]; return (a[(SITE,'up')][k],a[(SITE,'down')][k],b[(SITE,'down',SITE,'up')][k])
names=['n_up','n_dn','n_up n_dn']
def te(chi,k):
    if k>len(runs[chi]): return (np.nan,)*3
    r=runs[chi][k-1]; return (r['nu'][SITE],r['nd'][SITE],r['dd'][SITE])
print(f'site {SITE}; chis available {chis}; steps done',{c:len(runs[c]) for c in chis})
for j in range(3):
    print(f'\n### {names[j]} (site {SITE})')
    print('| t | step | '+' | '.join(f'TEBD chi={c}' for c in chis)+' | spread(top2) | TDVP(paper) | HW mm | HW mm+dr | HW raw |')
    for t in range(1,7):
        k=5*t
        vals=[te(c,k)[j] for c in chis]
        sp=abs(vals[-1]-vals[-2]) if len(vals)>1 else np.nan
        print(f'| {t} | {k} | '+' | '.join(f'{v:.6f}' for v in vals)+f' | {sp:.1e} | {tdvp(k)[j]:.6f} | {hwv("mm",k)[j]:.6f} | {hwv("mm_and_dr",k)[j]:.6f} | {hwv("raw_hw",k)[j]:.6f} |')
print('\nentropy/discarded weight:')
for c in chis:
    r=runs[c]; print(c,'Smax@t=1..6',[round(r[5*t-1]['Smax'],3) for t in range(1,7) if len(r)>=5*t],'chimax',max(x['chimax'] for x in r),'terr@t',[f"{r[5*t-1]['terr']:.1e}" for t in range(1,7) if len(r)>=5*t],'wall',round(r[-1]['wall']),'rss',round(r[-1]['rss'],2))
