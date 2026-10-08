"""Final table: per-step n_f and stag_SCV vs chi, best estimate (largest chi), error bar =
max(|v(chi_max)-v(chi_prev)|, |terr-linear extrapolation - v(chi_max)|); g=0 calibration vs exact free."""
import json,os,numpy as np
G=json.load(open('/tmp/su2-254-win/work/gauss.json'))['free']
fS=np.array([r['stag'] for r in G['SCV']]); fM=np.array([r['stag'] for r in G['meson']])
sg=(-1.0)**np.arange(60)
def load(c,chi,lam):
    f=f'tebd_{c}_chi{chi}_lam{lam:.1f}_0_60.json'
    if not os.path.exists(f): return None
    R=json.load(open(f))['rec']
    if len(R)<20: return None
    n=np.array([np.array(r['ni'])+np.array(r['no']) for r in R]); return n@sg,np.array([r['terr'] for r in R])
chis=[c for c in (256,512,1024,2048,2800,4096) if load('SCV',c,1.0) and load('meson',c,1.0)]
S={c:load('SCV',c,1.0) for c in chis}; M={c:load('meson',c,1.0) for c in chis}
out=[]
print('chis',chis)
hdr='step | '+' | '.join(f'nf chi{c}' for c in chis)+' | nf best ± err | '+' | '.join(f'stag chi{c}' for c in chis)+' | stag best ± err | terr_SCV(chimax)'
print(hdr)
for k in range(20):
    def est(vals,terrs):
        v=np.array(vals); t=np.array(terrs)
        e1=abs(v[-1]-v[-2])
        # linear extrapolation in truncation error using last two chis
        if t[-2]>t[-1]: ex=v[-1]-(v[-2]-v[-1])/(t[-2]-t[-1])*t[-1]
        else: ex=v[-1]
        return v[-1],max(e1,abs(ex-v[-1])),ex
    nf=[M[c][0][k]-S[c][0][k] for c in chis]; tn=[S[c][1][k]+M[c][1][k] for c in chis]
    st=[S[c][0][k] for c in chis]; ts=[S[c][1][k] for c in chis]
    nb,ne,nx=est(nf,tn); sb,se,sx=est(st,ts)
    out.append(dict(step=k+1,n_f=nb,n_f_err=ne,stag_SCV=sb,stag_SCV_err=se,n_f_chis=dict(zip(chis,nf)),stag_chis=dict(zip(chis,st)),terr_SCV=ts[-1]))
    print(f'{k+1:4d} | '+' | '.join(f'{x:+.6f}' for x in nf)+f' | {nb:+.6f} ± {ne:.1e} | '+' | '.join(f'{x:+.6f}' for x in st)+f' | {sb:+.6f} ± {se:.1e} | {ts[-1]:.1e}')
json.dump(out,open('final.json','w'),indent=1)
print('\n=== g=0 calibration: |TEBD - exact free| at step 16,18,20')
for c in (256,512,1024,2048,2800):
    a=load('SCV',c,0.0); b=load('meson',c,0.0)
    if a is None or b is None: continue
    print(f'chi={c}: stag err',' '.join(f'{abs(a[0][k]-fS[k]):.1e}' for k in (15,17,19)),' n_f err',' '.join(f'{abs(b[0][k]-a[0][k]-(fM[k]-fS[k])):.1e}' for k in (15,17,19)),' terr(20) SCV %.1e'%a[1][19])
