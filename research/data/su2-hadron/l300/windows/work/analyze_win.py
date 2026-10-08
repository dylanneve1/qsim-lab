import numpy as np, json, glob, os
res='res/'
def ld(L,c,lam): return np.load(f'{res}L{L}_{c}_lam{lam:+.2f}.npy')  # (20,L,2)
def sites(L): lo=30-L//2; return np.arange(lo,lo+L)
def dens_diff(L,lam=1.0):
    return ld(L,'meson',lam)-ld(L,'SCV',lam)   # (20,L,2)
def stagsum(arr,L,sel):
    s=sites(L); out=0
    for i,r in enumerate(s):
        if r in sel: out=out+(-1)**r*arr[:,i,:].sum(1)
    return out
Ls=[L for L in (6,8,10,12) if os.path.exists(f'{res}L{L}_meson_lam+1.00.npy')]
cen6=set(range(27,33)); cen2=set((29,30)); cen4=set(range(28,32))
out={}
print('== centre-site densities (sites 29,30; both chains) meson-SCV and SCV, by L, steps 1..20')
for sel,name in ((cen2,'c2'),(cen6,'c6')):
    print('--- nf contribution of sites',sorted(sel),'(sum (-1)^r (n_i+n_o)_meson - SCV)')
    tab=np.array([stagsum(dens_diff(L),L,sel) for L in Ls])  # (nL,20)
    out['nf_'+name]=tab.tolist()
    print('step '+' '.join(f'{"L="+str(L):>11}' for L in Ls)+'   '+' '.join(f'|d{Ls[i]}-{Ls[i+1]}|' .rjust(11) for i in range(len(Ls)-1)))
    for t in range(20):
        print(f'{t+1:4d} '+' '.join(f'{v:11.6f}' for v in tab[:,t])+'   '+' '.join(f'{abs(tab[i,t]-tab[i+1,t]):11.2e}' for i in range(len(Ls)-1)))
print('--- full-window n_f (sum over whole window)')
tabw=np.array([stagsum(dens_diff(L),L,set(sites(L))) for L in Ls]); out['nf_win']=tabw.tolist()
for t in range(20): print(f'{t+1:4d} '+' '.join(f'{v:11.6f}' for v in tabw[:,t]))
print('--- SCV stag per window and centre-6 SCV stag')
for L in Ls: pass
tabs=np.array([stagsum(ld(L,'SCV',1.0),L,cen6) for L in Ls]); out['scv_c6']=tabs.tolist()
print('SCV centre-6 stag'); 
for t in range(20): print(f'{t+1:4d} '+' '.join(f'{v:11.6f}' for v in tabs[:,t])+'   '+' '.join(f'{abs(tabs[i,t]-tabs[i+1,t]):11.2e}' for i in range(len(Ls)-1)))
# converged-up-to-step: max over per-site densities in central 6 sites (all 12 densities, both circuits)
print('== converged-up-to step (first step where max over ctr-6 per-site densities exceeds 1e-4, change L->L+2)')
for i in range(len(Ls)-1):
    for c in ('SCV','meson'):
        def cen(L):
            a=ld(L,c,1.0); s=sites(L); idx=[k for k,r in enumerate(s) if r in cen6]; return a[:,idx,:]
        d=np.abs(cen(Ls[i])-cen(Ls[i+1])).reshape(20,-1).max(1)
        bad=np.where(d>1e-4)[0]; cs=(bad[0] if len(bad) else 20)
        # also central 2
        def cen_(L,sel):
            a=ld(L,c,1.0); s=sites(L); idx=[k for k,r in enumerate(s) if r in sel]; return a[:,idx,:]
        d2=np.abs(cen_(Ls[i],cen2)-cen_(Ls[i+1],cen2)).reshape(20,-1).max(1)
        bad2=np.where(d2>1e-4)[0]; cs2=(bad2[0] if len(bad2) else 20)
        print(f'L={Ls[i]}->{Ls[i+1]} {c}: ctr6 converged through step {cs}; ctr2 through step {cs2}; maxdiff ctr6 per step:',' '.join(f'{x:.0e}' for x in d))
    tab=np.array([stagsum(dens_diff(L),L,cen6) for L in (Ls[i],Ls[i+1])]); d=abs(tab[0]-tab[1]); bad=np.where(d>1e-4)[0]
    print(f'   n_f(c6) L={Ls[i]}->{Ls[i+1]} converged through step {bad[0] if len(bad) else 20}')
json.dump(out,open('win_summary.json','w'))
