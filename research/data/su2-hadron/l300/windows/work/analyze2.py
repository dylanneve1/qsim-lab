import numpy as np, json
ld=lambda L,c,l: np.load(f'res/L{L}_{c}_lam{l:+.2f}.npy')
sites=lambda L: list(range(30-L//2,30-L//2+L))
G=json.load(open('gauss.json'))['free']
nfull={c:np.array([r['n'] for r in G[c]]) for c in ('SCV','meson')}   # (20,60,2)
Ls=(6,8,10,12)
cen6=list(range(27,33)); cen2=[29,30]
sg=lambda sel: np.array([(-1)**r for r in sel])
def pick(a,L,sel): s=sites(L); return a[:,[s.index(r) for r in sel],:]
def nfsum(a_m,a_s,sel,L=None):
    L=L or globals()["L"]; return ((pick(a_m,L,sel)-pick(a_s,L,sel)).sum(2)*sg(sel)).sum(1)
out={}
print('Per-site free full-system vs free window (hard wall) n_f c6 and SCV-c6')
for L in Ls:
    fw=nfsum(ld(L,'meson',0),ld(L,'SCV',0),cen6,L)
    ff=(((nfull['meson']-nfull['SCV'])[:,cen6,:].sum(2))*sg(cen6)).sum(1)
    print(f'L={L} max|free_window-free_full| n_f(c6) per step:',' '.join(f'{x:.0e}' for x in abs(fw-ff)))
print()
# correction Delta = E(1)-E(0) per site-density
D={}
for L in Ls:
    D[L]={c:ld(L,c,1.0)-ld(L,c,0.0) for c in ('SCV','meson')}
for name,sel in (('c6',cen6),('c2',cen2)):
    print(f'=== interaction shift Delta=E(1)-E(0) of n_f over sites {sel}')
    tab=np.array([nfsum(D[L]['meson'],D[L]['SCV'],sel,L) for L in Ls]); out['dnf_'+name]=tab.tolist()
    print('step '+' '.join(f'L={L:>2}'.rjust(11) for L in Ls)+'  '+' '.join(f'd{Ls[i]}-{Ls[i+1]}'.rjust(10) for i in range(3)))
    for t in range(20): print(f'{t+1:4d} '+' '.join(f'{v:11.6f}' for v in tab[:,t])+'  '+' '.join(f'{abs(tab[i,t]-tab[i+1,t]):10.1e}' for i in range(3)))
# per-site density shifts: converged-through
print('=== Delta densities (E(1)-E(0)), max over ctr-6 sites/chains and both circuits: |L - L+2|')
for i in range(3):
    for c in ('SCV','meson'):
        d=np.abs(pick(D[Ls[i]][c],Ls[i],cen6)-pick(D[Ls[i+1]][c],Ls[i+1],cen6)).reshape(20,-1).max(1)
        bad=np.where(d>1e-4)[0]
        print(f'L={Ls[i]}->{Ls[i+1]} {c}: converged (<1e-4) through step {bad[0] if len(bad) else 20}; per-step:',' '.join(f'{x:.0e}' for x in d))
print('=== raw densities E(1) ctr-6 max |L-(L+2)| (12 values) first step >1e-4:')
for i in range(3):
    for c in ('SCV','meson'):
        d=np.abs(pick(ld(Ls[i],c,1.0),Ls[i],cen6)-pick(ld(Ls[i+1],c,1.0),Ls[i+1],cen6)).reshape(20,-1).max(1)
        bad=np.where(d>1e-4)[0]; print(f'L={Ls[i]}->{Ls[i+1]} {c}: through step {bad[0] if len(bad) else 20}; per-step:',' '.join(f'{x:.0e}' for x in d))
# centre-site (29,30) raw densities by L
print('=== centre-site densities n_i+n_o at sites 29,30 (SCV, meson), lam=1 by L')
for c in ('SCV','meson'):
  for r in (29,30):
    print(c,'site',r)
    for t in (1,2,3,4,5,6,8,10,12,15,20):
        print(f'  step {t:2d} '+' '.join(f'L{L}:{pick(ld(L,c,1.0),L,[r])[t-1].sum():.6f}' for L in Ls))
# Estimate for full-system: free_full + Delta(L=12) per site, nf c6 and total free
print('=== free-full + Delta_L estimate of n_f(c6) and free_full values')
ff=(((nfull['meson']-nfull['SCV'])[:,cen6,:].sum(2))*sg(cen6)).sum(1)
for t in range(20): print(f'{t+1:3d} free_full c6 {ff[t]:+.6f} | +Delta L=8 {ff[t]+out["dnf_c6"][1][t]:+.6f} L=10 {ff[t]+out["dnf_c6"][2][t]:+.6f} L=12 {ff[t]+out["dnf_c6"][3][t]:+.6f} | raw win E1 L=12 {nfsum(ld(12,"meson",1.0),ld(12,"SCV",1.0),cen6,12)[t]:+.6f}')
json.dump(out,open('win2.json','w'))
