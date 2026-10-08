import numpy as np, os
os.environ['SU2_CIRCUITS']='/tmp/su2-254-win/circuits'
import pt2
L=10; lo=25; hi=35
lams=[0,.25,-.25,.5,-.5,.75,-.75,1,-1,1.5,-1.5,2,-2]
def ld(c,l): return np.load(f'res/L{L}_{c}_lam{l:+.2f}.npy')
eps=np.array([(-1.0)**(lo+i) for i in range(L)])
def obs(c,l): return (ld(c,l).sum(2)*eps).sum(1)   # stag over window, per step
def nf(l): return obs('meson',l)-obs('SCV',l)
ob={'stag_SCV':lambda l:obs('SCV',l),'n_f':nf}
lam=np.array(lams)
def fit(y,order):
    A=np.vander(lam,order+1,increasing=True); return np.linalg.lstsq(A,y,rcond=None)[0]
# 7 pts set from brief
s7=[0,.25,-.25,.5,-.5,1,-1]; i7=[lams.index(x) for x in s7]
D={c:pt2.collect(c,lo,hi,20) for c in ('SCV','meson')}
for name,f in ob.items():
    Y=np.array([f(l) for l in lams])   # (nlam,20)
    print('=====',name,'L=10 window (sites 25..34)')
    print('step | a0 | a1 | a2 (7pt fit) | a3 | a4 | a5 | a6 | a2 PT2 | a1 PT2 | |a2| |a4| |a6| terms at lam=1: | E(1) E(-1) | a8,a10 (13pt fit)')
    for t in range(20):
        A7=np.vander(np.array(s7),7,increasing=True); c7=np.linalg.solve(A7,Y[i7,t])
        c13=fit(Y[:,t],12)
        # pt2
        def pc(c): 
            d,v,w=D[c]; return np.array(pt2.pt_coeffs(d,v,w,eps,t+1))
        p=pc('meson')-pc('SCV') if name=='n_f' else pc('SCV')
        print(f'{t+1:2d} {c7[0]:+10.5f} a1={c7[1]:+.2e} a2={c7[2]:+.4e} a3={c7[3]:+.2e} a4={c7[4]:+.3e} a5={c7[5]:+.2e} a6={c7[6]:+.3e} | pt2 a2={p[2]:+.4e} a1={p[1]:+.2e} | E1={Y[lams.index(1),t]:+.5f} Em1={Y[lams.index(-1),t]:+.5f} | 13pt: a2={c13[2]:+.4e} a4={c13[4]:+.3e} a6={c13[6]:+.3e} a8={c13[8]:+.2e} a10={c13[10]:+.2e} a12={c13[12]:+.1e}')
