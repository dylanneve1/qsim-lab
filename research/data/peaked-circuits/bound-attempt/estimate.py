"""NON-RIGOROUS extrapolations from the certified central portion of the outer zipper.
Not bounds. Assumptions are stated in each line."""
import numpy as np
P={'P11_Hqap_98x1999':(0.3029,1999-59-224-1, 'O'),'P12_Hqap_98x2457':(0.2366,2457-60-217-2,'O')}
for n,(p,nmid,lab) in P.items():
    cs=np.load(f'costs2_{n}_{lab}_k9_t0.05.npy'); nf=np.load(f'nfs2_{n}_{lab}_k9_t0.05.npy')
    m=cs<0.05; steps={'P11_Hqap_98x1999':200,'P12_Hqap_98x2457':140}[n]
    rate=cs[m].sum()/steps; s=np.sqrt(p)
    lin=rate*nmid
    rss=np.sqrt((cs[m]**2).sum()*nmid/steps); nfr=np.sqrt((nf[m]**2).sum()*nmid/steps)
    print(f'{n}: p_model {p}, sqrt {s:.4f}; middle gates {nmid}; certified central gates {steps}')
    print(f'  lossy op-norm rate {rate:.2e}/gate -> linear extrapolation (what a complete telescoping bound would cost even with NO structural residuals) {lin:.2f}  => vacuous' if lin>=s else f'  linear {lin:.2f}')
    print(f'  incoherent (RSS) extrapolation of op-norm peel costs: {rss:.3f} -> p >~ {(s-rss)**2:.3f}   [heuristic: assumes errors add in quadrature]')
    print(f'  incoherent average-case (normalised Frobenius) extrapolation: {nfr:.3f} -> p ~ {(s-nfr)**2:.3f}   [heuristic: Haar-typical states + quadrature]')
