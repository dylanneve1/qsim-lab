import sys, quimb.tensor as qtn, cotengra as ctg, math, time, argparse
from tnexact_build import build_ops
from gen import blocks, truncate_A, build_W
import numpy as np
D='qat/data/observable-estimations/circuit-models/operator_loschmidt_echo/'
FAM={'49':D+'49Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm','70':D+'70Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm'}
ap=argparse.ArgumentParser(); ap.add_argument('fam'); ap.add_argument('L',type=int)
ap.add_argument('--repeats',type=int,default=64); ap.add_argument('--time',type=float,default=600)
ap.add_argument('--z',action='store_true'); ap.add_argument('--vkeep',default=None); ap.add_argument('--methods',default='kahypar,greedy')
a=ap.parse_args()
A6,V,S,_=blocks(FAM[a.fam])
if a.vkeep:
    VK=set(int(x) for x in a.vkeep.split(',')); V=[o for o in V if not (o[0]=='rz' and o[1][0] not in VK)]
W=build_W(truncate_A(A6,a.L),V,S)
z=None
if a.z:
    qs=sorted({q for o in W for q in o[1]}); rng=np.random.default_rng(1); z={q:int(rng.integers(2)) for q in qs}
tn,n=build_ops(W,[52,59,72],z=z)
print('fam',a.fam,'L',a.L,'z' if a.z else 'trace','tensors',tn.num_tensors,flush=True)
opt=ctg.HyperOptimizer(methods=a.methods.split(','),max_repeats=a.repeats,max_time=a.time,minimize='flops',parallel=False,progbar=False)
t=time.time(); tree=tn.contraction_tree(optimize=opt)
print('COST fam',a.fam,'L',a.L,'z' if a.z else 'trace','log2flops %.2f width %.1f search %.0f'%(math.log2(tree.contraction_cost()),tree.contraction_width(),time.time()-t),flush=True)
