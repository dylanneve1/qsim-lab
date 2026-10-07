import sys, numpy as np, quimb.tensor as qtn, cotengra as ctg, math, time, pickle, argparse
from tnexact_build import build_ops
from gen import blocks, truncate_A, build_W
D='qat/data/observable-estimations/circuit-models/operator_loschmidt_echo/'
FAM={'49':D+'49Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm','70':D+'70Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm'}
ap=argparse.ArgumentParser(); ap.add_argument('fam'); ap.add_argument('L',type=int)
ap.add_argument('--target',type=float,default=25); ap.add_argument('--repeats',type=int,default=128)
ap.add_argument('--time',type=float,default=900); ap.add_argument('--methods',default='kahypar')
ap.add_argument('--contract',action='store_true'); ap.add_argument('--nov',action='store_true'); ap.add_argument('--vkeep',default=None); ap.add_argument('--dtype',default='complex128')
a=ap.parse_args()
A6,V,S,_=blocks(FAM[a.fam])
if a.nov: V=[]
if a.vkeep:
    VK=set(int(x) for x in a.vkeep.split(',')); V=[o for o in V if not (o[0]=='rz' and o[1][0] not in VK)]
W=build_W(truncate_A(A6,a.L),V,S)
tn,n=build_ops(W,[52,59,72],a.dtype)
if tn.num_tensors==1:
    val=complex(tn.tensors[0].data)*10**tn.exponent/2**n; print(f'RESULT fam={a.fam} L={a.L} f={val.real:.12f} imag={val.imag:.2e} (fully simplified)'); sys.exit()
print('fam',a.fam,'L',a.L,'n',n,'tensors',tn.num_tensors,'exponent',tn.exponent,flush=True)
tag=f'{a.fam}_L{a.L}'+('_nov' if a.nov else '')+('_vk%d'%(abs(hash(a.vkeep))%1000) if a.vkeep else '')
try:
    tree=pickle.load(open(f'tree_{tag}.pkl','rb')); print('loaded tree',flush=True)
    if tree.contraction_width()>a.target:
        tree=tree.slice_and_reconfigure(target_size=2**a.target); print('resliced log2flops',math.log2(tree.contraction_cost()),'width',tree.contraction_width(),'nslices',tree.nslices,flush=True)
except FileNotFoundError:
    opt=ctg.HyperOptimizer(methods=a.methods.split(','),max_repeats=a.repeats,max_time=a.time,minimize='flops',parallel=False,progbar=False)
    t=time.time(); tree=tn.contraction_tree(optimize=opt)
    print('search',round(time.time()-t),'log2flops',math.log2(tree.contraction_cost()),'width',tree.contraction_width(),'nslices',tree.nslices,flush=True)
    if tree.contraction_width()>a.target:
        tree=tree.slice_and_reconfigure(target_size=2**a.target)
        print('sliced log2flops',math.log2(tree.contraction_cost()),'width',tree.contraction_width(),'nslices',tree.nslices,flush=True)
    pickle.dump(tree,open(f'tree_{tag}.pkl','wb'))
if a.contract:
    t=time.time(); arrays=tn.arrays; tot=0
    for i in range(tree.nslices):
        tot+=tree.contract_slice(arrays,i)
        if i%max(1,tree.nslices//20)==0: print(' slice',i,'/',tree.nslices,round(time.time()-t),flush=True)
    val=complex(tot)*10**tn.exponent/2**n
    print(f'RESULT fam={a.fam} L={a.L} f={val.real:.12f} imag={val.imag:.2e} time={time.time()-t:.0f}',flush=True)
