"""Exact amplitude <s|C|0> of the RAW P6 circuit by tensor-network contraction (quimb + cotengra), for verification.
usage: tn_amp.py BITSFILE [max_time_s] [flip_qubit]"""
import sys, time, numpy as np, quimb.tensor as qtn, cotengra as ctg
s=open(sys.argv[1]).read().strip(); T=float(sys.argv[2]) if len(sys.argv)>2 else 60
if len(sys.argv)>3:
    q=int(sys.argv[3]); s=s[:q]+('1' if s[q]=='0' else '0')+s[q+1:]
circ=qtn.Circuit.from_openqasm2_file('/tmp/peaked-gen/portal/P6_titan_pinnacle.qasm')
tn=circ.amplitude_tn(s, simplify_sequence='ADCRS')
print('tensors',tn.num_tensors,'inds',tn.num_indices,flush=True)
opt=ctg.HyperOptimizer(methods=['kahypar','greedy'],max_time=T,max_repeats=10**6,minimize='combo',parallel=False,progbar=False)
tree=tn.contraction_tree(opt)
print('width %.1f  log10 flops %.2f'%(tree.contraction_width(),__import__('math').log10(float(tree.contraction_cost()))),flush=True)
if tree.contraction_width()<=27:
    t0=time.time(); a=tn.contract(all,optimize=tree); print('amp',a,'P',abs(a)**2,'t',time.time()-t0)
