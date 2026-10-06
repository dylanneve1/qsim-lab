import sys, time, numpy as np
sys.path.insert(0,'/tmp/peaked-generic')
import solve_peaked_v2 as v2, solve_anchor as SA
f='/tmp/pk/research/data/peaked-circuits/peaked_circuit_P11_Hqap_98x1999.qasm'
old = v2.Core(f); t=time.time(); e = old.evaluate(); print("old initial p", round(e['p'],4), round(time.time()-t,1), flush=True)
new = SA.Core(f)
ops = new.ops(frozenset(), frozenset())
kinds = {}
for _,_,k in ops: kinds[k[0]] = kinds.get(k[0],0)+1
print("new op kinds", kinds)
orig = new.ops
def ops_noA(er, ep):
    return [o for o in orig(er, ep) if o[2][0] != 'A']
new.ops = ops_noA; new.marg = {}
e2 = new.evaluate(); print("new without A segments p", round(e2['p'],4), flush=True)
