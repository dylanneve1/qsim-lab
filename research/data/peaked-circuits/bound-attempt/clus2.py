import sys, collections, numpy as np
sys.path.insert(0,'/tmp/peaked-bound')
from solve_peaked_v2 import Core
from chains import region_cost
D='/tmp/pk/research/data/peaked-circuits/'
c=Core(D+'peaked_circuit_P11_Hqap_98x1999.qasm')
for ks in [[644,651,676,693,698,719],[651,676,698,719],[644,651,676,693,698],[651,676,698],[676,698,719]]:
    print(ks, region_cost(c,ks))
