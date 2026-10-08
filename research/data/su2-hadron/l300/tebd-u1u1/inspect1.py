import pickle,numpy as np
n0,ops=pickle.load(open('ops_SCV.pkl','rb'))
# step 1 ops
k=0; seq=[]
for o in ops:
    if o[0]=='step': break
    seq.append(o)
print(len(seq))
from collections import Counter
print(Counter(o[0] for o in seq))
# order summary
print([ (o[0],o[1],o[2]) if o[0]!='v' else ('v',o[1]) for o in seq[:30]])
# total phase per site per step on each chain
ph=np.zeros((2,60))
for o in seq:
    if o[0]=='p': ph[o[1],o[2]]+=o[3]
print('phase chain0',np.round(ph[0,:8],5)); print('phase chain1',np.round(ph[1,:8],5))
hs=[o for o in seq if o[0]=='h']
print('hop example',np.round(hs[0][4],5), np.round(hs[1][4],5))
print('vertex g', set(round(o[2],7) for o in seq if o[0]=='v'))
