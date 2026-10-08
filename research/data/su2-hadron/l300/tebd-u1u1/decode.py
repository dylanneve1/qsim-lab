"""Decode circuit into a compact op list per circuit: ('h',chain,s1,s2,Vsp) hop on adjacent sites (2x2 single-particle unitary,
basis (s1,s2)), ('p',chain,s,phase) single-site phase on occupied state, ('v',s,g) rung vertex exp(i g n_i n_o), ('step',k).
Global phases dropped."""
import os,pickle,numpy as np,functools
os.environ['SU2_CIRCUITS']='/tmp/su2-254-win/circuits'
import gauss
def decode(circ):
    occ,m,out=gauss.blocks(f'/tmp/su2-254-win/circuits/x_100_{circ}.qasm')
    n0=np.zeros((2,60),int)
    for w in range(120):
        s,l=m[w]; n0[l,s]=occ[w]
    ops=[]
    for b in out:
        if 'step' in b: ops.append(('step',b['step'])); continue
        w=b['w'];U=b['U'];sl=[m[x] for x in w]
        if len(w)==1:
            s,l=sl[0]; ops.append(('p',l,s,np.angle(U[1,1]/U[0,0])))
        elif sl[0][1]==sl[1][1]:
            (s1,l1),(s2,l2)=sl; V=U[1:3,1:3]/U[0,0]; Vsp=np.array([[V[1,1],V[1,0]],[V[0,1],V[0,0]]])
            ops.append(('h',l1,s1,s2,Vsp))
        else:
            (s1,l1),(s2,l2)=sl; ph=np.angle(np.diag(U)); a0=ph[2]-ph[0]; a1=ph[1]-ph[0]; g=ph[3]-ph[2]-ph[1]+ph[0]
            g=(g+np.pi)%(2*np.pi)-np.pi
            ops.append(('p',l1,s1,a0)); ops.append(('p',l2,s2,a1)); ops.append(('v',s1,g))
    return n0,ops
if __name__=='__main__':
    for c in ('SCV','meson'):
        n0,ops=decode(c); pickle.dump((n0,ops),open(f'ops_{c}.pkl','wb'))
        print(c,'n_i',''.join(map(str,n0[0])),'\n   n_o',''.join(map(str,n0[1])),len(ops))
