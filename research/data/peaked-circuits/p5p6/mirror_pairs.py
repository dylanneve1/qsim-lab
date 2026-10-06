import numpy as np,sys
B=np.load(sys.argv[1]); L=B[:,2].max()+1
S=[set(map(tuple,np.sort(B[B[:,2]==l][:,:2],axis=1))) for l in range(L)]
M=np.array([[len(S[i]&S[j]) for j in range(L)] for i in range(L)])
np.set_printoptions(linewidth=250)
for i in range(L):
    row=''.join('.' if M[i,j]==0 else (str(M[i,j]) if M[i,j]<10 else '#') for j in range(L))
    print(f'{i:3d} {len(S[i]):3d} {row}')
