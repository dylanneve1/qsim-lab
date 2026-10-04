OFF=[(-2,1),(2,1),(4,0),(2,-1),(-2,-1),(-4,0)]
def code(d):
    L=3*(d-1)//2; data=[]; pl=[]
    for y in range(L+1):
        color,pos={0:(1,2),1:(2,0),2:(0,1)}[y%3]
        x=2*y
        while x<=4*L-2*y:
            if ((x//2-y)//2)%3!=pos: data.append((x,y))
            else: pl.append((x,y,color))
            x+=4
    idx={c:i for i,c in enumerate(data)}
    P=[]
    for (x,y,c) in pl:
        P.append(dict(x=x,y=y,c=c,q=[idx.get((x+dx,y+dy)) for dx,dy in OFF]))
    return data,P
if __name__=="__main__":
    import sys
    d=int(sys.argv[1]); data,P=code(d)
    L=3*(d-1)//2
    grid={}
    for i,(x,y) in enumerate(data): grid[(x,y)]="%3d"%i
    for p in P: grid[(p['x'],p['y'])]=" "+"RGB"[p['c']]+str(sum(q is not None for q in p['q']))
    for y in range(L,-1,-1):
        print("".join(grid.get((x,y),"   ") if x%2==0 else "" for x in range(0,4*L+1)))
    for i,p in enumerate(P): print(i,p['x'],p['y'],"RGB"[p['c']],[q for q in p['q'] if q is not None])
