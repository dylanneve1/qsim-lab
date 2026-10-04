def uv(x,y): return ((x-2*y)//4, y)
def xy(u,v): return (2*v+4*u, v)
def show(L,S,mark=()):
    # S set of (u,v)
    for v in range(L,-1,-1):
        row=" "*v
        for u in range(0,L-v+1):
            if (u-v)%3==2: ch="rgb"[[1,2,0][v%3]] if False else "GBR"[v%3].lower()
            else: ch="#" if (u,v) in S else ("*" if (u,v) in mark else ".")
            row+=ch+" "
        print("%2d "%v+row)
