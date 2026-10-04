import sys,math
rows=[]
for f in sys.argv[1:]:
    for l in open(f):
        if l.startswith('#'): continue
        j,y,ms,ok=l.split()[:4]; rows.append((y,int(ms),int(ok)))
M=len(rows); vc=sum(ms>840 for _,ms,_ in rows); okunc=sum(ok and ms<=840 for _,ms,ok in rows)
p=okunc/M; q=vc/M
print(f"M={M} S_lo={p:.4f}±{math.sqrt(p*(1-p)/M):.4f} vcap={q:.4f}±{math.sqrt(q*(1-q)/M):.4f}")
