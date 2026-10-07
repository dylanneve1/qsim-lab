import re, math, json, numpy as np
def num(x): return float(eval(x.strip(), {"pi": math.pi, "__builtins__": {}}))
def load(path='/tmp/peaked-gen/portal/P6_titan_pinnacle.qasm'):
    G=[]
    for line in open(path):
        line=line.strip()
        m=re.match(r'(\w+)(\(([^)]*)\))?\s+(.*);',line)
        if not m or m.group(1) in ('qreg','creg','include','OPENQASM','measure','barrier'): continue
        name=m.group(1); ps=[num(p) for p in m.group(3).split(',')] if m.group(3) else []
        qs=[int(x) for x in re.findall(r'q\[(\d+)\]',m.group(4))]
        G.append((name,tuple(qs),tuple(ps)))
    return G
_r=json.load(open('/tmp/peaked-p5p6/P6_rings.json')); rings=[_r[0],_r[2],_r[1]]  # A, B (twin of A), C (22)
ringof={}; pos={}
for r,R in enumerate(rings):
    for i,q in enumerate(R): ringof[q]=r; pos[q]=i
twin={int(k):v for k,v in json.load(open('/tmp/peaked-p5p6/P6_twin.json')).items()}
def cls(a,b):
    if ringof[a]!=ringof[b]: return 'X'
    L=len(rings[ringof[a]]); d=(pos[a]-pos[b])%L
    return 'N' if d in (1,L-1) else 'L'
