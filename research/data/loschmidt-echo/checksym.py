import sys, math
from parse import parse
def perq(ops):
    d={}
    for o in ops:
        for q in o[1]: d.setdefault(q,[]).append((o[0],o[1],None if o[2] is None else round(o[2],9)))
    return d
def inv(ops):
    return [(n,qs,None if v is None else -v) for n,qs,v in reversed(ops)]
