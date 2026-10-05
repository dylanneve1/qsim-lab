import json
spec=[('cd-w6a-0.jsonl',0,180),('cd-w6a-1.jsonl',0,180),('cd-w6b-0.jsonl',180,216),('cd-w6b-1.jsonl',180,216),
      ('cd-w6c-0.jsonl',216,228),('cd-w6c-1.jsonl',216,234),('cd-w6d-0.jsonl',228,280),('cd-w6-0.jsonl',280,10**9),('cd-w6d-1.jsonl',234,294),('cd-w6-1.jsonl',294,10**9)]
seen=set(); n=0
with open('cd-all-w6.jsonl','w') as g:
    for f,lo,hi in spec:
        for l in open(f):
            r=json.loads(l)
            if lo<=r['n']<hi:
                key=(r['n'],r['l'],r['m'],r['A'],r['B'])
                if key in seen: continue
                seen.add(key); g.write(l); n+=1
print(n)
