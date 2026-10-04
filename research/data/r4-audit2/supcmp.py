import csv,json,subprocess,glob
B='/tmp/qsim-wt/r4-audit2-target/release/examples/simulability'
seen={}
for g in ['ct24','ct32','ctnn0','brick','arith','qaoa']:
  for r in csv.DictReader(open(f'/tmp/qsim-wt/r4-audit2/research/data/simulability/raw/{g}.csv')):
    k=(r['spec'],r['seed'])
    if k in seen: continue
    f=json.loads(r['features']); 
    if f.get('g3',0)==0: seen[k]=None; continue
    out=subprocess.run(['nice','-n','15',B,'features',r['spec'],r['seed'],'nohsf'],capture_output=True,text=True).stdout
    nf=json.loads(out); seen[k]=(f['sup'],nf['sup'],f['n'])
ch=[(k,v) for k,v in seen.items() if v and v[0]!=v[1]]
tot=sum(1 for v in seen.values() if v)
print('instances with Toffolis',tot,'changed',len(ch))
for k,v in ch[:40]: print(k[0],'old',v[0],'new',v[1],'n',v[2])
