import json,pickle
t1=pickle.load(open('/tmp/fh-231/tdvp_data_1_pt.pkl','rb'));t2=pickle.load(open('/tmp/fh-231/tdvp_data_2_pt.pkl','rb'))
r2={dt:json.load(open(f'runs/dtscan/tebd_L60_chi256_dt{dt}.json'))['rec'] for dt in ('0.1','0.05')}
r02=json.load(open('runs/tebd_L60_chi2048.json'))['rec']
print('Trotter-step scan at centre site 29, chi=256 (t<=2 converged in chi), dt in {0.2,0.1,0.05}; Richardson (dt^2) of the dt=0.1,0.05 pair vs ITensor-TDVP (continuous time)')
for T,k2 in ((1,5),(2,10)):
    for key,lab in (('nu','n_up'),('dd','n_up n_dn')):
        a=r02[k2-1][key][29]; b=r2['0.1'][int(round(T/0.1))-1][key][29]; c=r2['0.05'][int(round(T/0.05))-1][key][29]
        rich=c+(c-b)/3
        td=(t1[(29,'up')][k2] if key=='nu' else t2[(29,'down',29,'up')][k2])
        print(f't={T} {lab}: dt0.2={a:.5f} dt0.1={b:.5f} dt0.05={c:.5f} Richardson={rich:.5f} TDVP={td:.5f} (TDVP-Rich={td-rich:+.1e})')
