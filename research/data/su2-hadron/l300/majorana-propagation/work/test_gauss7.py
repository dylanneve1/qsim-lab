import gauss
rec_scv, _, _ = gauss.run('SCV', mode='free')
rec_mes, _, _ = gauss.run('meson', mode='free')
print("Step 20 SCV free:", rec_scv[19]['stag'])
print("Step 20 Mes free:", rec_mes[19]['stag'])
print("n_f:", rec_mes[19]['stag'] - rec_scv[19]['stag'])
