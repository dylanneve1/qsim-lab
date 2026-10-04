#!/usr/bin/env python3
"""Decode a held-out ptb64 file with a trained model; saves <out>.<name>.fails.npy + JSON line.
usage: nn_eval.py <model-dir> <test.ptb64> <rounds> <out> [name]"""
import sys, json, os, time
import numpy as np
import mlx.core as mx
from nd_common import *
from nd_model import Decoder

md, test, rounds, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
name = sys.argv[5] if len(sys.argv) > 5 else "nn"
cfg = json.load(open(os.path.join(md, "cfg.json")))
meta = load_meta(cfg["prefix"])
nd = meta.shape[0]
m = Decoder(meta, H=cfg["H"], L=cfg["L"], heads=cfg["heads"])
m.load_weights(os.path.join(md, "model.safetensors"))
bits = read_ptb64(test, nd + 1)
dets, obs = bits[:, :nd], bits[:, nd].astype(bool)
t0 = time.time()
cnt = dets.sum(1)
order = np.argsort(cnt, kind="stable")
pred = np.empty(len(cnt), bool)
for i in range(0, len(order), 8192):
    wait_lock()
    idx = order[i:i + 8192]
    T = max(8, -(-int(cnt[idx].max()) // 8) * 8)
    tok, _ = tokens(dets[idx], T, nd)
    pred[idx] = np.array(m(mx.array(tok))) > 0
fails = pred != obs
n = len(obs); f = int(fails.sum()); lo, hi = wilson(f, n)
r = dict(decoder=name, model=md, train_shots=cfg["shots"], shots=n, fails=f, p_L=f / n, ci95=[lo, hi],
         p_L_round=per_round(f / n, rounds), ci95_round=[per_round(lo, rounds), per_round(hi, rounds)],
         decode_s=round(time.time() - t0, 1))
np.save(f"{out}.{name}.fails.npy", fails)
print(json.dumps(r))
open(out + ".jsonl", "a").write(json.dumps(r) + "\n")
