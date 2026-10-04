#!/usr/bin/env python3
"""Train the sparse-token decoder on an endless FastSampler stream.

usage: train.py <prefix-for-geometry> <out-dir> [--train-stim a.stim,b.stim] [--steps N] [--batch B]
                [--H 128] [--L 4] [--lr 1e-3] [--tmax 64] [--val <val.ptb64>] [--seed S] [--resume ckpt]
Training stims must share <prefix>'s detector layout (same code/schedule, other p). Mixed p:
each step draws its 1024-shot blocks round-robin from the listed stims."""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from nd_common import *
from nd_model import Decoder

ap = argparse.ArgumentParser()
ap.add_argument("prefix"); ap.add_argument("out")
ap.add_argument("--train-stim", default=None)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--H", type=int, default=128); ap.add_argument("--L", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tmax", type=int, default=64)
ap.add_argument("--val", default=None)
ap.add_argument("--val-shots", type=int, default=1 << 17)
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--resume", default=None)
ap.add_argument("--eval-every", type=int, default=2000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
meta = load_meta(a.prefix)
nd = meta.shape[0]
stims = (a.train_stim or a.prefix + ".stim").split(",")
streams = [Stream(s, nd, a.seed + 17 * i) for i, s in enumerate(stims)]
mx.random.seed(a.seed)
model = Decoder(meta, H=a.H, L=a.L, heads=a.heads)
if a.resume:
    model.load_weights(a.resume)
nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
WU = min(1000, a.steps // 10)
sched = optim.join_schedules([optim.linear_schedule(1e-6, a.lr, WU),
                              optim.cosine_decay(a.lr, a.steps - WU, a.lr * 0.02)], [WU])
opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)


def loss_fn(m, tok, y):
    return nn.losses.binary_cross_entropy(m(tok), y, with_logits=True, reduction="mean")


lg = nn.value_and_grad(model, loss_fn)


def step(tok, y):
    loss, g = lg(model, tok, y)
    g, _ = optim.clip_grad_norm(g, 1.0)
    opt.update(model, g)
    return loss


def batch_tokens(dets, tmax):
    tok, cnt = tokens(dets, tmax, nd)
    T = int(min(tmax, max(8, -(-cnt.max() // 8) * 8)))
    return tok[:, :T], cnt


def predict(m, dets, bs=4096):
    """logits for every shot; shots sorted by weight so T stays small; no shot dropped"""
    cnt = dets.sum(1)
    order = np.argsort(cnt, kind="stable")
    out = np.empty(len(cnt), np.float32)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        T = max(8, -(-int(cnt[idx].max()) // 8) * 8)
        tok, _ = tokens(dets[idx], T, nd)
        out[idx] = np.array(m(mx.array(tok)))
    return out


val = None
if a.val:
    vb = read_ptb64(a.val, nd + 1)[:a.val_shots]
    val = (vb[:, :nd], vb[:, nd])
log = open(os.path.join(a.out, "log.jsonl"), "a")
t0 = time.time(); paused = 0.0; seen = 0; dropped = 0; run = []
blocks_per = a.batch // 1024
for it in range(1, a.steps + 1):
    paused += wait_lock([s.p for s in streams])
    s = streams[it % len(streams)]
    dets, obs = s.read(blocks_per)
    tok, cnt = batch_tokens(dets, a.tmax)
    dropped += int((cnt > a.tmax).sum())
    keep = cnt <= a.tmax
    if not keep.all():
        tok, obs = tok[keep], obs[keep]
    loss = step(mx.array(tok), mx.array(obs.astype(np.float32)))
    mx.eval(model.parameters(), opt.state, loss)
    run.append(loss.item()); seen += len(obs)
    if it % 200 == 0:
        print(f"it {it} loss {np.mean(run):.5f} shots {seen} {seen / (time.time() - t0 - paused):.0f}/s "
              f"lr {sched(opt.step).item():.2e} dropped {dropped} paused {paused:.0f}s", flush=True)
        run = []
    if (it % a.eval_every == 0 or it == a.steps):
        rec = dict(it=it, shots=seen, train_s=round(time.time() - t0 - paused, 1), params=int(nparams))
        if val is not None:
            pr = predict(model, val[0]) > 0
            f = int((pr != val[1].astype(bool)).sum())
            rec.update(val_fails=f, val_shots=len(pr), val_pL=f / len(pr))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        model.save_weights(os.path.join(a.out, "model.safetensors"))
        json.dump(dict(H=a.H, L=a.L, heads=a.heads, prefix=a.prefix, stims=stims, it=it, shots=seen),
                  open(os.path.join(a.out, "cfg.json"), "w"))
for s in streams:
    s.close()
