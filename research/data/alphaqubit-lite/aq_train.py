#!/usr/bin/env python3
"""Pretrain / fine-tune / evaluate AlphaQubit-lite on Sycamore 2022 data (MLX, Mac GPU under caps).

  pretrain : FastSampler samples of the training-half pij DEMs (aq_gen.py sim/), all areas and both
             bases of one distance in one model (canonical layout + a learned context embedding).
             Model selection on the real dev split (below) every --eval-every steps.
  finetune : real shots of the training half: per experiment the first 19,880 shots are training data,
             the remaining 5,120 the dev set (paper's split); weight decay towards the pretrained
             weights (--wd-anchor).
  eval     : the held-out half (25,000 shots per experiment), LER fitted over R = 3..25 per (area,
             basis), mean over datasets; per-shot fail vectors saved for paired comparisons.

Guards (swarm rules): MLX memory/cache caps (aq_model), refuse to start with free+inactive < 4 GB,
pause below 3 GB, stop if wired memory > 4 GB, pause while a peer holds the Mac bench lock, hard
wall-clock cap (--max-minutes, default 45).

usage: aq_train.py <data-dir> <syc-root> <out-dir> --mode pretrain|finetune|eval [--d 3] [--fold odd]
       [--steps N] [--batch 256] [--lr 5e-4] [--init ckpt-dir] [model size flags]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neural-decoder"))
from nd_common import wait_lock, wait_memory, mem_report, mem_available_gb
from aq_data import *

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--mode", default="pretrain")
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--fold", default="odd")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--wd-anchor", type=float, default=0.0)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--aux", type=float, default=0.02)
ap.add_argument("--ema", type=float, default=1e-3)
ap.add_argument("--rounds", default=",".join(str(r) for r in range(3, 26, 2)))
ap.add_argument("--scales", default="1.0")
ap.add_argument("--init", default=None)
ap.add_argument("--init-partial", default=None, help="checkpoint dir of another distance: load every shape-matching weight")
ap.add_argument("--eval-every", type=int, default=1000)
ap.add_argument("--dev-shots", type=int, default=5120)
ap.add_argument("--dev-max", type=int, default=0, help="evaluate model selection on the first N dev shots per experiment")
ap.add_argument("--max-minutes", type=float, default=45.0)
ap.add_argument("--eval-at-start", action="store_true")
ap.add_argument("--max-peak-mb", type=float, default=1750.0, help="stop if MLX peak + cache limit would pass ~2 GB")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--checkpoint", type=int, default=1)
ap.add_argument("--compile", type=int, default=1)
# paper's noise curriculum (Methods eq. 6-7): sample DEM scale f with p_f(t) ~ 1 + w_c G(f_c(t), sigma_c; f),
# f_c(t) = f_min + (1 - f_min) / (1 + exp(-s_c (t / t_c - 1))), t = training examples seen
ap.add_argument("--curr-tc", type=float, default=0, help="t_c in examples (0 = no noise curriculum)")
ap.add_argument("--curr-wc", type=float, default=12.0)
ap.add_argument("--curr-sigma", type=float, default=0.05)
ap.add_argument("--curr-fmin", type=float, default=0.0)
ap.add_argument("--curr-sc", type=float, default=1.0)
# rounds curriculum (paper, Pauli+ runs): max R grows linearly from --rmax0 to max(rounds) over --rcurr examples
ap.add_argument("--rmax0", type=int, default=0)
ap.add_argument("--rcurr", type=float, default=0)
ap.add_argument("--eval-split", default="test")
ap.add_argument("--eval-bs", type=int, default=512)
ap.add_argument("--test-max", type=int, default=0, help="first N held-out shots per experiment (0 = all)")
ap.add_argument("--weights", default="model.safetensors")
ap.add_argument("--eval-rounds", default=None, help="round counts for dev/test (default: --rounds)")
ap.add_argument("--source", default="pij", help="pretraining sample source tag (aq_gen --source)")
ap.add_argument("--n-ctx", type=int, default=32)
# model size
for k, v in dict(D=96, L=3, heads=4, key=24, widen=4, conv=48, bias_dim=24, bias_layers=2,
                 readout_dim=32, readout_layers=4).items():
    ap.add_argument(f"--{k}", type=int, default=v)
ap.add_argument("--dils", default=None)
ap.add_argument("--no-bias", action="store_true")
ap.add_argument("--no-indicators", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.mode != "eval" and mem_available_gb() < 4.0:
    raise SystemExit(f"refusing to start: free+inactive {mem_available_gb():.1f} GB < 4 GB")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from aq_model import AlphaQubitLite

cfg = vars(a).copy()
if a.init:  # architecture comes from the checkpoint
    old = json.load(open(os.path.join(a.init, "cfg.json")))
    for k in ("D", "L", "heads", "key", "widen", "conv", "bias_dim", "bias_layers", "readout_dim",
              "readout_layers", "dils", "no_bias", "no_indicators", "d", "n_ctx"):
        cfg[k] = old[k]
d = cfg["d"]
dils = tuple(int(x) for x in cfg["dils"].split(",")) if cfg["dils"] else ((1, 1, 1) if d == 3 else (1, 1, 2))
cfg["dils"] = ",".join(map(str, dils))
rounds = [int(x) for x in a.rounds.split(",")]
rng = np.random.default_rng(a.seed)
mx.random.seed(a.seed)

# ------------------------------------------------------------------ data
ap_eval_rounds = [int(x) for x in a.eval_rounds.split(",")] if a.eval_rounds else rounds
exps = [e for e in experiments(a.root) if e["d"] == d and (e["R"] in rounds or e["R"] in ap_eval_rounds)]
ctxs = sorted({(e["area"], e["basis"]) for e in exps})
CTX = {c: i for i, c in enumerate(ctxs)}
LAY, NDET = {}, {}
SIM, TRAIN, DEV, TEST = {}, {}, {}, {}
for e in exps:
    txt = open(os.path.join(e["path"], "circuit_ideal.stim")).read()
    L = Layout(txt, d, e["R"])
    key = (CTX[(e["area"], e["basis"])], e["R"])
    LAY[key] = L; NDET[key] = L.nd
    assert len(ctxs) <= a.n_ctx
    z = np.load(os.path.join(a.data, "real", e["name"] + ".npz"))
    rows = z["rows"]
    tr_idx = np.arange(0 if a.fold == "odd" else 1, len(rows), 2)   # training half
    te_idx = np.arange(1 if a.fold == "odd" else 0, len(rows), 2)   # held-out half
    TRAIN[key] = rows[tr_idx[:len(tr_idx) - a.dev_shots]]
    DEV[key] = rows[tr_idx[len(tr_idx) - a.dev_shots:]]
    TEST[key] = (rows[te_idx], te_idx)
    if a.mode == "pretrain" and e["R"] in rounds:
        for sc in a.scales.split(","):
            SIM.setdefault(float(sc), {})[key] = np.load(os.path.join(a.data, "sim", f"{e['name']}.{a.source}.s{sc}.npy"))
    if e["R"] not in rounds:
        TRAIN.pop(key)
    if e["R"] not in ap_eval_rounds:
        DEV.pop(key); TEST.pop(key)
L0 = next(iter(LAY.values()))
print(f"d={d} contexts={ctxs} rounds={rounds} train-half={'even' if a.fold == 'odd' else 'odd'}", flush=True)


def unpack(rows, key):
    nd = NDET[key]
    bits = np.unpackbits(rows, axis=1)[:, :nd + 1]
    return to_grid(bits[:, :nd], LAY[key]), bits[:, nd]


def make_batch(pool, R, B):
    keys = [k for k in pool if k[1] == R]
    cnt = rng.multinomial(B, np.ones(len(keys)) / len(keys))
    ev, ob, cx = [], [], []
    for k, c in zip(keys, cnt):
        if c == 0:
            continue
        rows = pool[k]
        g, o = unpack(rows[rng.integers(0, len(rows), c)], k)
        ev.append(g); ob.append(o); cx.append(np.full(c, k[0]))
    ev = np.concatenate(ev).astype(np.float32)
    return ev, np.concatenate(ob).astype(np.float32), np.concatenate(cx).astype(np.int32)


def tensors(ev, R, cx):
    return (mx.array(ev[:, :R]), mx.array(ev[:, R]), mx.array(cx), mx.array(np.full(len(cx), R, np.int32)))


# ------------------------------------------------------------------ model
model = AlphaQubitLite(L0.cell, L0.onbasis, d, D=cfg["D"], L=cfg["L"], heads=cfg["heads"], key=cfg["key"],
                       widen=cfg["widen"], conv=cfg["conv"], dils=dils, bias_dim=cfg["bias_dim"],
                       bias_layers=cfg["bias_layers"], readout_dim=cfg["readout_dim"],
                       readout_layers=cfg["readout_layers"], use_bias=not cfg["no_bias"],
                       indicators=not cfg["no_indicators"], aux=True, n_ctx=cfg.get("n_ctx", 32),
                       max_rounds=64)
if a.init:
    model.load_weights(os.path.join(a.init, a.weights if a.mode == "eval" else "model.safetensors"))
if a.init_partial:
    from mlx.utils import tree_unflatten
    src = mx.load(os.path.join(a.init_partial, "model.safetensors"))
    own = dict(tree_flatten(model.parameters()))
    take = [(k, v) for k, v in src.items() if k in own and own[k].shape == v.shape]
    model.update(tree_unflatten(take))
    print(f"partial init: {len(take)}/{len(own)} tensors from {a.init_partial} "
          f"(skipped: {sorted(k for k in own if k not in dict(take))})", flush=True)
nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
cfg["params"] = int(nparams)
json.dump(cfg, open(os.path.join(a.out, "cfg.json"), "w"), indent=1)
print(f"params {nparams}", flush=True)


def predict(m, rows_by_key, bs=None):
    bs = bs or a.eval_bs
    """logits per key (line-0 logit)"""
    out = {}
    for k, rows in rows_by_key.items():
        R = k[1]
        res = []
        for i in range(0, len(rows), bs):
            ev, ob = unpack(rows[i:i + bs], k)
            lo, _ = m(*tensors(ev.astype(np.float32), R, np.full(len(ob), k[0], np.int32)))
            res.append(np.array(lo[:, 0]))
            wait_lock()
        out[k] = np.concatenate(res)
    return out


def ler_table(m, split):
    """split: dict key -> rows. returns mean LER over contexts, per-ctx details, fail vectors"""
    lg = predict(m, split)
    fails = {}
    for k, rows in split.items():
        nd = NDET[k]
        obs = np.unpackbits(rows, axis=1)[:, nd]
        fails[k] = (lg[k] > 0) != obs.astype(bool)
    per = {}
    for c in range(len(ctxs)):
        rs = sorted(R for (cc, R) in fails if cc == c)
        if len(rs) < 2:
            continue
        e_, F0, r2 = fit_ler(rs, [fails[(c, R)].sum() for R in rs], [len(fails[(c, R)]) for R in rs])
        per[c] = dict(eps=e_, F0=F0, R2=r2)
    return float(np.mean([v["eps"] for v in per.values()])), per, fails


if a.mode == "eval":
    split = {k: (v[0][:a.test_max] if a.test_max else v[0]) for k, v in TEST.items()} if a.eval_split == "test" else DEV
    t0 = time.time()
    ler, per, fails = ler_table(model, split)
    os.makedirs(os.path.join(a.out, "fails"), exist_ok=True)
    for k, f in fails.items():
        area, basis = ctxs[k[0]]
        e = [x for x in exps if x["area"] == area and x["basis"] == basis and x["R"] == k[1]][0]
        name = f"{e['name']}.{'odd' if a.fold == 'odd' else 'even'}.nn.npy"
        np.save(os.path.join(a.out, "fails", name), f)
    byR = {}
    for (c, R), f in fails.items():
        byR.setdefault(R, []).append(eps_from_E(f.mean(), R))
    rec = dict(split=a.eval_split, ler=ler, eps_by_round={R: float(np.mean(v)) for R, v in sorted(byR.items())}, per={f"{ctxs[c][0]}{ctxs[c][1]}": v for c, v in per.items()},
               shots=int(sum(len(f) for f in fails.values())), secs=round(time.time() - t0, 1), **mem_report())
    print(json.dumps(rec), flush=True)
    print(f"eval {rec['shots']} shots in {rec['secs']} s", flush=True)
    json.dump(rec, open(os.path.join(a.out, f"eval_{a.eval_split}.json"), "w"), indent=1)
    sys.exit(0)

# ------------------------------------------------------------------ training
SCALES = sorted(SIM) if a.mode == "pretrain" else [1.0]
SIMVAL = {}
if a.mode == "pretrain":  # last 1024 shots of every full-noise source: held-out simulated dev set
    top = SCALES[-1]
    SIMVAL = {k: v[-1024:][:a.dev_max or 1024] for k, v in SIM[top].items()}
    SIM[top] = {k: v[:-1024] for k, v in SIM[top].items()}


def pick_pool(seen):
    if a.mode != "pretrain":
        return TRAIN
    if a.curr_tc <= 0 or len(SCALES) == 1:
        return SIM[SCALES[-1]] if len(SCALES) == 1 else SIM[SCALES[rng.integers(len(SCALES))]]
    fc = a.curr_fmin + (1 - a.curr_fmin) / (1 + np.exp(-a.curr_sc * (seen / a.curr_tc - 1)))
    w = np.array([1 + a.curr_wc * np.exp(-0.5 * ((f - fc) / a.curr_sigma) ** 2) for f in SCALES])
    return SIM[SCALES[rng.choice(len(SCALES), p=w / w.sum())]]


def pick_rounds(seen):
    if a.rmax0 <= 0 or a.rcurr <= 0:
        return int(rng.choice(rounds))
    rmax = a.rmax0 + (max(rounds) - a.rmax0) * min(1.0, seen / a.rcurr)
    ok = [r for r in rounds if r <= rmax] or [min(rounds)]
    return int(rng.choice(ok))
anchor = tree_map(lambda p: mx.array(p), model.parameters()) if a.wd_anchor > 0 else None
WU = min(a.warmup, a.steps // 10)
sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, WU),
                              optim.cosine_decay(a.lr, a.steps - WU, a.lr * 0.05)], [WU])
opt = optim.AdamW(learning_rate=sched, betas=[0.9, 0.95], weight_decay=a.wd)
ema = tree_map(lambda p: mx.array(p), model.parameters())


def loss_fn(m, ev, fe, cx, R, y):
    lo, aux = m(ev, fe, cx, R, checkpoint=bool(a.checkpoint))
    l = nn.losses.binary_cross_entropy(lo[:, 0], y, with_logits=True, reduction="mean")
    meas = mx.cumsum(ev, axis=1) % 2
    la = nn.losses.binary_cross_entropy(aux, meas[:, 1:], with_logits=True, reduction="mean")
    return l + a.aux * la, l


lg = nn.value_and_grad(model, loss_fn)


def train_step(ev, fe, cx, R, y):
    (loss, lmain), g = lg(model, ev, fe, cx, R, y)
    g, _ = optim.clip_grad_norm(g, 1.0)
    opt.update(model, g)
    return lmain


if a.compile:  # one trace per round count R (static shapes); fuses the many small kernels
    from functools import partial
    state = [model.state, opt.state]
    train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)
log = open(os.path.join(a.out, "log.jsonl"), "a")
log.write(json.dumps(dict(cfg=cfg)) + "\n")
BEST = [9.0]
t0 = time.time(); paused = 0.0; seen = 0; run = []


def evaluate(it):
    dev = {k: v[:a.dev_max] for k, v in DEV.items()} if a.dev_max else DEV
    cur = model.parameters()
    ler_raw, _, _ = ler_table(model, dev)
    model.update(ema)
    ler, per, _ = ler_table(model, dev)
    model.update(cur)
    sim_ler = None
    if SIMVAL:
        model.update(ema)
        sim_ler, _, _ = ler_table(model, SIMVAL)
        model.update(cur)
    rec = dict(it=it, shots=seen, dev_ler=ler, dev_ler_raw=ler_raw, sim_dev_ler=sim_ler, train_s=round(time.time() - t0 - paused, 1),
               wall_s=round(time.time() - t0, 1), **mem_report())
    rec["best"] = min(ler, ler_raw) < BEST[0]
    if rec["best"]:
        BEST[0] = min(ler, ler_raw)
        cur = model.parameters()
        if ler <= ler_raw:
            model.update(ema)
        model.save_weights(os.path.join(a.out, "model.safetensors"))
        model.update(cur)
    model.save_weights(os.path.join(a.out, "last.safetensors"))
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n"); log.flush()


if a.eval_at_start:
    evaluate(0)
for it in range(1, a.steps + 1):
    paused += wait_lock()
    if it % 50 == 0:
        paused += wait_memory(3.0)
        w = mem_report().get("wired_gb")
        if mx.get_peak_memory() / 2**20 > a.max_peak_mb:
            model.save_weights(os.path.join(a.out, "last.safetensors"))
            raise SystemExit(f"MLX peak {mx.get_peak_memory() / 2**20:.0f} MB > {a.max_peak_mb} MB at it {it}: stopping")
        if w is not None and w > 4.0:
            model.save_weights(os.path.join(a.out, "last.safetensors"))
            raise SystemExit(f"wired memory {w} GB > 4 GB at it {it}: stopping")
    R = pick_rounds(seen)
    ev, y, cx = make_batch(pick_pool(seen), R, a.batch)
    lmain = train_step(*tensors(ev, R, cx), mx.array(y))
    if anchor is not None:  # decoupled weight decay towards the pretrained weights (fine-tuning)
        lr = sched(opt.step)
        model.update(tree_map(lambda p, p0: p - lr * a.wd_anchor * (p - p0), model.parameters(), anchor))
    er = max(a.ema, 1.0 / (1.0 + 0.1 * it))  # EMA warm-up: horizon ~ it/10 steps until it reaches 1/ema
    ema = tree_map(lambda e, p: e + er * (p - e), ema, model.parameters())
    mx.eval(model.parameters(), opt.state, ema, lmain)
    run.append(lmain.item()); seen += len(y)
    if it % 100 == 0:
        el = time.time() - t0 - paused
        print(f"it {it} loss {np.mean(run):.5f} shots {seen} {seen / el:.0f}/s lr {sched(opt.step).item():.2e} "
              f"paused {paused:.0f}s {json.dumps(mem_report())}", flush=True)
        run = []
    over = time.time() - t0 > 60 * a.max_minutes
    if it % a.eval_every == 0 or it == a.steps or over:
        evaluate(it)
    if over:
        print(f"wall-clock cap {a.max_minutes} min reached at it {it}", flush=True)
        break
