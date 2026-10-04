"""Sparse fired-detector transformer decoder (MLX).

A shot is the set of fired detectors (typically 5-40 of 100-600). Each fired detector is a token:
learned per-detector embedding + MLP of its (x, y, t, type, colour) features. A CLS token is
prepended. Self-attention gets an additive bias per head and layer from an MLP of the pairwise
relative geometry (dx, dy, dt, |.|, both types), i.e. a learned, translation-aware 'matching
weight'. Readout: CLS -> logit of the observable flip."""
import math, os
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Dylan's laptop: hard caps on MLX memory (default 1.5 GB) and its buffer cache (256 MB)
mx.set_memory_limit(int(float(os.environ.get("ND_MEM_GB", "1.5")) * 2**30))
mx.set_cache_limit(256 * 2**20)


def det_features(meta):
    """meta columns: idx x y t is_x colour -> (nd, F) float32 normalised features"""
    x, y, t, isx, col = (meta[:, k].astype(np.float32) for k in (1, 2, 3, 4, 5))
    def nrm(v):
        return (v - v.min()) / max(1.0, v.max() - v.min()) * 2 - 1
    f = np.stack([nrm(x), nrm(y), nrm(t), isx * 2 - 1, (col == 0) * 1.0, (col == 1) * 1.0, (col == 2) * 1.0], 1)
    return f.astype(np.float32)


def grid_geom(meta):
    """integer lattice coordinates (x, y, t, is_x): x and y divided by their smallest step"""
    g = meta[:, 1:5].astype(np.int64)
    for k in (0, 1):
        u = np.unique(g[:, k])
        st = np.diff(u).min() if len(u) > 1 else 1
        g[:, k] = (g[:, k] - u.min()) // max(1, st)
    return g


R_XY, R_T = 10, 10  # relative-offset window of the attention-bias table (clipped beyond)


class Block(nn.Module):
    def __init__(self, H, heads, mlp=4):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(H)
        self.qkv = nn.Linear(H, 3 * H)
        self.o = nn.Linear(H, H)
        self.ln2 = nn.LayerNorm(H)
        self.f1 = nn.Linear(H, mlp * H)
        self.f2 = nn.Linear(mlp * H, H)

    def __call__(self, x, bias):
        # x (B,T,H); bias (B,heads,T,T) incl. key mask
        B, T, H = x.shape
        h = self.ln1(x)
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        sh = lambda a: a.reshape(B, T, self.heads, H // self.heads).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(H // self.heads), mask=bias)
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, H))
        return x + self.f2(nn.gelu(self.f1(self.ln2(x))))


class Decoder(nn.Module):
    def __init__(self, meta, H=128, L=4, heads=4, readout="cls"):
        super().__init__()
        self.readout = readout
        nd = meta.shape[0]
        self.nd, self.L, self.heads = nd, L, heads
        self.feat = mx.array(np.concatenate([det_features(meta), np.zeros((2, 7), np.float32)]))  # pad, cls
        g = grid_geom(meta)
        self.geom = mx.array(np.concatenate([g, np.zeros((2, 4), np.int64)]).astype(np.int32))
        nx, nt = 2 * R_XY + 1, 2 * R_T + 1
        self.nrel = nx * nx * nt * 4
        self.emb = nn.Embedding(nd + 2, H)
        self.fmlp = nn.Sequential(nn.Linear(7, H), nn.GELU(), nn.Linear(H, H))
        # per layer: learned bias per head for every (dx, dy, dt, type_i, type_j) offset, + 1 CLS slot
        self.rel = [nn.Embedding(self.nrel + 1, heads) for _ in range(L)]
        for r in self.rel:
            r.weight = mx.zeros_like(r.weight)
        self.blocks = [Block(H, heads) for _ in range(L)]
        self.lnf = nn.LayerNorm(H)
        self.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1))

    def __call__(self, tok):
        # tok (B,T) int32 fired detector ids, pad = nd
        B, T = tok.shape
        cls = mx.full((B, 1), self.nd + 1, dtype=mx.int32)
        tok = mx.concatenate([cls, tok], axis=1)
        T1 = T + 1
        x = self.emb(tok) + self.fmlp(self.feat[tok])
        g = self.geom[tok]  # (B,T1,4) int
        nx, nt = 2 * R_XY + 1, 2 * R_T + 1
        d = g[:, :, None, :3] - g[:, None, :, :3]
        dx = mx.clip(d[..., 0], -R_XY, R_XY) + R_XY
        dy = mx.clip(d[..., 1], -R_XY, R_XY) + R_XY
        dt = mx.clip(d[..., 2], -R_T, R_T) + R_T
        idx = (((dx * nx + dy) * nt + dt) * 2 + g[:, :, None, 3]) * 2 + g[:, None, :, 3]
        is_cls = mx.arange(T1) == 0
        idx = mx.where((is_cls[:, None] | is_cls[None, :])[None], self.nrel, idx)  # (B,T1,T1)
        keymask = mx.where(tok == self.nd, -1e9, 0.0)[:, None, None, :]  # (B,1,1,T1)
        for l, blk in enumerate(self.blocks):
            bias = self.rel[l](idx).transpose(0, 3, 1, 2) + keymask
            x = blk(x, bias)
        if self.readout == "cls":
            return self.head(self.lnf(x[:, 0]))[:, 0]
        # 'xor': every token (CLS included) emits a flip logit l_i; the observable flips with
        # P = (1 - prod_i(1 - 2 q_i)) / 2, q_i = sigmoid(l_i), i.e. a soft parity of local claims
        # ('my chain crosses the logical'). Returned as a logit: -2 atanh(prod(-tanh(l_i/2))).
        l = self.head(self.lnf(x))[..., 0]
        t = -mx.tanh(l / 2)
        t = mx.where(tok == self.nd, 1.0, t)
        z = mx.clip(mx.prod(t, axis=1), -1 + 1e-6, 1 - 1e-6)
        return -2 * mx.arctanh(z)
