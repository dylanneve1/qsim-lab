"""Sparse fired-detector transformer decoder (MLX).

A shot is the set of fired detectors (typically 5-40 of 100-600). Each fired detector is a token:
learned per-detector embedding + MLP of its (x, y, t, type, colour) features. A CLS token is
prepended. Self-attention gets an additive bias per head and layer from an MLP of the pairwise
relative geometry (dx, dy, dt, |.|, both types), i.e. a learned, translation-aware 'matching
weight'. Readout: CLS -> logit of the observable flip."""
import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def det_features(meta):
    """meta columns: idx x y t is_x colour -> (nd, F) float32 normalised features"""
    x, y, t, isx, col = (meta[:, k].astype(np.float32) for k in (1, 2, 3, 4, 5))
    def nrm(v):
        return (v - v.min()) / max(1.0, v.max() - v.min()) * 2 - 1
    f = np.stack([nrm(x), nrm(y), nrm(t), isx * 2 - 1, (col == 0) * 1.0, (col == 1) * 1.0, (col == 2) * 1.0], 1)
    return f.astype(np.float32)


def pair_geom(meta):
    """raw geometry used for relative features (x, y, t, is_x), scaled to unit lattice steps"""
    g = meta[:, 1:5].astype(np.float32)
    sx = np.unique(np.diff(np.unique(g[:, 0])))
    sy = np.unique(np.diff(np.unique(g[:, 1])))
    g[:, 0] /= max(1.0, sx.min() if len(sx) else 1.0)
    g[:, 1] /= max(1.0, sy.min() if len(sy) else 1.0)
    return g


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
    def __init__(self, meta, H=128, L=4, heads=4):
        super().__init__()
        nd = meta.shape[0]
        self.nd, self.L, self.heads = nd, L, heads
        self.feat = mx.array(np.concatenate([det_features(meta), np.zeros((2, 7), np.float32)]))  # pad, cls
        g = pair_geom(meta)
        self.geom = mx.array(np.concatenate([g, np.zeros((2, 4), np.float32)]))
        self.emb = nn.Embedding(nd + 2, H)
        self.fmlp = nn.Sequential(nn.Linear(7, H), nn.GELU(), nn.Linear(H, H))
        self.pb = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, heads * L))
        self.cls_bias = mx.zeros((heads * L,))
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
        g = self.geom[tok]  # (B,T1,4)
        dg = g[:, :, None, :3] - g[:, None, :, :3]
        ti = mx.broadcast_to(g[:, :, None, 3:4], dg.shape[:3] + (1,))
        tj = mx.broadcast_to(g[:, None, :, 3:4], dg.shape[:3] + (1,))
        rel = mx.concatenate([dg / 4, mx.abs(dg) / 4, ti, tj, mx.ones_like(ti)], axis=-1)  # (B,T1,T1,9)
        pb = self.pb(rel)  # (B,T1,T1,heads*L)
        is_cls = (mx.arange(T1) == 0)
        cls_pair = (is_cls[:, None] | is_cls[None, :])[None, :, :, None]
        pb = mx.where(cls_pair, self.cls_bias, pb)
        keymask = mx.where(tok == self.nd, -1e9, 0.0)[:, None, None, :]  # (B,1,1,T1)
        pb = pb.transpose(0, 3, 1, 2)  # (B, heads*L, T1, T1)
        for l, blk in enumerate(self.blocks):
            bias = pb[:, l * self.heads:(l + 1) * self.heads] + keymask
            x = blk(x, bias)
        return self.head(self.lnf(x[:, 0]))[:, 0]
