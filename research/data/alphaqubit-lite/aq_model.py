"""AlphaQubit-lite (MLX): the recurrent syndrome-transformer decoder of Bausch et al., Nature 635, 834
(2024), transcribed from the Supplementary-Information pseudocode (Algorithms 1-5) and Methods /
Extended Data Figs 4 and 8, scaled down for an M1 laptop. See research/alphaqubit-lite.md for the
paper-vs-lite table.

Per experiment with R measured rounds, the model sees for every round n = 0..R-1 and every stabilizer
s: the detection event e[n,s] and the 'measurement' m[n,s] = XOR_{k<=n} e[k,s] (paper: cumulative
events when the source is a DEM). Round R (computed from final data measurements) only exists for
on-basis stabilizers and is embedded with separate projections; off-basis final entries get one
learned 'undefined' embedding.

    StabilizerEmbedder : W_m m + W_e e + E_idx[s] (+ E_ctx) -> 2-layer residual MLP     -> S_n
    RNNCore            : X <- (X + S_n)/sqrt(2); L x [ X += MHA_bias(LN X); X += GatedDense(LN X);
                                                     X <- ScatteringResidualConvBlock(X) ]
    Readout            : scatter -> 2x2 conv to data qubits -> project -> mean-pool each data-qubit line
                         parallel to the logical observable (+ embedding of R) -> residual MLP -> logit
                         per line; line 0 carries the experiment's observable.
    Attention bias     : learned embedding of (cell_i, cell_j, signed offset, Manhattan distance,
                         same-on-basis bit) -> residual MLP (state independent, computed once per call),
                         concatenated with the 7 per-round event indicator features
                         (e_n,i e_n,j ; e_n,i e_n-1,j ; e_n-1,i e_n,j ; e_n-1,i e_n-1,j and diagonals)
                         and projected to one scalar per head per layer.
    Auxiliary          : next-stabilizer-measurement prediction (linear + logistic per stabilizer).
"""
import math, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

if os.environ.get("AQ_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)
if os.environ.get("AQ_NO_MEMCAP") != "1":
    # Dylan's laptop (swarm GPU/MEMORY RULE): hard caps on MLX memory and its buffer cache
    mx.set_memory_limit(int(float(os.environ.get("AQ_MEM_GB", "1.5")) * 2**30))
    mx.set_cache_limit(int(float(os.environ.get("AQ_CACHE_MB", "256")) * 2**20))


class ResMLP(nn.Module):
    """x + W2 gelu(W1 LN(x)), `layers` times (the paper's small ResNets)."""

    def __init__(self, dim, layers, widen=1):
        super().__init__()
        self.ln = [nn.LayerNorm(dim) for _ in range(layers)]
        self.w1 = [nn.Linear(dim, widen * dim) for _ in range(layers)]
        self.w2 = [nn.Linear(widen * dim, dim) for _ in range(layers)]

    def __call__(self, x):
        for ln, a, b in zip(self.ln, self.w1, self.w2):
            x = x + b(nn.gelu(a(ln(x))))
        return x


class GatedDense(nn.Module):
    """SI Algorithm 3: W2 (GELU(Y[:w d/2]) * Y[w d/2:]) + b2 with Y = W1 x + b1."""

    def __init__(self, dim, widen):
        super().__init__()
        self.w1 = nn.Linear(dim, widen * dim)
        self.w2 = nn.Linear(widen * dim // 2, dim)

    def __call__(self, x):
        y = self.w1(x)
        a, g = mx.split(y, 2, axis=-1)
        return self.w2(nn.gelu(a) * g)


class MHABias(nn.Module):
    """SI Algorithms 1-2: multi-head self-attention with an additive per-head bias B' = W_b B."""

    def __init__(self, dim, heads, key):
        super().__init__()
        self.h, self.k = heads, key
        self.q = nn.Linear(dim, heads * key)
        self.kk = nn.Linear(dim, heads * key)
        self.v = nn.Linear(dim, heads * key)
        self.o = nn.Linear(heads * key, dim)

    def __call__(self, x, bias):
        # x (B, S, D); bias (B or 1, heads, S, S) or None
        B, S, _ = x.shape
        sh = lambda t: t.reshape(B, S, self.h, self.k).transpose(0, 2, 1, 3)
        q, k, v = sh(self.q(x)), sh(self.kk(x)), sh(self.v(x))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(self.k), mask=bias)
        return self.o(a.transpose(0, 2, 1, 3).reshape(B, S, self.h * self.k))


class ScatterConv(nn.Module):
    """SI Algorithm 4: scatter stabilizers to the (d+1)x(d+1) grid (learned padding vector P at empty
    cells), L residual dilated 3x3 convs (LN -> conv -> GELU [-> 1x1 conv back to D]), gather back."""

    def __init__(self, dim, cells, d, chans, dilations):
        super().__init__()
        self.g = d + 1
        self._flat = mx.array((cells[:, 0] * self.g + cells[:, 1]).astype(np.int32))
        self._cellmap = mx.array(grid_index(cells, d))  # grid cell -> stabilizer index, S = padding
        self.pad = mx.zeros((1, 1, dim))
        self.ln = [nn.LayerNorm(dim) for _ in dilations]
        self.conv = [nn.Conv2d(dim, c, 3, padding=dl, dilation=dl) for c, dl in zip(chans, dilations)]
        self.back = [nn.Conv2d(c, dim, 1) if c != dim else None for c in chans]

    def __call__(self, x):
        B, S, D = x.shape
        xp = mx.concatenate([x, mx.broadcast_to(self.pad, (B, 1, D))], axis=1)
        y = xp[:, self._cellmap].reshape(B, self.g, self.g, D)
        for ln, cv, bk in zip(self.ln, self.conv, self.back):
            t = nn.gelu(cv(ln(y)))
            if bk is not None:
                t = bk(t)
            y = y + t
        return y.reshape(B, self.g * self.g, D)[:, self._flat]


class Layer(nn.Module):
    def __init__(self, dim, heads, key, widen, cells, d, chans, dils):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = MHABias(dim, heads, key)
        self.ff = GatedDense(dim, widen)
        self.conv = ScatterConv(dim, cells, d, chans, dils)

    def __call__(self, x, bias):
        x = x + self.att(self.ln1(x), bias)
        x = x + self.ff(self.ln2(x))
        return self.conv(x)


def grid_index(cells, d):
    g = d + 1
    m = np.full(g * g, len(cells), np.int32)
    m[cells[:, 0] * g + cells[:, 1]] = np.arange(len(cells))
    return m


def pair_features(cells, onb):
    """discrete pair features of SI 'Attention bias' -> integer ids per feature (S, S, 6)"""
    S = len(cells)
    i, j = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    ci, cj = cells[i], cells[j]
    off = ci - cj
    g = cells.max() + 1
    f = np.stack([ci[..., 0] * g + ci[..., 1],                      # coordinates of i
                  cj[..., 0] * g + cj[..., 1],                      # coordinates of j
                  (off[..., 0] + g) * (2 * g + 1) + off[..., 1] + g,  # signed offset
                  np.abs(off).sum(-1),                              # Manhattan distance
                  (onb[i] == onb[j]).astype(np.int64),              # same basis label
                  (i == j).astype(np.int64)], -1)
    return f, [g * g, g * g, (2 * g + 1) ** 2, 2 * g + 1, 2, 2]


class AlphaQubitLite(nn.Module):
    def __init__(self, cells, onbasis, d, D=96, L=3, heads=4, key=24, widen=4, conv=48, dils=(1, 1, 1),
                 bias_dim=24, bias_layers=2, indicators=True, readout_dim=32, readout_layers=4,
                 n_ctx=16, max_rounds=64, aux=True, use_bias=True):
        super().__init__()
        cells = np.asarray(cells); onbasis = np.asarray(onbasis, bool)
        self.d, self.S, self.D, self.L = d, len(cells), D, L
        self.aux, self.use_bias, self.indicators = aux, use_bias, indicators
        S = self.S
        # stabilizer embedder (bulk) and final-round embedder
        self.w_m = nn.Linear(1, D); self.w_e = nn.Linear(1, D)
        self.w_mf = nn.Linear(1, D); self.w_ef = nn.Linear(1, D)
        self.idx = nn.Embedding(S, D)
        self.ctx = nn.Embedding(n_ctx, D)
        self.undef = mx.zeros((D,))
        self.emb_res = ResMLP(D, 2)
        self._onb = mx.array(onbasis.astype(np.float32))[None, :, None]
        # syndrome transformer
        self.layers = [Layer(D, heads, key, widen, cells, d, [conv] * len(dils), dils) for _ in range(L)]
        # attention bias
        if use_bias:
            f, sizes = pair_features(cells, onbasis)
            self._pf = [mx.array(f[..., k].astype(np.int32)) for k in range(f.shape[-1])]
            self.pemb = [nn.Embedding(n, bias_dim) for n in sizes]
            self.pres = ResMLP(bias_dim, bias_layers)
            nin = bias_dim + (7 if indicators else 0)
            self.bproj = [nn.Linear(nin, heads) for _ in range(L)]
        # readout
        self.g = d + 1
        self._cellmap = mx.array(grid_index(cells, d))
        self.ro_conv = nn.Conv2d(D, D, 2)               # (d+1)^2 stabilizer grid -> d x d data grid
        self.ro_proj = nn.Linear(D, readout_dim)
        self.ro_round = nn.Embedding(max_rounds + 1, readout_dim)
        self.ro_res = ResMLP(readout_dim, readout_layers)
        self.ro_out = nn.Linear(readout_dim, 1)
        if aux:
            self.next_stab = nn.Linear(D, 1)

    # -------------------------------------------------------------------------------------------
    def bias_embedding(self):
        e = sum(emb(f) for emb, f in zip(self.pemb, self._pf))
        return self.pres(e)  # (S, S, bias_dim)

    def indicator_feats(self, e_now, e_prev):
        """7 features (B, S, S, 7): products of current/previous events + the 3 distinct diagonals"""
        a, b = e_now[:, :, None], e_now[:, None, :]
        pa, pb = e_prev[:, :, None], e_prev[:, None, :]
        S = e_now.shape[1]
        eye = mx.eye(S)[None]
        f = [a * b, a * pb, pa * b, pa * pb,
             eye * (e_now[:, :, None] * mx.ones((1, 1, S))),
             eye * (e_prev[:, :, None] * mx.ones((1, 1, S))),
             eye * ((e_now * e_prev)[:, :, None] * mx.ones((1, 1, S)))]
        return mx.stack(f, -1)

    def embed(self, m, e, cvec, final=False):
        # m, e: (B, S) float
        if final:
            x = self.w_mf(m[..., None]) + self.w_ef(e[..., None])
            x = self._onb * x + (1 - self._onb) * self.undef
        else:
            x = self.w_m(m[..., None]) + self.w_e(e[..., None])
        x = x + self.idx.weight[None] + cvec[:, None, :]
        return self.emb_res(x)

    def core(self, x, s, bemb, e_now, e_prev):
        x = (x + s) * (1 / math.sqrt(2))
        if self.use_bias:
            if self.indicators:
                feats = mx.concatenate([mx.broadcast_to(bemb[None], (x.shape[0],) + bemb.shape),
                                        self.indicator_feats(e_now, e_prev)], -1)
            else:
                feats = bemb[None]
        for l, lay in enumerate(self.layers):
            bias = self.bproj[l](feats).transpose(0, 3, 1, 2) if self.use_bias else None
            x = lay(x, bias)
        return x

    def readout(self, x, R):
        B = x.shape[0]
        xp = mx.concatenate([x, mx.zeros((B, 1, self.D))], axis=1)
        y = self.ro_conv(xp[:, self._cellmap].reshape(B, self.g, self.g, self.D))           # (B, d, d, D)
        y = self.ro_proj(nn.gelu(y))                                      # (B, d, d, r)
        y = y.mean(axis=2)                                                # rows: lines parallel to obs
        y = y + self.ro_round(R)[:, None, :]
        y = self.ro_res(y)
        return self.ro_out(y)[..., 0]                                     # (B, d) logits; line 0 = obs

    def __call__(self, ev, final_ev, ctx, R, checkpoint=False):
        """ev (B, R, S) bulk events (float 0/1), final_ev (B, S) final-round events (on-basis only),
        ctx (B,) int context id, R (B,) int rounds. Returns (logits (B, d), aux_logits (B, R-1, S) or None)"""
        B, Rn, S = ev.shape
        meas = mx.cumsum(ev, axis=1) % 2
        bemb = self.bias_embedding() if self.use_bias else mx.zeros((1,))
        x = mx.zeros((B, S, self.D))
        # nn.utils.checkpoint passes the module parameters as explicit inputs: a bare mx.checkpoint(self._step)
        # silently drops the gradients w.r.t. captured parameters (found by test_aq gradient check)
        step = self._step if not checkpoint else nn.utils.checkpoint(self, self._step)
        zeros = mx.zeros((B, S))
        cvec = self.ctx(ctx)
        aux = []
        for n in range(Rn):
            x = step(x, meas[:, n], ev[:, n], ev[:, n - 1] if n > 0 else zeros, cvec, bemb)
            if self.aux and n < Rn - 1:
                aux.append(self.next_stab(x)[..., 0])
        mf = (meas[:, -1] + final_ev) % 2
        sf = self.embed(mf, final_ev, cvec, final=True)
        x = self.core(x, sf, bemb, final_ev, ev[:, -1])
        out = self.readout(x, R)
        return out, (mx.stack(aux, 1) if aux else None)

    def _step(self, x, m, e, eprev, cvec, bemb):
        return self.core(x, self.embed(m, e, cvec), bemb, e, eprev)
