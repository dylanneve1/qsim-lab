"""Shared loaders for the transition-theory analysis."""
import glob, os, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, '..', 'magic-transition', 'raw.csv')
STEADY_COLS = "tag,n,depth,window,p_m,eta,pattern,init,seed,d_avg,d_prev,d_final,t_gates,t_act,secs".split(',')
SURV_COLS = "n,p_m,seed,k,t_inj,activated,tau,censored".split(',')
DECAY_COLS = "n,p_m,seed,t,d".split(',')


def jobs(name='jobs.txt'):
    f = os.path.join(HERE, name)
    return [l.split() for l in open(f) if l.strip()] if os.path.exists(f) else []

BATCHES = [('jobs.txt', 'parts')]


def parts(kind, partsdir=None):
    """Concatenate finished job outputs of one kind (steady/survival/decay).
    Uses the consolidated <kind>.csv (committed) when the raw parts/ directory is absent."""
    cons = os.path.join(HERE, f'{kind}.csv')
    if not os.path.isdir(os.path.join(HERE, 'parts')) and os.path.exists(cons):
        return pd.read_csv(cons)
    cols = dict(steady=STEADY_COLS, survival=SURV_COLS, decay=DECAY_COLS)[kind]
    frames = []
    for jf, pd_ in BATCHES:
        for i, j in enumerate(jobs(jf), start=1):
            f = os.path.join(HERE, pd_, f'{i}.csv')
            if j[0] != kind or not os.path.exists(f) or os.path.getsize(f) == 0:
                continue
            frames.append(pd.read_csv(f, header=None, names=cols))
    if os.path.exists(os.path.join(HERE, f'extra_{kind}.csv')):
        frames.append(pd.read_csv(os.path.join(HERE, f'extra_{kind}.csv'), header=None, names=cols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def steady_all():
    """Per-trajectory rho = d_avg/n from the new campaign and from magic-transition/raw.csv
    (d-only rows, p_t > 0; E6/E7 depth checks excluded)."""
    new = parts('steady')
    new['h'] = new.eta / new.n
    new['src'] = 'tt:' + new.tag.astype(str)
    old = pd.read_csv(OLD)
    old = old[(old['mode'] == 'dim') & (old.failed == 0) & (old.p_t > 0) & (~old.tag.isin(['E6', 'E7']))].copy()
    old['h'] = old.p_t
    old['eta'] = old.p_t * old.n
    old['pattern'] = 'poisson'
    old['init'] = 'zero'
    old['src'] = 'mt'
    old['d_prev'] = np.nan
    # identical circuit seeds were reused by the new campaign for eta = 1, 2
    # (same circ_seed mixing): keep only the new rows where a cell is in both
    newkeys = set(zip(new.n, new.eta.round(6), new.p_m.round(5), new.pattern, new.init))
    m = [(k not in newkeys) for k in zip(old.n, old.eta.round(6), old.p_m.round(5), old.pattern, old.init)]
    old = old[m]
    keep = ['src', 'n', 'depth', 'p_m', 'eta', 'h', 'pattern', 'init', 'seed', 'd_avg', 'd_prev']
    df = pd.concat([new[keep], old[keep]], ignore_index=True)
    df['rho'] = df.d_avg / df.n
    df['p_m'] = df.p_m.round(5)
    return df


def cells(df, by=('pattern', 'init', 'n', 'eta', 'p_m')):
    g = df.groupby(list(by)).agg(rho=('rho', 'mean'), err=('rho', 'sem'), cnt=('rho', 'size'),
                                 d=('d_avg', 'mean'), d_err=('d_avg', 'sem'),
                                 dprev=('d_prev', 'mean'), h=('h', 'first')).reset_index()
    return g


def consolidate():
    """parts/<id>.csv -> steady.csv, survival.csv, decay.csv (with the job id)."""
    for kind in ('steady', 'survival', 'decay'):
        cols = dict(steady=STEADY_COLS, survival=SURV_COLS, decay=DECAY_COLS)[kind]
        frames = []
        for i, j in enumerate(jobs(), start=1):
            f = os.path.join(HERE, 'parts', f'{i}.csv')
            if j[0] == kind and os.path.exists(f) and os.path.getsize(f) > 0:
                frames.append(pd.read_csv(f, header=None, names=cols).assign(job=i))
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(HERE, f'{kind}.csv'), index=False)


if __name__ == '__main__':
    consolidate()
