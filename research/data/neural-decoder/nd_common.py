"""Shared helpers: ptb64 I/O, FastSampler streaming, sparse token batches, LER statistics."""
import math, os, signal, subprocess, time
import numpy as np

ND_TOOL = os.environ.get("ND_TOOL", os.path.expanduser("~/qsim-nd/target/release/examples/nd_tool"))
LOCK = "/tmp/qsim-mac-bench.lock"


def load_meta(prefix):
    m = np.loadtxt(prefix + ".meta", dtype=np.int64, ndmin=2)
    return m  # columns: idx x y t is_x colour


def num_dets(prefix):
    for line in open(prefix + ".dem"):
        if line.startswith("# detectors"):
            return int(line.split()[-1])


def unpack_ptb64(raw, rows):
    """raw uint64 words, ptb64 (per 64-shot block: rows words) -> (shots, rows) uint8"""
    a = raw.reshape(-1, rows)
    bits = np.unpackbits(a.view(np.uint8).reshape(a.shape[0], rows, 8), axis=2, bitorder="little")
    return bits.transpose(0, 2, 1).reshape(-1, rows)


def read_ptb64(path, rows):
    return unpack_ptb64(np.fromfile(path, dtype="<u8"), rows)


class Stream:
    """Endless FastSampler stream (nd_tool stream) -> blocks of 1024 shots (dets, obs)."""

    def __init__(self, stim_path, nd, seed):
        self.rows = nd + 1
        self.nd = nd
        self.p = subprocess.Popen([ND_TOOL, "stream", stim_path, str(seed)], stdout=subprocess.PIPE,
                                  bufsize=1 << 22)

    def read(self, nblocks=1):
        n = self.rows * 8 * 16 * nblocks
        buf = self.p.stdout.read(n)
        assert len(buf) == n
        bits = unpack_ptb64(np.frombuffer(buf, dtype="<u8"), self.rows)
        return bits[:, :self.nd], bits[:, self.nd]

    def close(self):
        self.p.kill()


def tokens(dets, tmax, pad):
    """dense (B, nd) 0/1 -> (B, tmax) int32 fired-detector indices padded with `pad`, counts (B,)."""
    cnt = dets.sum(1, dtype=np.int64)
    b, j = np.nonzero(dets)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.arange(len(b)) - start[b]
    keep = pos < tmax
    out = np.full((dets.shape[0], tmax), pad, dtype=np.int32)
    out[b[keep], pos[keep]] = j[keep]
    return out, cnt


def wilson(f, n, z=1.96):
    p = f / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def per_round(x, r):
    return (1 - max(0.0, 1 - 2 * x) ** (1 / r)) / 2


def paired_ratio(fa, fb, reps=4000, seed=1):
    """fa, fb: bool fail vectors on the same shots. ratio = sum(fa)/sum(fb) with a paired
    (multinomial over the 4 joint cells) bootstrap 95% CI."""
    a, b = np.asarray(fa, bool), np.asarray(fb, bool)
    cells = np.array([(a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()], dtype=np.int64)
    n = cells.sum()
    rng = np.random.default_rng(seed)
    s = rng.multinomial(n, cells / n, size=reps)
    ra = (s[:, 0] + s[:, 1]) / np.maximum(1, s[:, 0] + s[:, 2])
    r = (cells[0] + cells[1]) / max(1, cells[0] + cells[2])
    return r, float(np.quantile(ra, 0.025)), float(np.quantile(ra, 0.975)), cells.tolist()


def wait_lock(procs=()):
    """MAC SHARING RULE: pause (SIGSTOP children, sleep self) while a peer's timing run holds the lock."""
    if not os.path.isdir(LOCK):
        return 0
    t0 = time.time()
    for p in procs:
        p.send_signal(signal.SIGSTOP)
    while os.path.isdir(LOCK):
        time.sleep(5)
    for p in procs:
        p.send_signal(signal.SIGCONT)
    return time.time() - t0


def mem_available_gb():
    """macOS free + inactive + speculative pages (vm_stat), in GB; large number elsewhere"""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return 1e9
    ps = int(out.split("page size of ")[1].split()[0])
    get = lambda k: int(out.split(k + ":")[1].split()[0].rstrip("."))
    return (get("Pages free") + get("Pages inactive") + get("Pages speculative")) * ps / 2**30


def wait_memory(min_gb=4.0, procs=()):
    """pause (SIGSTOP children) until free+inactive >= min_gb"""
    if mem_available_gb() >= min_gb:
        return 0
    t0 = time.time()
    for p in procs:
        p.send_signal(signal.SIGSTOP)
    while mem_available_gb() < min_gb:
        time.sleep(10)
    for p in procs:
        p.send_signal(signal.SIGCONT)
    return time.time() - t0
