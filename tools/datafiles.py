#!/usr/bin/env python3
"""Large raw research data is stored xz-compressed in-tree; this tool packs,
unpacks and verifies it against research/data/COMPRESSED.tsv.

    tools/datafiles.py unpack            # restore every original next to its .xz (verified)
    tools/datafiles.py verify            # check every .xz decompresses to the recorded sha256
    tools/datafiles.py pack [MIN_BYTES]  # compress tracked text files >= MIN_BYTES (default 200000)
    tools/datafiles.py pack FILE...      # compress specific files

Run from the repo root. The manifest records, per file: original path,
sha256 and size of the original, sha256 of the .xz. Unpacked originals are
listed in research/data/.gitignore, so `unpack` leaves `git status` clean.

Analysis scripts under research/data/ read the original (uncompressed)
paths: run `tools/datafiles.py unpack` once before re-running them.

Files read by Rust tests/examples are never packed (see KEEP).
Memory: xz preset 6e (8 MiB dictionary, < 100 MB RSS) — safe on the VPS.
"""
import hashlib
import lzma
import os
import subprocess
import sys

DATA = "research/data"
MANIFEST = os.path.join(DATA, "COMPRESSED.tsv")
IGNORE = os.path.join(DATA, ".gitignore")
IGNORE_HDR = "# >>> tools/datafiles.py: unpacked originals of *.xz (do not edit by hand)\n"
IGNORE_END = "# <<< tools/datafiles.py\n"
TEXT_EXT = (".csv", ".jsonl", ".json", ".txt", ".log", ".out", ".err", ".tsv", ".stim", ".raw", ".dem")
# read at test/build time by Rust code (tests/planner.rs): must stay plain
KEEP = {f"{DATA}/simulability/raw/{n}.csv" for n in ("ct24", "ct32", "ctnn0", "brick", "arith", "qaoa")}
PRESET = 6 | lzma.PRESET_EXTREME


def sha256(path=None, data=None):
    h = hashlib.sha256()
    if data is not None:
        h.update(data)
    else:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def read_manifest():
    rows = {}
    if os.path.exists(MANIFEST):
        for line in open(MANIFEST):
            if line.startswith("#") or not line.strip():
                continue
            path, sha, size, xsha = line.rstrip("\n").split("\t")
            rows[path] = (sha, int(size), xsha)
    return rows


def write_manifest(rows):
    with open(MANIFEST, "w") as fh:
        fh.write("# path\tsha256(original)\tbytes(original)\tsha256(path.xz) -- see tools/datafiles.py\n")
        for p in sorted(rows):
            sha, size, xsha = rows[p]
            fh.write(f"{p}\t{sha}\t{size}\t{xsha}\n")
    # keep the unpacked originals out of git status
    rel = sorted(os.path.relpath(p, DATA) for p in rows)
    old = open(IGNORE).read() if os.path.exists(IGNORE) else ""
    if IGNORE_HDR in old:
        old = old[: old.index(IGNORE_HDR)] + old[old.index(IGNORE_END) + len(IGNORE_END):]
    block = IGNORE_HDR + "".join(f"/{r}\n" for r in rel) + IGNORE_END
    with open(IGNORE, "w") as fh:
        fh.write(old + block)


def decompress(xz):
    with lzma.open(xz) as fh:
        return fh.read()


def pack(args):
    rows = read_manifest()
    if args and not args[0].isdigit():
        files = args
    else:
        min_bytes = int(args[0]) if args else 200_000
        tracked = subprocess.run(["git", "ls-files", "-z", DATA], capture_output=True, check=True).stdout
        files = [
            p for p in tracked.decode().split("\0")
            if p.endswith(TEXT_EXT) and os.path.isfile(p) and os.path.getsize(p) >= min_bytes and p not in KEEP
        ]
    before = after = 0
    for p in files:
        if p in KEEP:
            print(f"skip (read by Rust tests): {p}")
            continue
        raw = open(p, "rb").read()
        comp = lzma.compress(raw, preset=PRESET)
        assert lzma.decompress(comp) == raw
        with open(p + ".xz", "wb") as fh:
            fh.write(comp)
        rows[p] = (sha256(data=raw), len(raw), sha256(data=comp))
        before += len(raw)
        after += len(comp)
        subprocess.run(["git", "rm", "-q", "--cached", p], check=True)
        os.remove(p)
        subprocess.run(["git", "add", p + ".xz"], check=True)
        print(f"{len(raw):>10} -> {len(comp):>9}  {p}")
    write_manifest(rows)
    subprocess.run(["git", "add", MANIFEST, IGNORE], check=True)
    print(f"packed {len(files)} file(s): {before} -> {after} bytes", file=sys.stderr)


def check(rows, restore):
    bad = 0
    for p, (sha, size, xsha) in sorted(rows.items()):
        xz = p + ".xz"
        if not os.path.exists(xz):
            print(f"MISSING {xz}")
            bad += 1
            continue
        if sha256(xz) != xsha:
            print(f"BAD xz sha256 {xz}")
            bad += 1
            continue
        if restore and os.path.exists(p) and os.path.getsize(p) == size and sha256(p) == sha:
            continue
        data = decompress(xz)
        if len(data) != size or sha256(data=data) != sha:
            print(f"BAD content {xz}")
            bad += 1
            continue
        if restore:
            with open(p, "wb") as fh:
                fh.write(data)
    print(f"{len(rows) - bad}/{len(rows)} ok", file=sys.stderr)
    return bad


def main(argv):
    if not argv or argv[0] not in ("pack", "unpack", "verify"):
        sys.exit(__doc__)
    if argv[0] == "pack":
        pack(argv[1:])
    else:
        sys.exit(1 if check(read_manifest(), restore=argv[0] == "unpack") else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
