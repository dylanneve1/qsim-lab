#!/usr/bin/env python3
"""Read members of a remote zip (Zenodo supports HTTP Range) without downloading the archive.

usage: remote_zip.py <url> list [regex]
       remote_zip.py <url> get <regex> <out-dir> [--max-gb 3]
Used for the Willow dataset (Zenodo 13273331, CC-BY-4.0): google_105Q_surface_code_d3_d5_d7.zip is
5.7 GB; only the d = 3 / d = 5 members we evaluate on are fetched."""
import io, os, re, sys, time, urllib.request, zipfile


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(30 * (attempt + 1))  # Zenodo rate limit (~130 requests / min)
        time.sleep(0.5)
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    url, cmd = sys.argv[1], sys.argv[2]
    f = io.BufferedReader(HttpFile(url), buffer_size=1 << 22)
    z = zipfile.ZipFile(f)
    if cmd == "list":
        pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
        for i in z.infolist():
            if pat is None or pat.search(i.filename):
                print(i.filename, i.file_size, i.compress_size)
        return
    pat, out = re.compile(sys.argv[3]), sys.argv[4]
    max_gb = float(sys.argv[sys.argv.index("--max-gb") + 1]) if "--max-gb" in sys.argv else 3.0
    sel = [i for i in z.infolist() if pat.search(i.filename) and not i.is_dir()]
    tot = sum(i.compress_size for i in sel)
    print(f"{len(sel)} members, {tot / 1e9:.3f} GB compressed", flush=True)
    if tot > max_gb * 1e9:
        raise SystemExit("over the download cap")
    for i in sel:
        dst = os.path.join(out, i.filename)
        if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(i) as src, open(dst + ".part", "wb") as o:
            while True:
                b = src.read(1 << 22)
                if not b:
                    break
                o.write(b)
        os.replace(dst + ".part", dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
