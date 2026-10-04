#!/usr/bin/env python3
"""Interleaved LER campaign: arms are run chunk by chunk, round-robin. Each arm stops once it has
`fails` failures or `max_shots` shots.
Mac sharing rule: Monte-Carlo campaigns never hold the bench lock; with PAUSE_ON_LOCK=1 the chunk's
whole process group is SIGSTOPped while anyone holds /tmp/qsim-mac-bench.lock and SIGCONTed after.
(LOCK=1 restores the old behaviour of holding the lock per chunk; not used any more.)

usage: campaign.py <jobs.json> <out.jsonl>
jobs.json: list of {"dec": "tess"|"bposd", "d", "R", "noise", "p", "spec", "chunk", "workers",
                    "basis": "z"|"x", "orders", "beam", "osd", "fails", "max_shots", "tag"}
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)
CS = os.environ.get("CS", "/tmp/cf-target/release/examples/color_search")
LER = os.environ.get("LER", "/tmp/cf-target/release/examples/color_ler")
LOCK = "/tmp/qsim-mac-bench.lock"
jobs = json.load(open(sys.argv[1]))
out = sys.argv[2]
done = {}
if os.path.exists(out):
    for l in open(out):
        j = json.loads(l)
        k = j["tag"]
        f, n, c = done.get(k, (0, 0, 0))
        done[k] = (f + j["fails"], n + j["shots"], c + 1)


def run(job, seed):
    if job["dec"] == "tess":
        env = dict(os.environ, TESS_ORDERS=str(job.get("orders", 16)), TESS_BEAM=str(job.get("beam", 15)), CS=CS,
                   OWN_LOCK="1" if os.environ.get("LOCK") == "1" else "0")
        cmd = [PY, os.path.join(HERE, "ler_tess.py"), str(job["d"]), str(job["R"]), job["noise"], str(job["p"]),
               job["spec"], str(job["chunk"]), str(seed), str(job["workers"]), job.get("basis", "z")]
    else:
        env = dict(os.environ, BASIS=job.get("basis", "z"))
        cmd = [LER, str(job["d"]), str(job["R"]), job["noise"], str(job["p"]), job["spec"], str(job["chunk"]),
               str(seed), str(job["workers"]), str(job.get("osd", 100))]
    cmd = ["nice", "-n", os.environ.get("NICE", "10")] + cmd
    if os.environ.get("PAUSE_ON_LOCK") != "1":
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return json.loads(r.stdout.strip().splitlines()[-1])
    import signal, tempfile
    with tempfile.TemporaryFile("w+") as fo:
        pr = subprocess.Popen(cmd, env=env, stdout=fo, stderr=subprocess.DEVNULL, start_new_session=True)
        paused = False
        while pr.poll() is None:
            held = os.path.isdir(LOCK)
            if held and not paused:
                os.killpg(pr.pid, signal.SIGSTOP); paused = True
            elif not held and paused:
                os.killpg(pr.pid, signal.SIGCONT); paused = False
            time.sleep(1)
        if pr.returncode != 0:
            raise RuntimeError(f"chunk failed: {cmd}")
        fo.seek(0)
        return json.loads(fo.read().strip().splitlines()[-1])


k = 0
while True:
    active = [j for j in jobs if done.get(j["tag"], (0, 0, 0))[0] < j["fails"]
              and done.get(j["tag"], (0, 0, 0))[1] < j["max_shots"]]
    if not active:
        break
    for job in active:
        f, n, c = done.get(job["tag"], (0, 0, 0))
        # distinct per tag and chunk; SEED_OFFSET separates machines sharing a tag
        seed = int.from_bytes(job["tag"].encode()[-6:], "little") * 1000 + c + 1 + int(os.environ.get("SEED_OFFSET", "0"))
        if os.environ.get("LOCK") == "1":
            while True:
                try:
                    os.mkdir(LOCK)
                    break
                except FileExistsError:
                    time.sleep(5)
        t = time.time()
        try:
            r = run(job, seed)
        finally:
            if os.environ.get("LOCK") == "1":
                os.rmdir(LOCK)
        r.update(tag=job["tag"], dec=job["dec"], chunk_s=round(time.time() - t, 1), seed=seed)
        with open(out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        done[job["tag"]] = (f + r["fails"], n + r["shots"], c + 1)
        print(job["tag"], done[job["tag"]], r["chunk_s"], "s", flush=True)
    k += 1
