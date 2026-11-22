"""Fetch the planning images (and missing train reasoning files) by HTTP range reads from the HF tars, resized in
memory and written ONLY at 546x359-class size (never full resolution on disk).

Members come from data/plan_members.json (src/plan_members.py). Member offsets come from a saved tar index when one
covers them (the main track's nureasoning/img_index/img_a15/<split>/<clip>/_index.json, read-only, or ours in
data/plan_index/); otherwise the whole tar is walked in K resynced slices (src/tarindex.py) and the index saved to
data/plan_index/<split>/<clip>/_index.json.

Images: PIL draft + LANCZOS to (int(w*s), int(h*s)) with s = sqrt(196608 / (w*h)), JPEG quality 95 (stored-image
spec), at data/img_plan/<split>/<member path>.
Reasoning JSON: raw bytes at data/reasoning_extra/<split>/<member path> (parsed before writing).

Resumable: a clip is skipped when data/img_plan/<split>/<clip>/_done.json exists.
Storage cap: every process keeps its cumulative written bytes in data/img_plan/_bytes/<run>.txt; before each clip the
sum over all those files is checked and the process stops itself at --cap-gb (default 48).
HF token: read in-process from $HF_HOME/token (never argv).
"""
import argparse, glob, io, json, math, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarindex import cdn_url, Ranger, _resync, _walk, BLOCK

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_IDX = "/path/to/cluster/nureasoning/img_index/img_a15"      # read-only
MAXPIX = 196608


def small_jpeg(data):
    from PIL import Image
    im = Image.open(io.BytesIO(data)); w, h = im.size
    s = math.sqrt(MAXPIX / (w * h))
    size = (int(w * s), int(h * s)) if s < 1 else (w, h)
    im.draft("RGB", size)
    b = io.BytesIO(); im.convert("RGB").resize(size, Image.LANCZOS).save(b, "JPEG", quality=95)
    return b.getvalue()


def load_index(split, clip, own):
    for p in (os.path.join(own, split, clip, "_index.json"), os.path.join(MAIN_IDX, split, clip, "_index.json")):
        if os.path.isfile(p):
            try: return json.load(open(p))
            except Exception: pass
    return {}


def walk_all(rgs, size, wanted):
    """Index the whole tar: K slices, each resynced on a safe header, walked concurrently (no early stop)."""
    K = len(rgs); b = [size * k // K for k in range(K)]
    with ThreadPoolExecutor(K) as ex:
        st = list(ex.map(lambda k: 0 if k == 0 else _resync(rgs[k], b[k], size), range(K)))
    st = sorted({s for s in st if s is not None} | {0}) + [size]
    idx = [dict() for _ in range(len(st) - 1)]
    with ThreadPoolExecutor(len(st) - 1) as ex:
        list(ex.map(lambda k: _walk(rgs[k % K], st[k], st[k + 1], size, wanted, stop_at_non_camera=False, index=idx[k]),
                    range(len(st) - 1)))
    full = {}
    for d in idx: full.update(d)
    return full


class Bytes:
    def __init__(self, root, run, cap):
        self.dir = os.path.join(root, "_bytes"); os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{run}.txt"); self.cap = cap; self.mine = 0; self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.mine += n; open(self.path, "w").write(str(self.mine))

    def total(self):
        t = 0
        for p in glob.glob(os.path.join(self.dir, "*.txt")):
            try: t += int(open(p).read().strip() or 0)
            except Exception: pass
        return t


def one(row, split, a, token, B):
    clip = row["clip"]; dst = os.path.join(a.out, split)
    imgs = row["members"] if a.mode != "reasoning" else []
    reas = row.get("reasoning", []) if a.mode != "images" else []
    done = os.path.join(a.reason_out if a.mode == "reasoning" else dst, split if a.mode == "reasoning" else "", clip, a.done_name)
    if os.path.isfile(done) or not (imgs or reas): return {"status": "skipped"}
    if B.total() > B.cap: return {"status": "capped"}
    t0 = time.time(); want = imgs + reas
    idx = load_index(split, clip, a.index_out); walked = False; nreq = nb = 0; err = ""
    for attempt in range(3):
        try:
            url = cdn_url(row["tar"], token); rgs = [Ranger(url) for _ in range(a.K)]
            if not all(m in idx for m in want) and not walked:
                idx = {**idx, **walk_all(rgs, row["size"], set(want))}; walked = True
                p = os.path.join(a.index_out, split, clip, "_index.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(idx, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
            miss_img = [m for m in imgs if m not in idx]
            if miss_img: raise IOError(f"{len(miss_img)} images not in tar")
            todo = [m for m in want if m in idx]
            written = [0]

            def grp(k):
                rg = rgs[k]
                for m in todo[k::a.K]:
                    off, sz = idx[m]; data = rg.get(off, off + sz - 1)
                    if m.endswith(".json"):
                        json.loads(data); out = os.path.join(a.reason_out, split, m)
                    else:
                        data = small_jpeg(data); out = os.path.join(dst, m)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out + ".tmp", "wb") as fh: fh.write(data)
                    os.replace(out + ".tmp", out); written[0] += len(data); B.add(len(data))
            with ThreadPoolExecutor(a.K) as ex: list(ex.map(grp, range(a.K)))
            nreq = sum(r.nreq for r in rgs); nb = sum(r.nbytes for r in rgs)
            rec = {"clip": clip, "status": "ok", "sec": round(time.time() - t0, 1), "req": nreq, "dl_bytes": nb,
                   "written": written[0], "images": len(imgs),
                   "reasoning_found": sum(m in idx for m in reas), "reasoning_missing": [m for m in reas if m not in idx],
                   "walked": walked, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "resize": "draft+LANCZOS max_pixels 196608 q95"}
            os.makedirs(os.path.dirname(done), exist_ok=True)
            json.dump(rec, open(done + ".tmp", "w")); os.replace(done + ".tmp", done)
            return rec
        except Exception as e:
            err = f"{type(e).__name__}: {e}"; time.sleep(10 * (attempt + 1))
            if not walked: idx = {}      # a saved index can be stale for this tar (2 test clips share a train clip's name)
    return {"clip": clip, "status": "failed", "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="data/plan_members.json")
    ap.add_argument("--splits", default="validation,test,train")
    ap.add_argument("--out", default="data/img_plan")
    ap.add_argument("--index-out", default="data/plan_index")
    ap.add_argument("--reason-out", default="data/reasoning_extra")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--clips", type=int, default=4, help="clips in flight")
    ap.add_argument("--K", type=int, default=12, help="connections per clip")
    ap.add_argument("--cap-gb", type=float, default=48.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["all", "images", "reasoning"], default="all")
    ap.add_argument("--done-name", default="_done.json", help="per-clip marker (a second member list for the same clip needs its own)")
    a = ap.parse_args()
    token = open(os.path.join(os.environ["HF_HOME"], "token")).read().strip()
    run = f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}_{a.shard}"
    B = Bytes(a.out, run, a.cap_gb * 1e9)
    members = json.load(open(a.members))
    for split in a.splits.split(","):
        rows = members[split][a.shard::a.nshards]
        if a.limit: rows = rows[:a.limit]
        print(f"[plan_fetch] {a.mode} {split} shard {a.shard}/{a.nshards} clips={len(rows)} inflight={a.clips} K={a.K}", flush=True)
        t0 = time.time(); c = {"ok": 0, "skipped": 0, "failed": 0, "capped": 0}; wb = dl = nimg = 0; stop = False
        with ThreadPoolExecutor(a.clips) as ex:
            futs = []
            for r in rows:
                futs.append(ex.submit(one, r, split, a, token, B))
            for i, f in enumerate(futs):
                if not stop and B.total() > B.cap:
                    stop = True; print(f"CAP_REACHED total {B.total() / 1e9:.2f} GB > {a.cap_gb} GB: cancelling", flush=True)
                    for g in futs[i:]: g.cancel()
                if f.cancelled(): continue
                r = f.result(); c[r["status"]] += 1
                if r["status"] == "capped": stop = True
                if r["status"] == "failed": print(f"  FAIL {r['clip']} {r['err']}", flush=True)
                if r["status"] == "ok":
                    wb += r["written"]; dl += r["dl_bytes"]; nimg += r["images"]
                    if r["reasoning_missing"]: print(f"  NOTE {r['clip']} reasoning not in tar: {len(r['reasoning_missing'])}", flush=True)
                if (i + 1) % 25 == 0 or i + 1 == len(rows):
                    el = max(time.time() - t0, 1)
                    print(f"  [{i+1}/{len(rows)}] {c} {el:.0f}s {(c['ok'] + c['failed']) / el * 60:.1f} clips/min "
                          f"dl {dl / el / 1e6:.1f} MB/s written {wb / 1e9:.2f} GB ({wb / max(nimg, 1) / 1e3:.0f} KB/img) "
                          f"all-runs {B.total() / 1e9:.2f} GB", flush=True)
        print(f"[plan_fetch] {split} DONE {c}", flush=True)
        if stop: print("PLAN_FETCH_STOPPED_AT_CAP", flush=True); sys.exit(3)
    print("PLAN_FETCH_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
