"""Per-clip Alpamayo 2 Edge driving inputs (everything except pixels), built on the Mac from the main-track cache:
  - images: for each Edge slot the member path, capture timestamp (from the file name, microseconds) and camera id
    (mapping in src/edge_members.py; the tele slot 6 reuses the front frames, cropped 2x at load time);
  - 20 ego poses at 0.1 s (frames K-19..K: trajectory_history[-19:] + current pose; checked index-exact), in the
    ego frame of K, as (x, y, yaw) -> the runner builds xyz with z = 0 and yaw-only rotations;
  - nav text: "Turn left/right in Dm" at the first route point within 100 m whose heading departs > 30 deg from the
    ego heading (route = mission route in the ego frame from the ego's projection, as in src/plan_rows.py),
    else "Continue straight" (the phrase in the repo's tests); no route -> from the raw command;
  - pose (x, y, yaw) of K for the global conversion; fut64: the human 6.4 s future (frames K+1..K+64, ego frame of K,
    for fine-tuning; val/test are never trained on).
  python src/edge_inputs.py validation[,test]   -> data/edge_inputs_<split>.json
"""
import json, math, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_rows import Clip, route_ahead, to_ego, HERE, CACHE
from edge_members import SLOTS, KEY


def ts_of(member): return int(os.path.basename(member).rsplit("_", 1)[1].split(".")[0])


def nav_text(route, raw):
    if route is None:
        return {"TURN_LEFT": "Turn left", "TURN_RIGHT": "Turn right"}.get((raw or "").upper(), "Continue straight")
    s = np.arange(0.0, min(100.0, route.s[-1]) + 1e-6, 1.0)
    if len(s) < 3: return "Continue straight"
    xy = route.at(s); seg = np.diff(xy, axis=0); h = np.arctan2(seg[:, 1], seg[:, 0])
    for i in range(len(h)):
        if abs(h[i]) > math.radians(30):
            return f"Turn {'left' if h[i] > 0 else 'right'} in {int(round(s[i]))}m"
    return "Continue straight"


def future64(c, key, pose, min_valid=50):
    """6.4 s human future at exact 0.1 s steps (training target). Frame poses from key on, frames with a repeated
    timestamp dropped (283/1000 val clips repeat a frame within 100..164, which shifts index-based poses by 0.1 s),
    linearly interpolated at t_key + 0.1 k s, k = 1..64 (val: median 1 mm from the stored 10 Hz trajectory_future).
    If the clip ends before 6.4 s but covers >= min_valid steps, the tail is extended at the last 0.5 s mean velocity
    and yaw rate, and n_valid says how many steps are real (the trainer masks the rest). Else None."""
    ts, P = [], []
    for t in range(key, len(c.frames)):
        u = c.frames[t]["timestamp_us"]
        if ts and u <= ts[-1] + 1000: continue
        ts.append(u); q = c.ego(t).pose; P.append([q["x"], q["y"], q["yaw"]])
    ts = (np.array(ts, float) - ts[0]) / 1e6; P = np.array(P); P[:, 2] = np.unwrap(P[:, 2])
    want = np.arange(1, 65) * 0.1
    n_valid = int(np.sum(want <= ts[-1] + 1e-3))
    if n_valid < min_valid: return None, 0
    w = want[:n_valid]
    G = np.column_stack([np.interp(w, ts, P[:, k]) for k in range(3)])
    if n_valid < 64:
        v = (G[-1] - G[-6]) / 0.5; extra = G[-1] + np.outer(np.arange(1, 65 - n_valid) * 0.1, v)
        G = np.vstack([G, extra])
    fyaw = (G[:, 2] - pose["yaw"] + math.pi) % (2 * math.pi) - math.pi
    return np.column_stack([to_ego(G[:, :2], pose), fyaw]).round(4).tolist(), n_valid


def nearest(c, key, off):
    """Frame index whose timestamp is nearest to t_key + off/10 s (repeated frames make index offsets drift)."""
    t0 = c.frames[key]["timestamp_us"] + off * 100_000
    lo, hi = max(0, key + off - 3), min(len(c.frames), key + off + 4)
    return min(range(lo, hi), key=lambda t: abs(c.frames[t]["timestamp_us"] - t0))


def row_inputs(c, key, by_time, dupfix=False):
    """Edge inputs for clip c at key frame `key` (frame choice by index offset, or by_time: nearest timestamp).
    dupfix (index mode, val/test): when frame `key` is a byte-identical copy of key-1 (848/1000 test, 31/1000 val
    clips; same timestamp, same image files, same ego pose), every IMAGE offset is shifted by -1, i.e. the camera stack
    is built around frame key-1 = the true present, so -0.1 s is a real earlier frame (as WA_DUPFIX and the 10B dup
    map, data/plan_rows/dup_images_*.json). Ego history is NOT shifted: it comes from the key ego_state's
    trajectory_history, whose last gap is a real 0.1 s step on dup clips (checked: hist[-1] -> pose gap equals the
    gap before it), unlike WA's per-frame poses. The output frame stays the frame-`key` ego pose (= frame key-1's),
    which is what the devkit scores against. by_time mode already handles it (nearest timestamp skips the copy).
    Train rows (by_time) have a dup key on only ~5.5% of rows vs ~85% of test clips; flagged, no change."""
    e = c.ego(key); pose = e.pose
    dup = c.frames[key]["timestamp_us"] == c.frames[key - 1]["timestamp_us"]
    ikey = key - 1 if (dupfix and dup and not by_time) else key
    if ikey - 20 < 0: return None
    hist = np.asarray(e.trajectory_history, float)
    if len(hist) < 19 or key - 20 < 0: return None
    xy = to_ego(np.vstack([hist[-19:, :2], [[pose["x"], pose["y"]]]]), pose)
    yaw = np.concatenate([hist[-19:, 2], [pose["yaw"]]]) - pose["yaw"]
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    imgs = []
    for cam, cid, offs in SLOTS:
        for o in offs:
            t = nearest(c, key, o) if by_time else ikey + o
            m = f"{c.name}/{c.frames[t]['sensors']['cameras'][cam]}"
            imgs.append({"m": m, "cam": cid, "ts": ts_of(m)})
            if cid == 1: imgs.append({"m": m, "cam": 6, "ts": ts_of(m), "crop2": True})
    route = route_ahead(c.mission(key).get("route_path") or [], pose)
    fut, nv = future64(c, key, pose)
    return {"images": imgs, "hist": np.column_stack([xy, yaw]).round(4).tolist(),
            "nav": nav_text(route, c.mission(key).get("command")),
            "pose": [pose["x"], pose["y"], pose["yaw"]], "fut64": fut, "n_valid": nv,
            "dup_key": dup, "dupfix": bool(dupfix and dup and not by_time)}


def one(args):
    split, clip, dupfix = args                   # val/test inputs: key frame 100, index offsets
    return clip, row_inputs(Clip(split, clip), KEY, by_time=False, dupfix=dupfix)


def main():
    """validation: both ways (edge_inputs_validation.json without the fix, edge_inputs_validation_dupfix.json with
    it; they differ only on the 31 dup clips). test: dupfix on by default (edge_inputs_test.json), the unfixed
    variant kept as edge_inputs_test_nofix.json for reference."""
    for split in sys.argv[1].split(","):
        for dupfix in (False, True):
            run(split, dupfix)


def run(split, dupfix):
    if True:
        clips = sorted(os.listdir(os.path.join(CACHE, split)))
        with Pool(8) as p: res = dict(p.map(one, [(split, c, dupfix) for c in clips], chunksize=16))
        bad = [c for c, v in res.items() if v is None]
        res = {c: v for c, v in res.items() if v is not None}
        navs = [v["nav"].split(" in ")[0] for v in res.values()]
        name = {("validation", False): "validation", ("validation", True): "validation_dupfix",
                ("test", True): "test", ("test", False): "test_nofix"}.get((split, dupfix), f"{split}{'_dupfix' * dupfix}")
        print(name, len(res), "clips,", len(bad), "without history;", sum(v["dupfix"] for v in res.values()),
              "dup-fixed; nav:", {k: navs.count(k) for k in sorted(set(navs))}, flush=True)
        json.dump(res, open(os.path.join(HERE, "data", f"edge_inputs_{name}.json"), "w"))


if __name__ == "__main__":
    main()
