"""Planning SFT rows in a regular3 / delta format (decision frames plus regular frames; trajectory as deltas).

train: every Driving-decision frame with a valid meta action (two turns: action, then trajectory) plus 3 regular
  frames per clip (0.5 s grid range(15, n-50, 5), >= 10 frames from every decision frame, seeded per clip; one turn).
validation / test: key frame 100, decision prompt only (turn 1 user text, plus the turn 2 follow-up text).
Inputs: front_left / front / front_right at t-3..t (10 Hz) = 12 images, cleaned command, ego-frame velocity and
  acceleration, 16-point history (t-1.5..t, last [0, 0]), route every 10 m up to 100 m or null; 2-decimal numbers.
Target: 5 per-second displacements (delta) of the 2-decimal waypoints at future idx 9, 19, 29, 39, 49.
Images are relative names "<clip>/<camera rel path>" so the image store is decided later.
Train exclusions: train clips whose name is a val/test clip, or whose frames overlap a val/test clip of the same log
  in time (+-10 s). Every train row carries "shared_log" (its log also has val/test clips) for optional whole-log
  filtering.
  python src/plan_rows.py [--out data/plan_rows] [--workers 8] [--limit N]
"""
import argparse, collections, json, math, os, random, sys
from multiprocessing import Pool
import numpy as np

NR = os.environ.get("NR_ROOT", "/path/to/workspace/nureasoning")            # read-only
sys.path.insert(0, os.path.join(NR, "recon/devkit"))
CACHE = os.path.join(NR, "cache")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMS = ("front_left", "front", "front_right")
OFFS = (-3, -2, -1, 0)
FUT_IDX = (9, 19, 29, 39, 49)
HIST, FUT, GRID, GAP, NREG, KEY = 15, 50, 5, 10, 3, 100
NREG = int(os.environ.get("PR_NREG", NREG))   # r3 draw (11-15): 6 more regular frames -> 12 per clip (regular12)
OVERLAP_MARGIN_US = 10_000_000
# Second frame draw (Edge run B, 2026-11-13): PR_SALT changes the regular-frame seed; PR_EXCLUDE (an earlier
# planning_train.jsonl) removes the regular frames that file already used, so the new draw holds only new frames.
SALT = os.environ.get("PR_SALT", "sft_regular")
OLD_REG = set()
if os.environ.get("PR_EXCLUDE"):
    for _f in os.environ["PR_EXCLUDE"].split(","):   # comma-separated: every earlier draw
        for _l in open(_f):
            _r = json.loads(_l)
            if _r.get("kind") == "regular": OLD_REG.add((_r["clip"], _r["t"]))

COORDS = "Coordinate system: X is positive forward and Y is positive left. Units are SI."
DECISION_TASK = ("Task: Inspect the input and make the driving decision.\n"
                 "Output format:\nlongitudinal action: xx; lateral action: xx")
_DELTA = ("(5 steps at 1 Hz) of the ego vehicle as per-step displacements: [dx_k, dy_k] is waypoint k minus "
          "waypoint k-1, waypoint 0 being the current position.\n"
          "Output format:\n[dx_1, dy_1], [dx_2, dy_2], [dx_3, dy_3], [dx_4, dy_4], [dx_5, dy_5]")
TRAJ_TASK = "Task: Predict the optimal 5-second future trajectory " + _DELTA
TRAJ_FOLLOWUP = "Task: Given the above information, predict the optimal 5-second future trajectory " + _DELTA


# ------------------------------------------------------------------ formatting
def num(x): return f"{round(float(x), 2) + 0.0:.2f}"          # + 0.0 turns -0.00 into 0.00
def pt(p): return f"[{num(p[0])}, {num(p[1])}]"
def pts(ps): return ", ".join(pt(p) for p in ps)
def r3(a): return np.round(np.asarray(a, float), 3).tolist()  # features are stored at 3 decimals, printed at 2


def delta_text(points):
    p = [[round(float(v), 2) for v in q] for q in points]
    d = [[a - b for a, b in zip(q, prev)] for q, prev in zip(p, [[0.0, 0.0]] + p[:-1])]
    return pts(d)


# ------------------------------------------------------------------ geometry
def to_ego(xy, pose):
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    d = np.asarray(xy, float)[:, :2] - [pose["x"], pose["y"]]
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1)


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def dedup(xy, eps=0.05):
    return xy[np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > eps])]


def densify(xy, step=0.5):
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out.append(a + (b - a) * np.linspace(1 / n, 1, n)[:, None])
    return np.concatenate(out)


class Poly:
    def __init__(self, xy):
        self.xy = xy; seg = np.diff(xy, axis=0)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        h = np.arctan2(seg[:, 1], seg[:, 0]); self.h = np.concatenate([h, h[-1:]])

    def at(self, s):
        return np.stack([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])], -1)


def route_ahead(route_path, pose):
    """Route in the ego frame from the ego's projection (nearest densified point) onward, or None."""
    if not route_path or len(route_path) < 2: return None
    full = Poly(densify(dedup(to_ego(route_path, pose))))
    s0 = full.s[int(np.linalg.norm(full.xy, axis=1).argmin())]
    ahead = full.xy[full.s >= s0]
    return Poly(ahead) if len(ahead) >= 2 else None


def clean_command(route, raw):
    """Route heading at min(40 m, route length) vs ego heading: > +45 deg left, < -45 deg right; raw if route < 10 m."""
    if route is None or route.s[-1] < 10.0: return raw
    j = int(np.searchsorted(route.s, min(40.0, route.s[-1]))) - 1
    dh = math.degrees(wrap(route.h[max(j, 0)]))
    return "TURN_LEFT" if dh > 45.0 else "TURN_RIGHT" if dh < -45.0 else "GO_STRAIGHT"


# ------------------------------------------------------------------ clip access
class Clip:
    def __init__(self, split, name):
        from nureasoning.common.pickle_io import load_pickle
        self.split, self.name, self.dir, self._lp = split, name, os.path.join(CACHE, split, name), load_pickle
        self.frames = json.load(open(os.path.join(self.dir, "metadata.json")))["frames"]; self._ego = {}

    def ego(self, t):
        if t not in self._ego: self._ego[t] = self._lp(os.path.join(self.dir, self.frames[t]["ego_state"]))
        return self._ego[t]

    def mission(self, t): return self.frames[t].get("mission_goal") or {}

    def meta_action(self, t):
        """(lon, lat) normalized by the devkit, None if not a decision frame, False if its action is unusable."""
        rel = self.frames[t].get("reasoning")
        if not rel: return None
        p = os.path.join(self.dir, rel)
        if not os.path.isfile(p):                  # fetched later from the HF tar (src/plan_fetch.py; Driving blocks
            cands = [os.path.join(HERE, d, self.split, self.name, rel)       # in data/reasoning_driving)
                     for d in ("data/reasoning_driving", "data/reasoning_extra")]
            p = next((q for q in cands if os.path.isfile(q)), None)
            if p is None: return None
        dv = json.load(open(p)).get("Driving")
        if not isinstance(dv, dict) or not dv: return None
        from nureasoning.vqa.generate import _normalize_action_pair
        dd = dv.get("Driving decision") or {}
        lon = dd.get("Longitudinal"); lon = lon.replace(" (speed up)", "") if isinstance(lon, str) else lon
        pair = _normalize_action_pair(lon, dd.get("Lateral"))
        return pair if pair else False


def features(c, t, with_future):
    e = c.ego(t); pose = e.pose; hist = e.trajectory_history or []
    if len(hist) < HIST: return None
    if any(t + o < 0 for o in OFFS): return None
    route = route_ahead(c.mission(t).get("route_path") or [], pose)
    raw = c.mission(t).get("command") or "GO_STRAIGHT"
    f = {"t": t, "command": clean_command(route, raw), "command_raw": raw,
         "velocity": r3([e.velocity["vx"], e.velocity["vy"]]),
         "acceleration": r3([e.acceleration["ax"], e.acceleration["ay"]]),
         "history": r3(np.concatenate([to_ego(hist[-HIST:], pose), np.zeros((1, 2))])),
         "route": r3(route.at(np.arange(10.0, min(100.0, route.s[-1]) + 1e-6, 10.0))) if route else [],
         "pose": [pose["x"], pose["y"], pose["yaw"]],
         "dup_key": c.frames[t]["timestamp_us"] == c.frames[t - 1]["timestamp_us"]}
    if with_future:
        fut = e.trajectory_future or []
        if len(fut) < FUT: return None
        f["future"] = r3(to_ego(np.asarray(fut[:FUT], float), pose))
    return f


def image_block(c, t):
    labels = [f"t0-{-o / 10:g}s" if o else "t0 (present)" for o in OFFS]
    groups, paths = [], []
    for o, lab in zip(OFFS, labels):
        cams = c.frames[t + o]["sensors"]["cameras"]
        groups.append(f"at {lab}: " + "; ".join(f"{k}_view: <image>" for k in CAMS))
        paths += [f"{c.name}/{cams[k]}" for k in CAMS]
    text = (f"- {len(OFFS)} frames of multi-view images collected from the ego vehicle at "
            f"{', '.join(labels[:-1])} and {labels[-1]} timesteps, listed from oldest to newest: " + "; ".join(groups))
    return text, paths


def planning_input(images, f):
    return ("You are an expert driver.\nInput:\n" + images + "\n"
            f"- Current navigation instruction: {f['command']}\n"
            f"- Current velocity is {pt(f['velocity'])}\n"
            f"- Current acceleration is {pt(f['acceleration'])}\n"
            f"- 1.5-second past trajectory (16 steps at 10 Hz): {pts(f['history'])}\n"
            f"- Route ahead (every 10 m along the planned route, up to 100 m): {pts(f['route']) if f['route'] else 'null'}\n"
            + COORDS + "\n")


def row(c, f, kind, action=None, shared_log=None):
    imgs, paths = image_block(c, f["t"])
    prompt = planning_input(imgs, f)
    meta = {"clip": c.name, "t": f["t"], "kind": kind, "velocity": f["velocity"], "pose": f["pose"],
            "dup_key": f["dup_key"], "command_raw": f["command_raw"]}
    if shared_log is not None: meta["shared_log"] = shared_log
    if kind == "eval":
        conv = [{"from": "human", "value": prompt + DECISION_TASK}, {"from": "human_followup", "value": TRAJ_FOLLOWUP}]
        if "future" in f: meta["gt5"] = [f["future"][i] for i in FUT_IDX]
    else:
        traj = delta_text(f["future"][i] for i in FUT_IDX)
        if kind == "decision":
            conv = [{"from": "human", "value": prompt + DECISION_TASK},
                    {"from": "gpt", "value": f"longitudinal action: {action[0]}; lateral action: {action[1]}"},
                    {"from": "human", "value": TRAJ_FOLLOWUP}, {"from": "gpt", "value": traj}]
        else:
            conv = [{"from": "human", "value": prompt + TRAJ_TASK}, {"from": "gpt", "value": traj}]
    return {"conversations": conv, "images": paths, "sample_id": f"{c.name}_{f['t']}", **meta}


def build_train(args):
    name, shared_log = args
    c = Clip("train", name); n = len(c.frames); out = []; st = collections.Counter()
    dec = {}
    for t in range(n):
        a = c.meta_action(t)
        if a is not None: dec[t] = a
    for t, a in sorted(dec.items()):
        if a is False: st["decision_bad_action"] += 1; continue
        f = features(c, t, True)
        if f is None: st["decision_no_feat"] += 1; continue
        out.append(row(c, f, "decision", a, shared_log)); st["decision"] += 1
    grid = [t for t in range(HIST, n - FUT, GRID) if t not in dec and all(abs(t - d) >= GAP for d in dec)]
    feats = {t: features(c, t, True) for t in grid}
    elig = [t for t in grid if feats[t] is not None and (name, t) not in OLD_REG]
    for t in sorted(random.Random(f"{name}:{SALT}").sample(elig, min(NREG, len(elig)))):
        out.append(row(c, feats[t], "regular", None, shared_log)); st["regular"] += 1
    st["dup_rows"] += sum(r["dup_key"] for r in out)
    return out, st


def build_eval(args):
    split, name = args
    c = Clip(split, name)
    f = features(c, KEY, split != "test")
    return [row(c, f, "eval")] if f else [], collections.Counter(eval=int(bool(f)), dup_key=int(bool(f and f["dup_key"])))


def log_of(c): return "_".join(c.split("_")[:2])


def span(split, c):
    fr = json.load(open(os.path.join(CACHE, split, c, "metadata.json")))["frames"]
    return fr[0]["timestamp_us"], fr[-1]["timestamp_us"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/plan_rows")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); out = os.path.join(HERE, a.out); os.makedirs(out, exist_ok=True)
    S = {s: sorted(d for d in os.listdir(os.path.join(CACHE, s)) if os.path.isfile(os.path.join(CACHE, s, d, "metadata.json")))
         for s in ("train", "validation", "test")}
    held = collections.defaultdict(list)
    for s in ("validation", "test"):
        for c in S[s]: held[log_of(c)].append((s, span(s, c)))
    names = set(S["validation"]) | set(S["test"]); train, excl = [], collections.Counter()
    for c in S["train"]:
        if c in names: excl["same_name"] += 1; continue
        lg = log_of(c); sl = lg in held
        if sl:
            a0, b0 = span("train", c)
            hit = [s for s, (x, y) in held[lg] if a0 - OVERLAP_MARGIN_US <= y and x <= b0 + OVERLAP_MARGIN_US]
            if hit: excl["time_overlap_" + hit[0]] += 1; continue
        train.append((c, sl))
    if a.limit: train = train[:a.limit]; S = {k: v[:a.limit] for k, v in S.items()}
    print(f"train clips {len(train)} (excluded {dict(excl)}; shared-log kept {sum(s for _, s in train)})", flush=True)
    with Pool(a.workers) as pool:
        tot = collections.Counter()
        with open(os.path.join(out, "planning_train.jsonl"), "w") as fo:
            for rows, st in pool.imap(build_train, train, chunksize=8):
                tot.update(st)
                for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("train", dict(tot), flush=True)
        for s in ("validation", "test"):
            et = collections.Counter()
            with open(os.path.join(out, f"planning_{s}.jsonl"), "w") as fo:
                for rows, st in pool.imap(build_eval, [(s, c) for c in S[s]], chunksize=8):
                    et.update(st)
                    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(s, dict(et), flush=True)
    json.dump({"excluded": dict(excl), "train_clips": len(train), "train": dict(tot)}, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
