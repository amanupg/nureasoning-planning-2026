# k16_features.py SPLIT TAG OUT: per-clip planner features from an Edge K-sample run, one jsonl row per clip (for the
# reasoning thread). Ego frame (x forward, y left), 1 Hz points at 1..5 s of each sample.
#   p_long over stop/decel/keep/accel (DriveMA thresholds, as tools/dec_oracle_edge.py), p_lat over left/straight/right
#   (5 s lateral offset > +2 m / < -2 m), d5 = the K five-second distances, vmin = min speed over 0.5..4.5 s (K-mean),
#   yaw5 = mean final heading change (rad, + left), curv = median |yaw5| / d5 over samples.
import sys, json, glob, numpy as np
sys.path.insert(0, "tools")
from dec_oracle_edge import cls5
split, tag, out = sys.argv[1:4]
names, xy, yaw = [], [], []
for f in sorted(glob.glob(f"{tag}_{split}_shard*of*.npz")):
    if f.endswith(".partial.npz"): continue
    z = np.load(f); names += list(map(str, z["names"])); xy += list(z["xy"]); yaw += list(z["yaw"])
assert len(set(names)) == len(names) == 1000, len(names)
with open(out, "w") as fo:
    for c, X, Y in zip(names, xy, yaw):
        P = X[:, [9, 19, 29, 39, 49]]; K = len(P)
        lc = np.bincount([cls5(p) for p in P], minlength=4) / K
        y5 = P[:, 4, 1]; lat = np.array([(y5 > 2).mean(), (np.abs(y5) <= 2).mean(), (y5 < -2).mean()])
        d5 = np.linalg.norm(P[:, 4], axis=1); m = P.mean(0); v = np.linalg.norm(np.diff(np.vstack([[0, 0], m]), axis=0), axis=1)
        yw = Y[:, 49]
        fo.write(json.dumps({"split": split, "clip": c, "source": f"{tag.split('/')[-1]} (Edge e2-2400, K={K}, seed 43)",
                             "p_long": dict(zip(["stop", "decel", "keep", "accel"], lc.round(4).tolist())),
                             "p_lat": dict(zip(["left", "straight", "right"], lat.round(4).tolist())),
                             "d5": d5.round(3).tolist(), "vmin": round(float(v.min()), 3),
                             "yaw5": round(float(np.arctan2(np.sin(yw).mean(), np.cos(yw).mean())), 4),
                             "curv": round(float(np.median(np.abs(yw) / np.maximum(d5, 1.0))), 5)}) + "\n")
print(out, len(names))
