# Score the mean of each sample-index subset (all 4-of-6 + full 6) for an Edge val run; paired log bootstrap vs full.
import os, sys, glob, json, itertools, numpy as np
sys.path.insert(0, "src")
from concurrent.futures import ProcessPoolExecutor
import edge_score as E
from plan_traj import to_global
SUBS = [tuple(range(6))] + list(itertools.combinations(range(6), 4)) + [(0, 2, 4), (0, 2, 4, 5, 1), (0, 1, 2, 4, 5)]

def job(a):
    c, T = a; return c, [(E.nps(r) if (r := E.score("validation", c, t)) else np.nan) for t in T]

if __name__ == "__main__":
    tag = sys.argv[1]; names, xy, yaw = [], [], []
    for f in sorted(glob.glob(f"runs/{tag}_validation_shard?of2.npz")):
        z = np.load(f); names += list(map(str, z["names"])); xy += list(z["xy"]); yaw += list(z["yaw"])
    xy = np.array(xy); yaw = np.array(yaw); inp = json.load(open("data/edge_inputs_validation.json"))
    def mt(i, s):
        s = list(s); m = xy[i, s].mean(0); y = np.arctan2(np.sin(yaw[i, s]).mean(0), np.cos(yaw[i, s]).mean(0))
        return to_global(E.rows51(m, y), inp[names[i]]["pose"])
    args = [(c, [mt(i, s) for s in SUBS]) for i, c in enumerate(names)]
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(job, args, chunksize=4))
    S = np.array([R[c] for c in names]); json.dump({"subs": SUBS, "R": R}, open(f"runs/{os.path.basename(tag)}_subsets.json", "w"))
    log = np.array(["_".join(c.split("_")[:2]) for c in names]); L = np.unique(log); rng = np.random.default_rng(0)
    idx = {l: np.where(log == l)[0] for l in L}
    def ci(d):
        bs = [np.nanmean(np.concatenate([d[idx[l]] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        return np.percentile(bs, [2.5, 97.5])
    full = S[:, 0]; print(f"{tag}: full-6 mean {np.nanmean(full):.4f}  ({len(L)} logs)")
    for j, s in enumerate(SUBS[1:], 1):
        d = S[:, j] - full; lo, hi = ci(d)
        print(f"  {''.join(map(str, s)):6s} {np.nanmean(S[:, j]):.4f}  diff {np.nanmean(d):+.4f} [{lo:+.4f}, {hi:+.4f}]")
