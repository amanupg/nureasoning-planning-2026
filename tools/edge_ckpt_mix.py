"""Paired 3215 vs 2400 (edge alone), and the 2400+3215 trajectory mean, on validation."""
import collections, json, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
from wajepa_run import score
from ensemble_eval import wmean, nps

def one(a):
    c, t1, t2 = a; r = score("validation", c, wmean([t1, t2], [0.5, 0.5])); return c, (nps(r) if r else np.nan)

def boot(d, by, L):
    rng = np.random.RandomState(0)
    bs = [np.nanmean(np.concatenate([[d[c] for c in by[l]] for l in rng.choice(L, len(L))])) for _ in range(2000)]
    return np.nanmean(list(d.values())), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

if __name__ == "__main__":
    A = json.load(open("runs/edge_ft2400_dupfix_val_scores.json")); B = json.load(open("runs/edge_ft3215_dupfix_val_scores.json"))
    za = np.load("runs/edge_ft2400_dupfix_validation.npz"); zb = np.load("runs/edge_ft3215_dupfix_validation.npz")
    Ta = dict(zip(map(str, za["names"]), za["traj"])); Tb = dict(zip(map(str, zb["names"]), zb["traj"]))
    names = sorted(set(A) & set(B)); log = lambda c: "_".join(c.split("_")[:2]); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)
    L = sorted(by)
    print("3215 - 2400 alone: %+.4f [%+.4f, %+.4f]" % boot({c: B[c]["edge"] - A[c]["edge"] for c in names}, by, L))
    with ProcessPoolExecutor(8) as ex: M = dict(ex.map(one, [(c, Ta[c], Tb[c]) for c in names], chunksize=4))
    print(f"mean(2400, 3215) {np.nanmean(list(M.values())):.4f}  (3215 {np.nanmean([B[c]['edge'] for c in names]):.4f})")
    print("mix - 3215: %+.4f [%+.4f, %+.4f]" % boot({c: M[c] - B[c]["edge"] for c in names}, by, L))
