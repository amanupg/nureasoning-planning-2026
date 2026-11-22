"""Does a new WA-JEPA checkpoint help the v9.5 blend? Swaps / adds runs/wajepa_ft_<tag>_validation.npz as the
'rl' member and scores each variant with the devkit; paired log-level bootstrap and split halves vs v9.5.
Usage: python src/blend_member.py <tag>"""
import collections, json, os, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
from wajepa_run import score
from ensemble_eval import wmean, nps
TAG = sys.argv[1] if len(sys.argv) > 1 else "rl4_s1987"
F = {"b": "runs/v71B_validation.npz", "u4": "runs/wajepa_ft_h10u4_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz", "rl": f"runs/wajepa_ft_{TAG}_validation.npz"}
C = {"v95": {"b": .4, "u4": .35, "wa": .2, "dj": .25, "raw": .1},
     "v95_u4->rl": {"b": .4, "rl": .35, "wa": .2, "dj": .25, "raw": .1},
     "v95_u4+rl": {"b": .4, "u4": .175, "rl": .175, "wa": .2, "dj": .25, "raw": .1},
     "v95_u4.35+rl.2": {"b": .4, "u4": .35, "rl": .2, "wa": .2, "dj": .25, "raw": .1}}
def one(a):
    c, M = a; out = {}
    for k, w in C.items():
        ks = [m for m in w if w[m] > 0]; r = score("validation", c, wmean([M[m] for m in ks], [w[m] for m in ks])); out[k] = nps(r) if r else float("nan")
    return c, out
if __name__ == "__main__":
    Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
    names = sorted(set.intersection(*[set(v) for v in Z.values()]))
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(one, [(c, {k: np.asarray(Z[k][c], float) for k in Z}) for c in names], chunksize=4))
    json.dump(R, open(f"runs/blend_{TAG}_val_scores.json", "w"))
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)
    H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    for k in C:
        d = {l: [R[c][k] - R[c]["v95"] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        hA = np.nanmean([R[c][k] - R[c]["v95"] for c in names if log(c) in H]); hB = np.nanmean([R[c][k] - R[c]["v95"] for c in names if log(c) not in H])
        print(f"  {k:16s} {np.nanmean([R[c][k] for c in names]):.4f}  vs v9.5 {np.nanmean([R[c][k]-R[c]['v95'] for c in names]):+.4f} "
              f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  halves {hA:+.4f} / {hB:+.4f}")
