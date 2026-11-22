"""Route-cleaned turn command (src/route_cmd.py, WA_CMD=route) on the two WA-JEPA members of v9.5: singles and blend,
paired log-level bootstrap + split halves vs the originals. Usage: python src/blend_rc.py [suffix=_rc]"""
import collections, json, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
from wajepa_run import score
from ensemble_eval import wmean, nps
S = sys.argv[1] if len(sys.argv) > 1 else "_rc"
F = {"b": "runs/v71B_validation.npz", "u4": "runs/wajepa_ft_h10u4_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz",
     "u4r": f"runs/wajepa_ft_h10u4{S}_validation.npz", "war": f"runs/wajepa_169_validation{S}.npz"}
C = {"v95": {"b": .4, "u4": .35, "wa": .2, "dj": .25, "raw": .1},
     "v95_u4->rc": {"b": .4, "u4r": .35, "wa": .2, "dj": .25, "raw": .1},
     "v95_wa->rc": {"b": .4, "u4": .35, "war": .2, "dj": .25, "raw": .1},
     "v95_both->rc": {"b": .4, "u4r": .35, "war": .2, "dj": .25, "raw": .1},
     "u4": {"u4": 1}, "u4r": {"u4r": 1}, "wa": {"wa": 1}, "war": {"war": 1}}
PAIR = {"v95_u4->rc": "v95", "v95_wa->rc": "v95", "v95_both->rc": "v95", "u4r": "u4", "war": "wa"}
def one(a):
    c, M = a; out = {}
    for k, w in C.items():
        r = score("validation", c, wmean([M[m] for m in w], [w[m] for m in w])); out[k] = nps(r) if r else float("nan")
    return c, out
if __name__ == "__main__":
    Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
    names = sorted(set.intersection(*[set(v) for v in Z.values()]))
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(one, [(c, {k: np.asarray(Z[k][c], float) for k in Z}) for c in names], chunksize=4))
    json.dump(R, open(f"runs/blend{S}_val_scores.json", "w"))
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)
    H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    print(f"n={len(names)} clips, {len(L)} logs")
    for k in C:
        if k not in PAIR: print(f"  {k:14s} {np.nanmean([R[c][k] for c in names]):.4f}"); continue
        b = PAIR[k]; d = {l: [R[c][k] - R[c][b] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        hA = np.nanmean([R[c][k] - R[c][b] for c in names if log(c) in H]); hB = np.nanmean([R[c][k] - R[c][b] for c in names if log(c) not in H])
        print(f"  {k:14s} {np.nanmean([R[c][k] for c in names]):.4f}  vs {b} {np.nanmean([R[c][k]-R[c][b] for c in names]):+.4f} "
              f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  halves {hA:+.4f} / {hB:+.4f}")
