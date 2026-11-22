"""Ceiling of decision conditioning (B1 milestone): how much does knowing the TRUE longitudinal meta-action help?

Meta-actions from each frame's recorded 5 s future with DriveMA's thresholds (arXiv 2605.31271): stop if v_max <= 1.333
and distance <= 1.952 m; else by the speed slope k_v (m/s^2, least squares over the 5 s): decelerate <= -0.448,
accelerate >= 0.429, keep otherwise. Also a finer 9-bin version (stop + 8 slope bins) (k_v bins + stop) to see how resolution matters.
GBDT trained on ALL train rows (history features + true class) -> 5 s length; applied to validation with the true class;
then (CV by log on validation) combined linearly with the blend's own length; blend path retimed and scored.
  python src/meta_oracle.py
"""
import collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean, nps
from dist_stack import retime_scale, plen
from wajepa_run import NR
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}
log_of = lambda c: "_".join(c.split("_")[:2])


def future_speeds(args):
    from nureasoning.common.pickle_io import load_pickle
    split, clip, rs = args
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    out = []
    for r in rs:
        if not all(r + k in fr for k in range(51)): out.append((clip, r, None)); continue
        E = [load_pickle(os.path.join(cd, fr[r + k]["ego_state"])) for k in range(51)]
        out.append((clip, r, np.array([np.hypot(e.velocity["vx"], e.velocity["vy"]) for e in E])))
    return out


def classes(v, dist):
    t = np.arange(51) * 0.1; kv = np.polyfit(t, v, 1)[0]
    c4 = 0 if (v.max() <= 1.333 and dist <= 1.952) else (1 if kv <= -0.448 else (3 if kv >= 0.429 else 2))      # stop, decel, keep, accel
    edges = [-1.5, -0.8, -0.448, 0.0, 0.429, 0.8, 1.5]
    c8 = 0 if c4 == 0 else 1 + int(np.searchsorted(edges, kv))
    return c4, c8, kv


def _score(a):
    from wajepa_run import score
    c, ts = a; return c, [nps(r) if (r := score("validation", c, t)) else np.nan for t in ts]


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    D = {}
    for split in ("train", "validation"):
        z = np.load(os.path.join(HERE, f"data/dist_feats_{split}.npz")); by = collections.defaultdict(list)
        for c, r in zip(map(str, z["clip"]), map(int, z["r"])): by[c].append(r)
        with ProcessPoolExecutor(8) as ex:
            V = {(c, r): v for part in ex.map(future_speeds, [(split, c, rs) for c, rs in by.items()], chunksize=4) for c, r, v in part}
        keep = [i for i, (c, r) in enumerate(zip(map(str, z["clip"]), map(int, z["r"]))) if V.get((c, r)) is not None and np.isfinite(z["Y"][i, 4])]
        C = [str(z["clip"][i]) for i in keep]; X = z["X"][keep]; Y = z["Y"][keep, 4]
        K = np.array([classes(V[(C[j], int(z["r"][i]))], Y[j]) for j, i in enumerate(keep)])
        D[split] = (C, X, Y, K)
        print(f"{split}: {len(C)} rows; 4-class counts (stop, decel, keep, accel) {np.bincount(K[:, 0].astype(int), minlength=4).tolist()}", flush=True)
    Ct, Xt, Yt, Kt = D["train"]; Cv, Xv, Yv, Kv = D["validation"]
    oh = lambda k, n: np.eye(n)[k.astype(int)]
    mk = lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    pH = mk().fit(Xt, Yt).predict(Xv)
    p4 = mk().fit(np.c_[Xt, oh(Kt[:, 0], 4)], Yt).predict(np.c_[Xv, oh(Kv[:, 0], 4)])
    p8 = mk().fit(np.c_[Xt, oh(Kt[:, 1], 9)], Yt).predict(np.c_[Xv, oh(Kv[:, 1], 9)])
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    B = {c: wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]) for c in Cv}; bl = np.array([plen(B[c]) for c in Cv])
    G = np.array([log_of(c) for c in Cv]); ul = np.unique(G); fo = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}; f = np.array([fo[g] for g in G])
    comb = {}
    for nm, p in (("4-class", p4), ("8-class", p8)):
        out = np.zeros(len(Yv))
        for k in range(5):
            lr = LinearRegression().fit(np.c_[bl, p][f != k], Yv[f != k]); out[f == k] = lr.predict(np.c_[bl, p][f == k])
        comb[nm] = out
    mae = lambda p: np.mean(np.abs(p - Yv))
    print(f"5 s length MAE on validation: blend {mae(bl):.2f} | history only {mae(pH):.2f} | history + TRUE 4-class {mae(p4):.2f} | + TRUE 8-class {mae(p8):.2f} "
          f"| blend+4-class (CV) {mae(comb['4-class']):.2f} | blend+8-class (CV) {mae(comb['8-class']):.2f}")
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [B[c], retime_scale(B[c], max(p4[i], 0)), retime_scale(B[c], max(comb["4-class"][i], 0)),
                                      retime_scale(B[c], max(p8[i], 0)), retime_scale(B[c], max(comb["8-class"][i], 0))]) for i, c in enumerate(Cv)], chunksize=8))
    S = np.array([R[c] for c in Cv])
    for j, nm in enumerate(["blend (v9.5)", "TRUE 4-class + history", "TRUE 4-class + blend (CV)", "TRUE 9-class + history", "TRUE 9-class + blend (CV)"]):
        print(f"  {nm:28s} NPS {np.nanmean(S[:, j]):.4f}")


if __name__ == "__main__":
    main()
