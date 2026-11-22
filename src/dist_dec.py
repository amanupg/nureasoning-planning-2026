"""Decision-aware 5 s distance (Aman approved predicted decisions as a planning input, 2026-10-21).

Decision probabilities: the main track's lon_decision_probs_{split}.json (read-only; nureasoning/src/dec_head.py over
DINOv3 scene tokens + kinematics; train rows out-of-fold by drive log, validation / test = mean of fold models; never
trained on validation). 8 classes x 2 sources (vision, kinematics).
Stage A, train frame-100 rows: GBDT(10 s history [+ decision probs]) -> 5 s path length. Planner distances are NOT
used here (every planner was trained on these frames, so their train distances are optimistic).
Stage B, validation, 5-fold by drive log: linear combination of the blend's own 5 s length and stage A's prediction
(+ decision entropy interaction) -> final length; the blend path is retimed to it and scored with the devkit.
  python src/dist_dec.py           -> runs/dist_dec_validation.json (+ coefficients for test)
"""
import collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean, nps
from dist_stack import retime_scale, plen
NR = "/path/to/workspace/nureasoning"
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}
log_of = lambda c: "_".join(c.split("_")[:2])


def probs(split):
    J = json.load(open(os.path.join(NR, f"artifacts/lon_decision_probs_{split}.json")))
    cls = sorted(next(iter(J.values()))["vision"])
    return {c: np.r_[[v["vision"][k] for k in cls], [v["kinematics"][k] for k in cls]] for c, v in J.items()}, cls


def frame100(split):
    z = np.load(os.path.join(HERE, f"data/dist_feats_{split}.npz")); k = z["r"] == 100
    return list(map(str, z["clip"][k])), z["X"][k].astype(np.float32), z["Y"][k, 4].astype(np.float32)


def _score(a):
    from wajepa_run import score
    c, ts = a; return c, [nps(r) if (r := score("validation", c, t)) else np.nan for t in ts]


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    Pt, cls = probs("train"); Pv, _ = probs("validation")
    Ct, Xt, Yt = frame100("train"); Cv, Xv, Yv = frame100("validation")
    ht = np.array([c in Pt for c in Ct]); hv = np.array([c in Pv for c in Cv])
    Dt = np.array([Pt.get(c, np.full(16, np.nan)) for c in Ct]); Dv = np.array([Pv.get(c, np.full(16, np.nan)) for c in Cv])
    ok = np.isfinite(Yt); mk = lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    mH = mk().fit(Xt[ok], Yt[ok]); m1 = ok & ht
    mD = mk().fit(np.c_[Xt[m1], Dt[m1]], Yt[m1])
    aH = mH.predict(Xv); aD = np.where(hv, mD.predict(np.c_[Xv, np.nan_to_num(Dv)]), aH)
    print(f"train frame-100 rows {ok.sum()} ({m1.sum()} with decision probs); validation {len(Cv)} ({hv.sum()} with probs); classes {cls}")
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    B = {c: wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]) for c in Cv}; bl = np.array([plen(B[c]) for c in Cv])
    ent = np.array([-(p[:8] * np.log(p[:8] + 1e-9)).sum() if np.isfinite(p).all() else np.log(8) for p in Dv])
    mae = lambda p, m=slice(None): np.mean(np.abs(p[m] - Yv[m]))
    print(f"5 s length MAE on validation: blend {mae(bl):.2f} | A: history {mae(aH):.2f} | A: history + decisions {mae(aD):.2f} "
          f"| on the {hv.sum()} clips with probs: blend {mae(bl, hv):.2f}, A+dec {mae(aD, hv):.2f}")
    # stage B: 5-fold by log on validation, linear in [blend, A+dec, A+dec * (1 - entropy / log 8)]
    G = np.array([log_of(c) for c in Cv]); ul = np.unique(G); fo = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}
    f = np.array([fo[g] for g in G]); S = np.c_[bl, aD, aD * (1 - ent / np.log(8))]; fin = np.zeros(len(Cv)); coefs = []
    for k in range(5):
        lr = LinearRegression().fit(S[f != k], Yv[f != k]); fin[f == k] = lr.predict(S[f == k]); coefs.append(np.r_[lr.intercept_, lr.coef_])
    print(f"B (CV by log): final MAE {mae(fin):.2f} m; mean coefficients [intercept, blend, A+dec, A+dec x confidence] {np.round(np.mean(coefs, 0), 3).tolist()}")
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [B[c], retime_scale(B[c], max(fin[i], 0.0)), retime_scale(B[c], max(aD[i], 0.0))]) for i, c in enumerate(Cv)], chunksize=8))
    X = np.array([R[c] for c in Cv]); L = sorted(set(G)); H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    for j, nm in ((1, "blend path at B length"), (2, "blend path at A+dec length")):
        d = X[:, j] - X[:, 0]; by = collections.defaultdict(list)
        for g, x in zip(G, d): by[g].append(x)
        rng = np.random.RandomState(0); bs = [np.nanmean(np.concatenate([by[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        print(f"  {nm:24s} NPS {np.nanmean(X[:, j]):.4f} vs v9.5 {np.nanmean(X[:, 0]):.4f}: {np.nanmean(d):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] "
              f"halves {np.nanmean(d[[g in H for g in G]]):+.4f} / {np.nanmean(d[[g not in H for g in G]]):+.4f}")
    json.dump({"coef": np.mean(coefs, 0).tolist(), "clips": {c: {"A": float(aD[i]), "B": float(fin[i]), "blend": float(bl[i]), "human": float(Yv[i]), "nps": R[c]} for i, c in enumerate(Cv)}},
              open(os.path.join(HERE, "runs/dist_dec_validation.json"), "w"))


if __name__ == "__main__":
    main()
