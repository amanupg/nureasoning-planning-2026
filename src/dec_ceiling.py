"""Ceiling of the decision lever (2026-11-05): how much would knowing the annotated longitudinal decision help the
5 s distance, and so the planning score? Uses the TRUE annotated decision on validation (537 of 1000 clips carry one),
then the same decision corrupted to a given accuracy, so the gain can be read off for any decision predictor.
Stage A, train rows with an annotated decision: GBDT(10 s history + one-hot decision) -> 5 s path length.
Stage B, validation, 5-fold by drive log: linear in [blend length, A] -> final length; the blend path is retimed to it.
Unlabelled validation clips keep the blend unchanged. No validation label is used to fit stage A.
  python src/dec_ceiling.py   -> runs/dec_ceiling_validation.json
"""
import collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean, nps
from dist_stack import retime_scale, plen
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}                    # v9.5 recipe
ACC = [1.0, 0.87, 0.80, 0.70]                                             # decision accuracies to simulate
NDRAW = 3                                                                 # noise draws per accuracy < 1
log_of = lambda c: "_".join(c.split("_")[:2])


def feats(split):
    z = np.load(os.path.join(HERE, f"data/dist_feats_{split}.npz"))
    return {(str(c), int(r)): i for i, (c, r) in enumerate(zip(z["clip"], z["r"]))}, z["X"].astype(np.float32), z["Y"][:, 4].astype(np.float32)


def labels(split):
    z = np.load(os.path.join(HERE, f"data/dec_labels_{split}.npz"))
    return {(str(c), int(r)): int(k) for c, r, k in zip(z["clip"], z["r"], z["lon"])}


def corrupt(y, acc, prior, rng):
    """Keep each label with probability acc; otherwise replace it by another class drawn from the label prior."""
    out = y.copy()
    for i in np.where(rng.rand(len(y)) > acc)[0]:
        p = prior.copy(); p[y[i]] = 0; out[i] = rng.choice(len(p), p=p / p.sum())
    return out


def _score(a):
    from wajepa_run import score
    c, ts = a; return c, [nps(r) if (r := score("validation", c, t)) else np.nan for t in ts]


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    It, Xt, Yt = feats("train"); Iv, Xv, Yv = feats("validation"); Lt = labels("train"); Lv = labels("validation")
    kt = [k for k in Lt if k in It and np.isfinite(Yt[It[k]])]
    xt = Xt[[It[k] for k in kt]]; yt = Yt[[It[k] for k in kt]]; dt = np.array([Lt[k] for k in kt])
    oh = lambda d: np.eye(8, dtype=np.float32)[d]
    mk = lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    mA = mk().fit(np.c_[xt, oh(dt)], yt); mH = mk().fit(xt, yt)
    prior = np.bincount(dt, minlength=8) / len(dt)
    kv = [k for k in Lv if k in Iv]; C = [k[0] for k in kv]; xv = Xv[[Iv[k] for k in kv]]; yv = Yv[[Iv[k] for k in kv]]; dv = np.array([Lv[k] for k in kv])
    print(f"stage A: {len(kt)} train rows with an annotated decision; validation {len(kv)} labelled clips of 1000")
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    B = {c: wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]) for c in C}; bl = np.array([plen(B[c]) for c in C])
    G = np.array([log_of(c) for c in C]); ul = np.unique(G); fo = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}
    f = np.array([fo[g] for g in G]); mae = lambda p: float(np.mean(np.abs(p - yv)))
    def stageB(a):
        S = np.c_[bl, a]; out = np.zeros(len(C))
        for k in range(5):
            out[f == k] = LinearRegression().fit(S[f != k], yv[f != k]).predict(S[f == k])
        return out
    aH = mH.predict(xv); fH = stageB(aH)
    print(f"5 s length MAE on the {len(C)} labelled clips: blend {mae(bl):.2f} m | history only A {mae(aH):.2f}, B {mae(fH):.2f}")
    variants = [("history only (no decision)", fH)]; rng = np.random.RandomState(0)
    for acc in ACC:
        for j in range(1 if acc == 1.0 else NDRAW):
            d = dv if acc == 1.0 else corrupt(dv, acc, prior, rng)
            a = mA.predict(np.c_[xv, oh(d)]); fin = stageB(a)
            print(f"  decision accuracy {acc:.2f} (draw {j}, realised {np.mean(d == dv):.3f}): A {mae(a):.2f} m, B {mae(fin):.2f} m")
            variants.append((f"acc {acc:.2f} draw {j}", fin))
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [B[c]] + [retime_scale(B[c], max(v[i], 0.0)) for _, v in variants]) for i, c in enumerate(C)], chunksize=8))
    X = np.array([R[c] for c in C]); L = sorted(set(G)); H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    print(f"\nNPS on the {len(C)} labelled clips (v9.5 blend {np.nanmean(X[:, 0]):.4f}); effect on all 1000 = x {len(C) / 1000:.3f}")
    res = {}
    for j, (nm, _) in enumerate(variants, 1):
        d = X[:, j] - X[:, 0]; by = collections.defaultdict(list)
        for g, x in zip(G, d): by[g].append(x)
        r2 = np.random.RandomState(0); bs = [np.nanmean(np.concatenate([by[l] for l in r2.choice(L, len(L))])) for _ in range(2000)]
        res[nm] = float(np.nanmean(d))
        print(f"  {nm:28s} {np.nanmean(X[:, j]):.4f}  {np.nanmean(d):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] "
              f"halves {np.nanmean(d[[g in H for g in G]]):+.4f} / {np.nanmean(d[[g not in H for g in G]]):+.4f}  | all 1000: {np.nanmean(d) * len(C) / 1000:+.4f}")
    json.dump({"n": len(C), "effects_labelled": res, "clips": C, "decision": dv.tolist(), "blend_len": bl.tolist(), "human_len": yv.tolist(),
               "nps": X.tolist(), "variants": [nm for nm, _ in variants]}, open(os.path.join(HERE, "runs/dec_ceiling_validation.json"), "w"))


if __name__ == "__main__":
    main()
