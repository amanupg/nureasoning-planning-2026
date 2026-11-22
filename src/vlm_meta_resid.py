"""Residual value of the main session's meta-action VLM (P(stop/decel/keep/accel) of the future 5 s, frame 100) on top
of the v9.5 blend (2026-11-07, validation 1000 clips; probs from nureasoning/artifacts/meta_vlm_probs_validation.json,
read-only). No train OOF exists, so every mapping is fitted 5-fold by log ON validation and applied out of fold.
Pre-registered variants (all LinearRegression -> 5 s path length, blend path retimed to it):
  control  [blend length]                                   (CV recalibration alone, to separate it from the VLM)
  primary  [bl, p_stop, p_decel, p_accel, bl * each]        (VLM probs as residual features)
  keep-gate [bl, p_accel*K, p_slow*K, bl*p_accel*K, bl*p_slow*K], K = blend's own implied class is keep
  off-*     second pass after the control lost: blend length kept as an offset, Huber fit of (human - blend) on the
            same VLM terms (the full refit distorted the length; the offset form leaves it unchanged where the VLM is silent)
Paired log-level bootstrap vs v9.5 and vs control; halves from RandomState(0) log permutation.
  .venv/bin/python src/vlm_meta_resid.py
"""
import collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean, nps
from dist_stack import retime_scale, plen
from meta_oracle import F, W, log_of, _score


def implied_c4(P):                                        # blend's own 4-class meta-action, DriveMA thresholds
    v = np.r_[np.hypot(*(P[1] - P[0])[:2]) / .1, np.hypot(*np.diff(P[:, :2], axis=0).T) / .1]
    t = np.arange(len(v)) * .1; kv = np.polyfit(t, v, 1)[0]
    return 0 if (v.max() <= 1.333 and plen(P) <= 1.952) else (1 if kv <= -.448 else (3 if kv >= .429 else 2))


def main():
    from sklearn.linear_model import LinearRegression
    z = np.load(os.path.join(HERE, "data/dist_feats_validation.npz")); C = list(map(str, z["clip"])); Y = z["Y"][:, 4].astype(float)
    pr = json.load(open(os.path.join(HERE, "../nureasoning/artifacts/meta_vlm_probs_validation.json")))
    ok = [i for i, c in enumerate(C) if np.isfinite(Y[i]) and c in pr]; C = [C[i] for i in ok]; Y = Y[ok]
    Pv = np.array([[pr[c]["vlm_meta"][k] for k in ("stop", "decel", "keep", "accel")] for c in C])
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    B = {c: wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]) for c in C}; bl = np.array([plen(B[c]) for c in C])
    K = np.array([implied_c4(B[c]) for c in C]) == 2
    ps, pd, pa = Pv[:, 0], Pv[:, 1], Pv[:, 3]; sl = ps + pd
    feats = {"control": np.c_[bl], "primary": np.c_[bl, ps, pd, pa, bl * ps, bl * pd, bl * pa],
             "keep-gate": np.c_[bl, pa * K, sl * K, bl * pa * K, bl * sl * K]}
    G = np.array([log_of(c) for c in C]); ul = np.unique(G); fo = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}; f = np.array([fo[g] for g in G])
    from sklearn.linear_model import HuberRegressor
    pred = {}
    for nm, X in feats.items():                           # full refit of the length (first pass; control already loses)
        out = np.zeros(len(Y))
        for k in range(5): out[f == k] = LinearRegression().fit(X[f != k], Y[f != k]).predict(X[f == k])
        pred[nm] = np.maximum(out, 0)
    off = {"off-primary": np.c_[ps, pd, pa, bl * ps, bl * pd, bl * pa], "off-keep-gate": np.c_[pa * K, sl * K, bl * pa * K, bl * sl * K]}
    for nm, X in off.items():                             # blend length kept as offset; Huber fit of the correction only
        out = np.zeros(len(Y))
        for k in range(5): out[f == k] = HuberRegressor(alpha=1e-3, max_iter=500).fit(X[f != k], (Y - bl)[f != k]).predict(X[f == k])
        pred[nm] = np.maximum(bl + out, 0)
    mae = lambda p: np.mean(np.abs(p - Y))
    print(f"{len(C)} clips; blend implied keep on {K.mean():.2f}; 5 s length MAE: blend {mae(bl):.3f} | " + " | ".join(f"{k} {mae(p):.3f}" for k, p in pred.items()), flush=True)
    names = list(pred)
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [B[c]] + [retime_scale(B[c], pred[k][i]) for k in names]) for i, c in enumerate(C)], chunksize=8))
    S = np.array([R[c] for c in C]); m = np.all(np.isfinite(S), 1); S, Cm = S[m], [c for c, q in zip(C, m) if q]
    Ls = sorted({log_of(c) for c in Cm}); H = set(np.random.RandomState(0).permutation(Ls)[:len(Ls) // 2]); h = np.array([log_of(c) in H for c in Cm])
    def cmp(a, b, la, lb):
        d = S[:, a] - S[:, b]; by = collections.defaultdict(list)
        for c, v in zip(Cm, d): by[log_of(c)].append(v)
        rng = np.random.RandomState(0); bs = [np.mean(np.concatenate([by[l] for l in rng.choice(Ls, len(Ls))])) for _ in range(2000)]
        print(f"  {la} - {lb}: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] halves {d[h].mean():+.4f} / {d[~h].mean():+.4f}")
    print(f"NPS ({len(Cm)} clips): v9.5 {S[:, 0].mean():.4f} | " + " | ".join(f"{k} {S[:, j + 1].mean():.4f}" for j, k in enumerate(names)))
    for j, k in enumerate(names): cmp(j + 1, 0, k, "v9.5")
    for j, k in enumerate(names[1:]): cmp(j + 2, 1, k, "control")


if __name__ == "__main__":
    main()
