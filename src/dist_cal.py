"""Score-aware 5 s distance for the v9.5 blend path (options A1 / A2), validation, 5-fold CV by drive log.

The blend path is right (cross-track 0.19 m); its 5 s length is where the score is lost. Candidates, each fitted on 4
folds of drive logs and applied to the 5th:
  scale   one global factor on the blend length
  resid   GBDT (shallow, strong shrinkage) on the residual (human - blend length) from: blend length, member lengths
          (min / max / std, weighted), current speed / acceleration, time since stop, 1 / 3 s distance, command
  mbr     expected-score decoding: the members' lengths (v9.5 weights) widened by N(0, (s0 + s1 * L)^2) form a predictive
          mixture; pick d maximising E[human score(|d - D|)] (smoothstep, 1 m full / 8 m zero); s0, s1 fitted by CV
Each final length retimes the blend path (its own speed shape scaled); devkit NPS vs the v9.5 recipe, halves + bootstrap.
  python src/dist_cal.py
"""
import collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean, nps
from dist_stack import retime_scale, plen
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}
log_of = lambda c: "_".join(c.split("_")[:2])


def hscore(e):
    t = np.clip((np.abs(e) - 1.0) / 7.0, 0, 1); return 1 - t * t * (3 - 2 * t)


def mbr(L, w, s0, s1, grid):
    """argmax_d sum_k w_k E_{D ~ N(L_k, sig_k)} hscore(d - D), by quadrature."""
    z = np.linspace(-3, 3, 25); pz = np.exp(-z ** 2 / 2); pz /= pz.sum()
    D = (L[:, None] + (s0 + s1 * L)[:, None] * z[None])                     # [K, 25]
    ev = (w[:, None, None] * pz[None, None] * hscore(grid[None, :, None] - D[:, None, :])).sum((0, 2))
    return grid[np.argmax(ev)]


def _score(a):
    from wajepa_run import score
    c, ts = a; return c, [nps(r) if (r := score("validation", c, t)) else np.nan for t in ts]


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    fz = np.load(os.path.join(HERE, "data/dist_feats_validation.npz")); cols = list(fz["cols"]); FX = dict(zip(map(str, fz["clip"]), fz["X"])); FY = dict(zip(map(str, fz["clip"]), fz["Y"][:, 4]))
    names = sorted(set.intersection(*[set(v) for v in Z.values()]) & set(FX))
    ks = list(W); w = np.array([W[k] for k in ks]); w = w / w.sum()
    B = {c: wmean([np.asarray(Z[k][c], float) for k in ks], [W[k] for k in ks]) for c in names}
    Lm = np.array([[plen(Z[k][c]) for k in ks] for c in names]); bl = np.array([plen(B[c]) for c in names]); y = np.array([FY[c] for c in names])
    H = np.array([FX[c][[cols.index(n) for n in ("v_0.0", "a_x", "since_stop", "d1", "d3", "cmd")]] for c in names])
    X = np.c_[bl, Lm.min(1), Lm.max(1), np.sqrt((w * (Lm - (Lm * w).sum(1, keepdims=True)) ** 2).sum(1)), H]
    G = np.array([log_of(c) for c in names]); ul = np.unique(G); fo = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}; f = np.array([fo[g] for g in G])
    out = {"scale": np.zeros(len(y)), "resid": np.zeros(len(y)), "mbr": np.zeros(len(y))}; grid = np.linspace(0, 80, 321); par = []
    for k in range(5):
        tr, te = f != k, f == k
        sc = (bl[tr] * y[tr]).sum() / (bl[tr] ** 2).sum(); out["scale"][te] = sc * bl[te]
        g = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=40, l2_regularization=5.0, random_state=0)
        out["resid"][te] = bl[te] + g.fit(X[tr], y[tr] - bl[tr]).predict(X[te])
        best = (-1, None)
        for s0 in (0.5, 1.0, 2.0, 3.0):
            for s1 in (0.0, 0.03, 0.06, 0.1):
                sub = np.where(tr)[0][::3]
                v = np.mean([hscore(mbr(Lm[i], w, s0, s1, grid) - y[i]) for i in sub])
                if v > best[0]: best = (v, (s0, s1))
        par.append(best[1]); out["mbr"][te] = [mbr(Lm[i], w, *best[1], grid) for i in np.where(te)[0]]
    print(f"{len(names)} clips; 5 s length MAE: blend {np.mean(np.abs(bl - y)):.2f} | " + " | ".join(f"{k} {np.mean(np.abs(v - y)):.2f}" for k, v in out.items())
          + f" ; human-score of length: blend {hscore(bl - y).mean():.4f} | " + " | ".join(f"{k} {hscore(v - y).mean():.4f}" for k, v in out.items()))
    print("mbr widths per fold (s0, s1):", par)
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [B[c]] + [retime_scale(B[c], max(out[k][i], 0.0)) for k in out]) for i, c in enumerate(names)], chunksize=8))
    S = np.array([R[c] for c in names]); L = sorted(set(G)); Hh = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    for j, k in enumerate(out, 1):
        d = S[:, j] - S[:, 0]; by = collections.defaultdict(list)
        for g_, x in zip(G, d): by[g_].append(x)
        rng = np.random.RandomState(0); bs = [np.nanmean(np.concatenate([by[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        print(f"  {k:6s} NPS {np.nanmean(S[:, j]):.4f} vs v9.5 {np.nanmean(S[:, 0]):.4f}: {np.nanmean(d):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] "
              f"halves {np.nanmean(d[[g_ in Hh for g_ in G]]):+.4f} / {np.nanmean(d[[g_ not in Hh for g_ in G]]):+.4f}")


if __name__ == "__main__":
    main()
