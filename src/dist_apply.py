"""Apply predicted travel distances to the v9.5 blend's path and score on validation (devkit NPS).

Input: runs/dist_<tag>/val_ep<k>.npz {clip, pred [N, 5] = path length at 1..5 s} from src/dist_net.py.
Variants: (a) "scale": the blend path's own speed shape scaled to the predicted 5 s length; (b) "profile": the path
re-timed to the predicted 1..5 s lengths (linear in between); (c) "mix": 5 s length = mean of predicted and blend's own.
Reports distance MAE (pred vs human, blend vs human) and NPS vs the v9.5 recipe with split halves + log bootstrap.
  python src/dist_apply.py --pred runs/dist_d0/val_ep3.npz
"""
import argparse, collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean, nps
from dist_stack import retime_scale
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}
log_of = lambda c: "_".join(c.split("_")[:2])


def retime_profile(path, L):
    """Re-time the path to arc lengths L at 1..5 s (0 at t=0, linear between), extending along the last heading."""
    t = np.arange(51) * 0.1; s_t = np.interp(t, np.arange(6), np.concatenate([[0], np.maximum.accumulate(np.maximum(L, 0))]))
    seg = np.hypot(*np.diff(path[:, :2], axis=0).T); s = np.concatenate([[0], np.cumsum(seg)]); out = np.zeros_like(path)
    for i, st in enumerate(s_t):
        if st <= s[-1] and s[-1] > 1e-6:
            j = min(np.searchsorted(s, st, side="right") - 1, len(s) - 2); f = (st - s[j]) / max(s[j + 1] - s[j], 1e-9)
            out[i] = path[j] + f * (path[j + 1] - path[j])
        else:
            h = path[-1, 2]; out[i] = [path[-1, 0] + (st - s[-1]) * np.cos(h), path[-1, 1] + (st - s[-1]) * np.sin(h), h]
    return out


def _one(a):
    from wajepa_run import score
    c, g, L = a; own = float(np.hypot(*np.diff(g[:, :2], axis=0).T).sum())
    T = {"blend": g, "scale": retime_scale(g, max(L[4], 0.0)), "profile": retime_profile(g, L), "mix": retime_scale(g, 0.5 * (max(L[4], 0.0) + own))}
    return c, {k: (nps(r) if (r := score("validation", c, t)) else np.nan) for k, t in T.items()}, own


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); a = ap.parse_args()
    q = np.load(os.path.join(HERE, a.pred)); Pd = dict(zip(map(str, q["clip"]), q["pred"]))
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    names = sorted(set.intersection(*[set(v) for v in Z.values()]) & set(Pd))
    with ProcessPoolExecutor(8) as ex:
        R = list(ex.map(_one, [(c, wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]), Pd[c]) for c in names], chunksize=8))
    S = {c: s for c, s, _ in R}; own = np.array([o for _, _, o in R])
    fz = np.load(os.path.join(HERE, "data/dist_feats_validation.npz")); Yh = dict(zip(map(str, fz["clip"]), fz["Y"][:, 4]))
    y = np.array([Yh[c] for c in names]); p = np.array([Pd[c][4] for c in names])
    print(f"{len(names)} clips; 5 s distance MAE: blend {np.mean(np.abs(own - y)):.2f} m, predicted {np.mean(np.abs(p - y)):.2f} m, mean of both {np.mean(np.abs(0.5 * (p + own) - y)):.2f} m")
    L = sorted({log_of(c) for c in names}); H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    for k in ("blend", "scale", "profile", "mix"):
        d = np.array([S[c][k] - S[c]["blend"] for c in names]); by = collections.defaultdict(list)
        for c, x in zip(names, d): by[log_of(c)].append(x)
        rng = np.random.RandomState(0); bs = [np.nanmean(np.concatenate([by[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        hA = np.nanmean([x for c, x in zip(names, d) if log_of(c) in H]); hB = np.nanmean([x for c, x in zip(names, d) if log_of(c) not in H])
        print(f"  {k:8s} NPS {np.nanmean([S[c][k] for c in names]):.4f}  vs v9.5 {np.nanmean(d):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  halves {hA:+.4f} / {hB:+.4f}")
    json.dump(S, open(os.path.join(HERE, a.pred.replace(".npz", "_nps.json")), "w"))


if __name__ == "__main__":
    main()
