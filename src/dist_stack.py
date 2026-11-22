"""Predict the 5 s travel distance directly, then stretch the blend's path to it.

Diagnosis (validation, LEDGER 2026-10-21): 96% of the blend's endpoint error is along-track; keeping the blend's path
and using the human's 5 s distance alone gives 0.835 vs 0.701. So the target is one number per clip.
Features (all available on test): each member's own predicted 5 s distance (path length of its 51 poses) and the
blend's; the full 10 s ego history from the clip cache (speed at 0, -0.5 ... -10 s, acceleration now, min / max /
mean speed, time since last near-stop, distance covered in the last 1 / 3 / 10 s); the driving command.
Target: the human's 5 s path length. Model: gradient-boosted trees, 5-fold cross-validation grouped by drive log
(every clip is predicted by a model that never saw its log). The blend's path is then retimed to that distance
(its own speed shape kept, scaled) and scored with the devkit.
  python src/dist_stack.py --split validation
"""
import argparse, collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, KEY, CMD
from ensemble_eval import wmean, nps
MEM = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw", "rl3": "wajepa_ft_rl3_s1987"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}                   # v9.5 recipe
log_of = lambda c: "_".join(c.split("_")[:2])
plen = lambda g: float(np.hypot(*np.diff(np.asarray(g)[:, :2], axis=0).T).sum())


def history(split, clip):
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    E = [load_pickle(os.path.join(cd, fr[i]["ego_state"])) for i in range(0, KEY + 1)]
    v = np.array([np.hypot(e.velocity["vx"], e.velocity["vy"]) for e in E]); a = E[-1].acceleration
    P = np.array([[e.pose["x"], e.pose["y"]] for e in E]); step = np.hypot(*np.diff(P, axis=0).T)
    stopped = np.where(v < 0.5)[0]; since = (KEY - stopped[-1]) * 0.1 if len(stopped) else 10.5
    f = {f"v_{k / 10:.1f}": v[KEY - k] for k in range(0, 101, 5)}
    f.update(a_x=a["ax"], a_y=a["ay"], v_min=v.min(), v_max=v.max(), v_mean=v.mean(), since_stop=since,
             d1=step[-10:].sum(), d3=step[-30:].sum(), d10=step.sum(), cmd=CMD.get(((fr[KEY].get("mission_goal") or {}).get("command")), 3))
    G = None
    if split != "test" and KEY + 50 in fr:                                       # target only (never a feature)
        G = np.array([[q.pose["x"], q.pose["y"]] for q in (load_pickle(os.path.join(cd, fr[KEY + k]["ego_state"])) for k in range(51))])
    return clip, f, (float(np.hypot(*np.diff(G, axis=0).T).sum()) if G is not None else np.nan)


def retime_scale(path, target_len):
    """The path's own speed shape, scaled so its 5 s length equals target_len (extends along the last heading)."""
    seg = np.hypot(*np.diff(path[:, :2], axis=0).T); s = np.concatenate([[0], np.cumsum(seg)])
    tgt = s * (target_len / s[-1]) if s[-1] > 1e-3 else np.linspace(0, target_len, len(path))
    out = np.zeros_like(path)
    for i, st in enumerate(tgt):
        if st <= s[-1] and s[-1] > 1e-6:
            j = min(np.searchsorted(s, st, side="right") - 1, len(s) - 2); fr_ = (st - s[j]) / max(s[j + 1] - s[j], 1e-9)
            out[i] = path[j] + fr_ * (path[j + 1] - path[j])
        else:
            h = path[-1, 2]; out[i] = [path[-1, 0] + (st - s[-1]) * np.cos(h), path[-1, 1] + (st - s[-1]) * np.sin(h), h]
    return out


def _score(a):
    from wajepa_run import score
    c, ts = a; return c, [nps(r) if (r := score("validation", c, t)) else np.nan for t in ts]


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); a = ap.parse_args()
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in MEM.items()}
    names = sorted(set.intersection(*[set(v) for v in Z.values()]))
    with ProcessPoolExecutor(8) as ex: H = {c: (f, y) for c, f, y in ex.map(history, ["validation"] * len(names), names, chunksize=8)}
    blend = {c: wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]) for c in names}
    cols = list(H[names[0]][0]) + [f"len_{k}" for k in MEM] + ["len_blend"]
    X = np.array([list(H[c][0].values()) + [plen(Z[k][c]) for k in MEM] + [plen(blend[c])] for c in names], float)
    y = np.array([H[c][1] for c in names]); G = np.array([log_of(c) for c in names])
    ul = np.unique(G); fold = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}; fo = np.array([fold[g] for g in G])
    pred = np.zeros(len(y)); pred_h = np.zeros(len(y))
    hist_cols = [i for i, n in enumerate(cols) if not n.startswith("len_")]
    for k in range(5):
        tr, te = fo != k, fo == k
        mk = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
        pred[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        pred_h[te] = mk().fit(X[tr][:, hist_cols], y[tr]).predict(X[te][:, hist_cols])
    lb = X[:, cols.index("len_blend")]
    print(f"{len(names)} clips; 5 s distance error |pred - human| mean / median:")
    for nm, p in (("blend as is", lb), ("constant speed (v_now x 5 s)", X[:, cols.index("v_0.0")] * 5), ("learned: history only", pred_h), ("learned: history + members", pred)):
        print(f"  {nm:32s} {np.mean(np.abs(p - y)):.2f} / {np.median(np.abs(p - y)):.2f} m")
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [blend[c], retime_scale(blend[c], max(pred[i], 0.0)), retime_scale(blend[c], max(pred_h[i], 0.0))]) for i, c in enumerate(names)], chunksize=8))
    S = np.array([R[c] for c in names]); L = sorted(set(G)); by = collections.defaultdict(list)
    print(f"NPS: blend {np.nanmean(S[:, 0]):.4f}; blend path at learned distance (history + members) {np.nanmean(S[:, 1]):.4f}; (history only) {np.nanmean(S[:, 2]):.4f}")
    for j, nm in ((1, "history + members"), (2, "history only")):
        d = S[:, j] - S[:, 0]
        for c, x in zip(names, d): by[(j, log_of(c))].append(x)
        rng = np.random.RandomState(0); bs = [np.nanmean(np.concatenate([by[(j, l)] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        print(f"  {nm:18s} vs blend {np.nanmean(d):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    json.dump({c: {"pred": float(pred[i]), "pred_hist": float(pred_h[i]), "human": float(y[i]), "nps": R[c]} for i, c in enumerate(names)},
              open(os.path.join(HERE, "runs/dist_stack_validation.json"), "w"))


if __name__ == "__main__":
    main()
