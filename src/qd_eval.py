"""Qwen-Drive-1.0 on validation: as a planner on its own and as a member of the v9.5 blend.

Reads runs/qwendrive_<tag>_validation.npz (src/qd_scenes.py collect). Scores (devkit NPS) the per-timestep mean of
its samples and its first sample, then blends: the v9.5 recipe (b .4, h10u4 .35, wa .2, dj .25, raw .1) plus Qwen-Drive
at several weights, and with Qwen-Drive replacing members. Held-out check as before: split halves by drive log and
a log-level bootstrap vs the v9.5 recipe.
  python src/qd_eval.py --tag rl
"""
import argparse, collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import score
from ensemble_eval import wmean, nps

F = {"b": "runs/v71B_validation.npz", "h4": "runs/wajepa_ft_h10u4_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz", "rl3": "runs/wajepa_ft_rl3_s1987_validation.npz"}
V95 = {"b": 0.4, "h4": 0.35, "wa": 0.2, "dj": 0.25, "raw": 0.1}


def cands():
    C = {"qd_mean": {"qd": 1}, "qd_first": {"qd1": 1}, "v95": V95}
    for w in (0.1, 0.2, 0.3, 0.45, 0.6, 0.8):
        C[f"v95+qd{w}"] = {**V95, "qd": w}
    C["v95_wa->qd"] = {**{k: v for k, v in V95.items() if k != "wa"}, "qd": 0.2}
    C["v95_raw->qd"] = {**{k: v for k, v in V95.items() if k != "raw"}, "qd": 0.1}
    C["v95_h4->rl3+qd0.3"] = {**{k: v for k, v in V95.items() if k != "h4"}, "rl3": 0.35, "qd": 0.3}
    return C


def _one(a):
    c, M = a; out = {}
    for k, w in cands().items():
        ks = [m for m in w if w[m] > 0]
        r = score("validation", c, wmean([M[m] for m in ks], [w[m] for m in ks])); out[k] = nps(r) if r else float("nan")
    return c, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="rl"); a = ap.parse_args()
    q = np.load(os.path.join(HERE, f"runs/qwendrive_{a.tag}_validation.npz"))
    Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
    Z["qd"] = dict(zip(map(str, q["names"]), q["traj"])); Z["qd1"] = dict(zip(map(str, q["names"]), q["first"]))
    names = sorted(set.intersection(*[set(v) for v in Z.values()]))
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_one, [(c, {k: np.asarray(Z[k][c], float) for k in Z}) for c in names], chunksize=4))
    json.dump(R, open(os.path.join(HERE, f"runs/qd_{a.tag}_val_scores.json"), "w"))
    M = {k: np.nanmean([R[c][k] for c in names]) for k in R[names[0]]}
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)
    H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    print(f"{len(names)} validation clips (Qwen-Drive {a.tag}, {q['samples'].shape[1]} samples)")
    for k in R[names[0]]:
        d = {l: [R[c][k] - R[c]["v95"] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        hA = np.nanmean([R[c][k] - R[c]["v95"] for c in names if log(c) in H]); hB = np.nanmean([R[c][k] - R[c]["v95"] for c in names if log(c) not in H])
        print(f"  {k:20s} {M[k]:.4f}  vs v9.5 {M[k] - M['v95']:+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  halves {hA:+.4f} / {hB:+.4f}")


if __name__ == "__main__":
    main()
