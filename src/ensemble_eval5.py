"""Blend re-weighting with the RL round 3 planner (rl3 = runs/wajepa_ft_rl3_s1987) in h10u4's slot.
rl3 alone beats h10u4 by +0.0099 on validation, but swapping it into the v9.5 recipe scored -0.0016, so the
weights tuned around h10u4 may not suit it. Reference = v9.5 recipe (b .4, h10u4 .35, wa .2, dj .25, raw .1).
Same discipline: pick on one half of the drive logs, score on the other (vs v9.5), and a log-level bootstrap.
  python src/ensemble_eval5.py -> runs/ensemble5_val_scores.json
"""
import collections, itertools, json, sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wajepa_run import score
from ensemble_eval import wmean, nps

F = {"b": "runs/v71B_validation.npz", "h4": "runs/wajepa_ft_h10u4_validation.npz", "rl": "runs/wajepa_ft_rl3_s1987_validation.npz",
     "wa": "runs/wajepa_169_validation.npz", "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz"}
V95 = {"b": 0.4, "h4": 0.35, "wa": 0.2, "dj": 0.25, "raw": 0.1}


def grid():
    C = {"v95": V95, "rl": {"rl": 1}, "v95_h4->rl": {**{k: v for k, v in V95.items() if k != "h4"}, "rl": 0.35}}
    for b, u, wa, dj, raw in itertools.product((0.2, 0.3, 0.4), (0.35, 0.5, 0.7, 1.0), (0.1, 0.2), (0.0, 0.15, 0.25), (0.0, 0.1)):
        C[f"b{b}_rl{u}_wa{wa}_dj{dj}_x{raw}"] = {"b": b, "rl": u, "wa": wa, "dj": dj, "raw": raw}
    return C


def _one(args):
    c, M = args; out = {}
    for k, w in grid().items():
        ks = [m for m in w if w[m] > 0]
        r = score("validation", c, wmean([M[m] for m in ks], [w[m] for m in ks])); out[k] = nps(r) if r else float("nan")
    return c, out


def main():
    Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
    names = sorted(set.intersection(*[set(v) for v in Z.values()]))
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_one, [(c, {k: np.asarray(Z[k][c], float) for k in Z}) for c in names], chunksize=4))
    json.dump(R, open("runs/ensemble5_val_scores.json", "w"))
    keys = list(R[names[0]]); M = {k: np.nanmean([R[c][k] for c in names]) for k in keys}
    print(f"{len(names)} validation clips; candidates {len(keys)}")
    for k in ("v95", "rl", "v95_h4->rl"): print(f"  [{k}] {M[k]:.4f}")
    for k in sorted(M, key=lambda k: -M[k])[:8]: print(f"  {k:28s} {M[k]:.4f}")
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)

    def boot(k, ref="v95"):
        d = {l: [R[c][k] - R[c][ref] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        return M[k] - M[ref], np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    tuned = [k for k in keys if k.startswith("b0")] + ["v95_h4->rl"]
    rng = np.random.RandomState(0); A = set(rng.permutation(L)[:len(L) // 2])
    for nm, inA in (("A->B", True), ("B->A", False)):
        P = [c for c in names if (log(c) in A) == inA]; Q = [c for c in names if (log(c) in A) != inA]
        best = max(tuned, key=lambda k: np.nanmean([R[c][k] for c in P]))
        print(f"  split {nm}: chose {best}; held-out {np.nanmean([R[c][best] for c in Q]):.4f} vs v9.5 recipe "
              f"{np.nanmean([R[c]['v95'] for c in Q]):.4f}")
    for k in sorted(tuned, key=lambda k: -M[k])[:5]:
        d, lo, hi = boot(k); print(f"  {k:28s} vs v9.5 recipe {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
