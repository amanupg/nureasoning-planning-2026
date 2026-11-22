"""Blend search with h10u4 (5 s head + 4 unfrozen encoder blocks) replacing w1; also simpler blends.
Original: blend search with the fine-tuned WA-JEPA (w1) added to the v9.3 members (validation, devkit scorer).
Members: b = v7.1-B (option-B chooser), waft = WA-JEPA fine-tuned w1, wa = WA-JEPA zero-shot,
dj = Drive-JEPA fine-tuned (3 seeds), raw = option-B raw Alpamayo path. Candidates are weight dicts
(normalised); v93 = the v9.3 recipe is the reference. Same discipline as before: pick on one half of the
drive logs, report on the other (both ways), and a log-level bootstrap vs v93.
  python src/ensemble_eval3.py -> runs/ensemble3_val_scores.json
"""
import collections, itertools, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wajepa_run import score
from ensemble_eval import wmean, nps

F = {"b": "runs/v71B_validation.npz", "waft": "runs/wajepa_ft_w1_validation.npz", "u4": "runs/wajepa_ft_h10u4_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz"}
REF = {"b": 0.4, "wa": 0.25, "dj": 0.25, "raw": 0.1}


def grid():
    C = {"v93": REF, "v94": {"b": 0.4, "waft": 0.35, "wa": 0.2, "dj": 0.25, "raw": 0.1}, "u4": {"u4": 1},
         "v94_w1->u4": {"b": 0.4, "u4": 0.35, "wa": 0.2, "dj": 0.25, "raw": 0.1}}
    for b, u, wa, dj, raw in itertools.product((0.2, 0.3, 0.4, 0.5), (0.25, 0.35, 0.5), (0.0, 0.1, 0.2), (0.0, 0.15, 0.25), (0.0, 0.1)):
        C[f"b{b}_u{u}_wa{wa}_dj{dj}_x{raw}"] = {"b": b, "u4": u, "wa": wa, "dj": dj, "raw": raw}
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
    json.dump(R, open("runs/ensemble4_val_scores.json", "w"))
    keys = list(R[names[0]]); M = {k: np.nanmean([R[c][k] for c in names]) for k in keys}
    print(f"{len(names)} validation clips; candidates {len(keys)}")
    for k in ("v93", "v94", "u4", "v94_w1->u4"): print(f"  [{k}] {M[k]:.4f}")
    for k in sorted(M, key=lambda k: -M[k])[:10]: print(f"  {k:28s} {M[k]:.4f}")
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)

    def boot(k, ref="v94"):
        d = {l: [R[c][k] - R[c][ref] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        return M[k] - M[ref], np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    tuned = [k for k in keys if k.startswith("b0")] + ["v94_w1->u4"]
    rng = np.random.RandomState(0); A = set(rng.permutation(L)[:len(L) // 2])
    for nm, inA in (("A->B", True), ("B->A", False)):
        P = [c for c in names if (log(c) in A) == inA]; Q = [c for c in names if (log(c) in A) != inA]
        best = max(tuned, key=lambda k: np.nanmean([R[c][k] for c in P]))
        print(f"  split {nm}: chose {best}; held-out {np.nanmean([R[c][best] for c in Q]):.4f} vs v9.3 recipe "
              f"{np.nanmean([R[c]['v93'] for c in Q]):.4f}")
    simple = [k for k in tuned if k.endswith("_x0.0") and "_wa0.0_" in k]
    for k in ["v94_w1->u4"] + sorted(tuned, key=lambda k: -M[k])[:4] + sorted(simple, key=lambda k: -M[k])[:2]:
        d, lo, hi = boot(k); print(f"  {k:28s} vs v9.4 recipe {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
