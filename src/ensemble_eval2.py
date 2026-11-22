"""Re-tune the planning blend around the main track's option-B chooser ("v7.1-B", handed over 2026-10-01,
nureasoning/runs_handoff, copied read-only into runs/). Members (validation, global x/y/yaw x 51):
  b    = v71B (option-B chooser + fallback)          v71  = v7.1 (reference)
  wa   = WA-JEPA zero-shot                           dj   = Drive-JEPA fine-tuned, 3-seed average
  raw  = option-B Alpamayo raw mean path (optional extra)
Candidates: v9.2 recipe (reference) and with v71 -> v71B; grid w_b in 0.4..0.8 x r (WA share of the JEPA
weight) x raw weight in {0, 0.1}. Same selection discipline as ensemble_eval.py: pick on one half of the
drive logs, report on the other (both directions), plus a log-level bootstrap vs the v9.2 recipe.
  python src/ensemble_eval2.py -> runs/ensemble2_val_scores.json, printed summary
"""
import collections, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wajepa_run import score
from ensemble_eval import wmean, nps

F = {"v71": "runs/v71_validation.npz", "b": "runs/v71B_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz"}


def candidates(M):
    C = {"v71": M["v71"], "b": M["b"],
         "v92": wmean([M["v71"], M["wa"], M["dj"]], [0.6, 0.268, 0.132]),
         "v92b": wmean([M["b"], M["wa"], M["dj"]], [0.6, 0.268, 0.132])}
    for w in (0.4, 0.5, 0.6, 0.7, 0.8):
        for r in (0.5, 0.67, 0.8, 1.0):
            for x in (0.0, 0.1):
                j = 1 - w - x
                C[f"b{w}_r{r}_x{x}"] = wmean([M["b"], M["wa"], M["dj"], M["raw"]], [w, j * r, j * (1 - r), x])
    return C


def _one(args):
    c, M = args
    return c, {k: (lambda r: nps(r) if r else float("nan"))(score("validation", c, g)) for k, g in candidates(M).items()}


def main():
    Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
    names = sorted(set.intersection(*[set(v) for v in Z.values()]))
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_one, [(c, {k: np.asarray(Z[k][c], float) for k in Z}) for c in names], chunksize=8))
    json.dump(R, open("runs/ensemble2_val_scores.json", "w"))
    keys = list(R[names[0]]); M = {k: np.nanmean([R[c][k] for c in names]) for k in keys}
    print(f"{len(names)} validation clips")
    for k in ("v71", "b", "v92", "v92b"): print(f"  [{k}] {M[k]:.4f}")
    for k in sorted(M, key=lambda k: -M[k])[:10]: print(f"  {k:18s} {M[k]:.4f}")
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)

    def boot(k, ref="v92"):
        d = {l: [R[c][k] - R[c][ref] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        return M[k] - M[ref], np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    tuned = [k for k in keys if k.startswith("b0") or k.startswith("b1")] + ["v92b"]
    rng = np.random.RandomState(0); A = set(rng.permutation(L)[:len(L) // 2])
    for nm, inA in (("A->B", True), ("B->A", False)):
        P = [c for c in names if (log(c) in A) == inA]; Q = [c for c in names if (log(c) in A) != inA]
        best = max(tuned, key=lambda k: np.nanmean([R[c][k] for c in P]))
        print(f"  split {nm}: chose {best}; held-out {np.nanmean([R[c][best] for c in Q]):.4f} vs v9.2 recipe "
              f"{np.nanmean([R[c]['v92'] for c in Q]):.4f}")
    for k in ["v92b"] + sorted(tuned, key=lambda k: -M[k])[:4]:
        d, lo, hi = boot(k); print(f"  {k:18s} vs v9.2 recipe {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
