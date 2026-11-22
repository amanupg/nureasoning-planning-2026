"""NPS + components of WA-JEPA validation runs and paired log-level bootstrap of the first tag vs the others.
Usage: python src/paired_vs.py <new_tag> <ref_tag> [<ref_tag> ...]"""
import collections, sys, numpy as np
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
def ld(t):
    z = np.load(f"runs/wajepa_ft_{t}_validation.npz"); return dict(zip(map(str, z["names"]), np.asarray(z["labels"], float)))
T = sys.argv[1:]; M = {t: ld(t) for t in T}; names = sorted(set.intersection(*[set(v) for v in M.values()]))
X = {t: np.array([M[t][c] for c in names]) for t in T}
for t in T: L = X[t]; print(f"{t:12s} {nps(L).mean():.4f} " + " ".join(f"{L[:, i].mean():.4f}" for i in range(5)))
log = lambda c: "_".join(c.split("_")[:2]); Ls = sorted({log(c) for c in names}); H = set(np.random.RandomState(0).permutation(Ls)[:len(Ls) // 2])
for b in T[1:]:
    d = nps(X[T[0]]) - nps(X[b]); by = collections.defaultdict(list)
    for c, v in zip(names, d): by[log(c)].append(v)
    rng = np.random.RandomState(0); bs = [np.mean(np.concatenate([by[l] for l in rng.choice(Ls, len(Ls))])) for _ in range(2000)]
    print(f"  {T[0]} - {b}: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] halves "
          f"{d[[log(c) in H for c in names]].mean():+.4f} / {d[[log(c) not in H for c in names]].mean():+.4f}")
