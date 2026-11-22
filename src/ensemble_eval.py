"""Score trajectory ensembles of WA-JEPA, Drive-JEPA and v7.1 on validation (devkit scorer).

Members are global (x, y, yaw) x 51 at the same 0.1 s times, so an ensemble is a per-timestep
weighted mean of positions (yaw: circular mean). Candidates:
  singles; mean(wa, drive); mean(all 3); v7.1-weighted mean (w_v71 in a small grid, rest split
  wa:drive = r); medoid (per clip, the member closest on average to the other two).
Tuned variants are checked by split-half over drive logs: pick on half A, report on half B, and
vice versa, so the reported number is never the one used to choose.
  python src/ensemble_eval.py runs/wajepa_169_validation.npz runs/drivejepa_validation.npz
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wajepa_run import score

nps = lambda r: r[0] * r[1] * (.3 * r[2] + .2 * r[3] + .5 * r[4])


def wmean(G, w):
    w = np.asarray(w, float) / np.sum(w)
    g = np.tensordot(w, np.asarray(G), 1)
    g[:, 2] = np.arctan2(np.tensordot(w, np.sin([x[:, 2] for x in G]), 1), np.tensordot(w, np.cos([x[:, 2] for x in G]), 1))
    return g


def medoid(G):
    d = [[np.mean(np.hypot(*(a[:, :2] - b[:, :2]).T)) for b in G] for a in G]
    return G[int(np.argmin(np.sum(d, 1)))]


def candidates(wa, dr, v7):
    C = {"wa": wa, "drive": dr, "v71": v7, "mean_wa_drive": wmean([wa, dr], [1, 1]),
         "mean_all3": wmean([wa, dr, v7], [1, 1, 1]), "medoid3": medoid([wa, dr, v7])}
    for w in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for r in (0.5, 0.67, 0.8, 1.0):                                 # r = share of the JEPA weight on WA-JEPA
            C[f"w{w}_r{r}"] = wmean([v7, wa, dr], [w, (1 - w) * r, (1 - w) * (1 - r)])
    return C


def _one(args):
    c, wa, dr, v7 = args
    out = {}
    for k, g in candidates(wa, dr, v7).items():
        r = score("validation", c, g); out[k] = nps(r) if r else float("nan")
    return c, out


def main():
    wa_z, dr_z = np.load(sys.argv[1]), np.load(sys.argv[2]); v7_z = np.load("runs/v71_validation.npz")
    W = dict(zip(map(str, wa_z["names"]), wa_z["traj"])); D = dict(zip(map(str, dr_z["names"]), dr_z["traj"]))
    V = dict(zip(map(str, v7_z["names"]), v7_z["traj"]))
    names = sorted(set(W) & set(D) & set(V))
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_one, [(c, W[c], D[c], V[c]) for c in names], chunksize=8))
    json.dump(R, open("runs/ensemble_val_scores.json", "w"))
    keys = list(next(iter(R.values())))
    M = {k: np.nanmean([R[c][k] for c in names]) for k in keys}
    print(f"{len(names)} validation clips")
    for k in sorted(M, key=lambda k: -M[k])[:12]:
        print(f"  {k:16s} {M[k]:.4f}")
    for k in ("wa", "drive", "v71", "mean_wa_drive", "mean_all3", "medoid3"):
        print(f"  [{k}] {M[k]:.4f}")
    log = lambda c: "_".join(c.split("_")[:2]); logs = sorted({log(c) for c in names})
    rng = np.random.RandomState(0); A = set(rng.permutation(logs)[:len(logs) // 2])
    tuned = [k for k in keys if k.startswith("w")]
    for name, pick, rep in (("A->B", A, None), ("B->A", None, A)):
        P = [c for c in names if (log(c) in A) == (pick is not None)]; Q = [c for c in names if c not in set(P)]
        best = max(tuned, key=lambda k: np.nanmean([R[c][k] for c in P]))
        print(f"  split {name}: chose {best} on {len(P)} clips; held-out {len(Q)} clips: {best} "
              f"{np.nanmean([R[c][best] for c in Q]):.4f} vs v71 {np.nanmean([R[c]['v71'] for c in Q]):.4f} "
              f"vs mean_all3 {np.nanmean([R[c]['mean_all3'] for c in Q]):.4f}")


if __name__ == "__main__":
    main()
