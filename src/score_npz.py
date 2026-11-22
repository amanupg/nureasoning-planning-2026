"""Score a member's runs/<tag>_<split>.npz ('traj' = 51 global poses at 0.1 s) with the devkit, add 'labels', and
report NPS, components and the 5 s distance against the human (validation).  python src/score_npz.py <tag> [split]"""
import os, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "src"))
from dist_stack import plen
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
TAG = sys.argv[1]; SPLIT = sys.argv[2] if len(sys.argv) > 2 else "validation"


def _one(a):
    from wajepa_run import score
    c, g = a; return score(SPLIT, c, g) or [np.nan] * 5


if __name__ == "__main__":
    p = os.path.join(HERE, f"runs/{TAG}_{SPLIT}.npz"); z = dict(np.load(p)); names = list(map(str, z["names"]))
    with ProcessPoolExecutor(8) as ex: L = np.array(list(ex.map(_one, [(c, g) for c, g in zip(names, z["traj"])], chunksize=8)), float)
    z["labels"] = L; np.savez_compressed(p, **z)
    print(f"{TAG} {SPLIT}: NPS {np.nanmean(nps(L)):.4f} on {len(names)} clips | collision {np.nanmean(L[:, 0]):.3f} drivable {np.nanmean(L[:, 1]):.3f} "
          f"progress {np.nanmean(L[:, 2]):.3f} comfort {np.nanmean(L[:, 3]):.3f} human {np.nanmean(L[:, 4]):.3f}")
    if SPLIT != "test":
        f = np.load(os.path.join(HERE, f"data/dist_feats_{SPLIT}.npz")); H = dict(zip(map(str, f["clip"]), f["Y"][:, 4])); V = dict(zip(map(str, f["clip"]), f["X"][:, 0]))
        d = np.array([plen(g) for g in z["traj"]]); h = np.array([H[c] for c in names]); v = np.array([V[c] for c in names]); e = d - h
        print(f"  5 s distance vs human: bias {np.nanmean(e):+.2f} m, median {np.nanmedian(e):+.2f}, MAE {np.nanmean(np.abs(e)):.2f} "
              + " | ".join(f"v {lo}-{hi}: bias {np.nanmean(e[(v >= lo) & (v < hi)]):+.2f} MAE {np.nanmean(np.abs(e[(v >= lo) & (v < hi)])):.2f}" for lo, hi in [(0, 3), (3, 8), (8, 40)]))
