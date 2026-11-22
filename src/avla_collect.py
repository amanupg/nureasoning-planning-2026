"""AutoVLA predictions (runs/avla/<tag>_<split>_s*.jsonl, 10 poses at 0.5 s in the ego frame) -> runs/avla_<tag>_<split>.npz
{names, traj [N, 51, 3] global (wajepa_run.to_51 with n=10: native 5 s, linear in between), text}; --score adds devkit labels.
  python src/avla_collect.py --tag zs --split validation --score
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import to_51, score


def _s(a):
    c, g = a; return c, score("validation", c, g)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="zs"); ap.add_argument("--split", default="validation"); ap.add_argument("--score", action="store_true")
    a = ap.parse_args(); R = {}
    for f in sorted(glob.glob(os.path.join(HERE, f"runs/avla/{a.tag}_{a.split}_s*.jsonl"))):
        for l in open(f): r = json.loads(l); R[r["clip"]] = r
    z = np.load(os.path.join(HERE, f"data/wa_{a.split}_meta.npz")); O = dict(zip(map(str, z["clip"]), z["origin"]))
    names = sorted(c for c in R if len(R[c]["traj10"]) == 10); bad = len(R) - len(names)
    T = np.array([to_51(np.asarray(R[c]["traj10"], float), O[c]) for c in names])
    out = {"names": np.array(names), "traj": T, "text": np.array([R[c]["text"] for c in names])}
    if a.score:
        with ProcessPoolExecutor(8) as ex: S = dict(ex.map(_s, [(c, t) for c, t in zip(names, T)], chunksize=8))
        L = np.array([S[c] if S[c] else [np.nan] * 5 for c in names], float); out["labels"] = L
        N = L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4])
        print(f"AutoVLA {a.tag} {a.split}: NPS {np.nanmean(N):.4f} on {len(names)} clips ({bad} without 10 poses) | collision {np.nanmean(L[:, 0]):.3f} "
              f"drivable {np.nanmean(L[:, 1]):.3f} progress {np.nanmean(L[:, 2]):.3f} comfort {np.nanmean(L[:, 3]):.3f} human {np.nanmean(L[:, 4]):.3f}")
    np.savez(os.path.join(HERE, f"runs/avla_{a.tag}_{a.split}.npz"), **out)
