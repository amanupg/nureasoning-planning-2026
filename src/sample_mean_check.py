"""Score the per-clip MEAN of the K sampled trajectories (positions averaged, yaw circular) on the 300
validation clips, vs one random sample and the model's single run. Shows how much of the gap between a
random sample and the best one is 'just average your own samples' (a test-time trick, no training)."""
import os, sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import score
from ensemble_eval import wmean
nps = lambda r: r[0] * r[1] * (.3 * r[2] + .2 * r[3] + .5 * r[4])

def one(a):
    c, G = a; g = wmean(list(G), [1] * len(G)); r = score("validation", c, g)
    return c, nps(r) if r else np.nan

if __name__ == "__main__":
    z = np.load(os.path.join(HERE, "runs/samples_w1_validation.npz")); s = np.load(os.path.join(HERE, "runs/samples_w1_validation_scores.npz"))
    S = s["labels"]; S = S[..., 0] * S[..., 1] * (.3 * S[..., 2] + .2 * S[..., 3] + .5 * S[..., 4]); names = list(map(str, s["names"]))
    T = dict(zip(map(str, z["names"]), z["traj"]))
    with ProcessPoolExecutor(8) as ex: M = dict(ex.map(one, [(c, T[c]) for c in names]))
    d = np.load(os.path.join(HERE, "runs/wajepa_ft_w1_validation.npz")); D = dict(zip(map(str, d["names"]), d["labels"]))
    det = np.mean([nps(D[c]) for c in names]); E = json.load(open(os.path.join(HERE, "runs/ensemble3_val_scores.json")))
    print(f"{len(names)} clips: random sample {np.nanmean(S):.4f} | model as run {det:.4f} | MEAN of its 16 samples {np.nanmean([M[c] for c in names]):.4f} | "
          f"current blend {np.mean([E[c]['v93'] for c in names]):.4f} | best of 16 {np.nanmean(S.max(1)):.4f}")
