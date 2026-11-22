# best-of-K oracle headroom for an Edge val run: score every candidate, report mean of each, oracle max, and mean-traj.
import sys, os, glob, json, numpy as np
sys.path.insert(0, "src")
from concurrent.futures import ProcessPoolExecutor
import edge_score as E
from plan_traj import to_global

def job(a):
    c, cands = a; out = []
    for T in cands:
        r = E.score("validation", c, T); out.append(E.nps(r) if r else np.nan)
    return c, out

if __name__ == "__main__":
    tag = sys.argv[1]; names, xy, yaw = [], [], []
    for f in sorted(glob.glob(f"{tag}_validation_shard*of*.npz")):
        if f.endswith(".partial.npz"): continue
        z = np.load(f); names += list(map(str, z["names"])); xy += list(z["xy"]); yaw += list(z["yaw"])
    inp = json.load(open("data/edge_inputs_validation.json")); xy = np.array(xy); yaw = np.array(yaw)
    args = [(c, [to_global(E.rows51(xy[i, k], yaw[i, k]), inp[c]["pose"]) for k in range(xy.shape[1])]) for i, c in enumerate(names)]
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(job, args, chunksize=4))
    S = np.array([R[c] for c in names]); json.dump(R, open(f"{tag}_cand_scores.json", "w"))
    mean_sc = json.load(open(f"{tag}_val_scores.json"))
    print(f"K={S.shape[1]} n={len(names)}: single-sample mean {np.nanmean(S):.4f}; mean-traj {np.mean([mean_sc[c]['edge'] for c in names]):.4f}; "
          f"oracle best-of-K {np.nanmean(np.nanmax(S, 1)):.4f}; oracle best-of-2 {np.nanmean(np.nanmax(S[:, :2], 1)):.4f}; "
          f"worst {np.nanmean(np.nanmin(S, 1)):.4f}; per-clip spread (max-min) median {np.nanmedian(np.nanmax(S,1)-np.nanmin(S,1)):.3f}")
