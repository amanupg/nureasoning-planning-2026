"""Merge Edge test shards into a member file: python src/edge_merge_test.py runs/<tag>  ->  runs/<tag>_test.npz
(names, traj = K-mean global 51x3, traj0 = candidate 0, ego_mean, K), with the same frame code as src/edge_score.py."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "src")
from edge_score import rows51
from plan_traj import to_global

if __name__ == "__main__":
    tag = sys.argv[1]; names, xy, yaw = [], [], []
    for f in sorted(glob.glob(f"{tag}_test_shard*of*.npz")):
        if f.endswith(".partial.npz"): continue
        z = np.load(f); names += list(map(str, z["names"])); xy += list(z["xy"]); yaw += list(z["yaw"])
    o = np.argsort(names); names = [names[i] for i in o]; xy = np.array(xy)[o]; yaw = np.array(yaw)[o]
    inp = json.load(open("data/edge_inputs_test.json"))
    mxy = xy.mean(1); myaw = np.arctan2(np.sin(yaw).mean(1), np.cos(yaw).mean(1))
    traj = np.array([to_global(rows51(mxy[i], myaw[i]), inp[c]["pose"]) for i, c in enumerate(names)])
    traj0 = np.array([to_global(rows51(xy[i, 0], yaw[i, 0]), inp[c]["pose"]) for i, c in enumerate(names)])
    assert len(set(names)) == len(names) == len(inp), (len(names), len(inp))
    assert np.isfinite(traj).all()
    np.savez(f"{tag}_test.npz", names=np.array(names), traj=traj, traj0=traj0, ego_mean=mxy, K=xy.shape[1])
    d = np.linalg.norm(traj[:, 50, :2] - traj[:, 0, :2], axis=1)
    print(f"{tag}_test.npz: {len(names)} clips, K={xy.shape[1]}, 5 s disp median {np.median(d):.2f} m, <1 m {np.mean(d < 1):.3f}")
