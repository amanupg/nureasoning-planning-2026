"""pool_test.py SPLIT OUT TAG1 [TAG2 ...]: pool the K samples of several Edge runs exactly as tools/pool2.py does on val
(pointwise mean of the union of samples, circular-mean yaw) and write a member file OUT (names, traj = pooled global
51x3, traj0 = TAG1 candidate 0, ego_mean, K) in the format of src/edge_merge_test.py.
SPLIT = test (reads <tag>_test_shard*.npz) or validation (reads <tag>_validation_shard*.npz, for a self-check vs pool2)."""
import sys, glob, json, numpy as np
sys.path.insert(0, "src")
from edge_score import rows51
from plan_traj import to_global

def load(tag, split):
    D = {}
    for f in sorted(glob.glob(f"{tag}_{split}_shard*of*.npz")):
        if f.endswith(".partial.npz"): continue
        z = np.load(f)
        for n, x, y in zip(map(str, z["names"]), z["xy"], z["yaw"]): D[n] = (x, y)
    return D

if __name__ == "__main__":
    split, out, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    Ds = [load(t, split) for t in tags]; inp = json.load(open(f"data/edge_inputs_{split}.json"))
    for t, D in zip(tags, Ds): print(f"{t}: {len(D)} clips, K={len(next(iter(D.values()))[0])}")
    names = sorted(inp) if split == "test" else sorted(set.intersection(*[set(D) for D in Ds]))
    assert all(set(names) <= set(D) for D in Ds), "a member is missing clips"
    traj, traj0, em = [], [], []
    for c in names:
        xy = np.concatenate([D[c][0] for D in Ds]); yaw = np.concatenate([D[c][1] for D in Ds])
        m = xy.mean(0); y = np.arctan2(np.sin(yaw).mean(0), np.cos(yaw).mean(0)); em.append(m)
        traj.append(to_global(rows51(m, y), inp[c]["pose"]))
        traj0.append(to_global(rows51(Ds[0][c][0][0], Ds[0][c][1][0]), inp[c]["pose"]))
    traj = np.array(traj); assert np.isfinite(traj).all()
    K = sum(len(D[names[0]][0]) for D in Ds)
    np.savez(out, names=np.array(names), traj=traj, traj0=np.array(traj0), ego_mean=np.array(em), K=K)
    d = np.linalg.norm(traj[:, 50, :2] - traj[:, 0, :2], axis=1)
    print(f"{out}: {len(names)} clips, pooled K={K}, 5 s disp median {np.median(d):.2f} m")
