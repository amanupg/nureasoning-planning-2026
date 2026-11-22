# Pool K samples from several Edge val runs: score the pointwise mean of the union vs each run's own mean.
import sys, glob, json, collections, numpy as np
sys.path.insert(0, "src")
from concurrent.futures import ProcessPoolExecutor
import edge_score as E
from plan_traj import to_global

def load(tag):
    D = {}
    for f in sorted(glob.glob(f"{tag}_validation_shard*of*.npz")):
        if f.endswith(".partial.npz"): continue
        z = np.load(f)
        for n, x, y in zip(map(str, z["names"]), z["xy"], z["yaw"]): D[n] = (x, y)
    return D

def job(a):
    c, T = a; r = E.score("validation", c, T); return c, (E.nps(r) if r else np.nan)

if __name__ == "__main__":
    tags = sys.argv[1:]; Ds = [load(t) for t in tags]; inp = json.load(open("data/edge_inputs_validation.json"))
    names = sorted(set.intersection(*[set(D) for D in Ds]))
    def meantraj(c):
        xy = np.concatenate([D[c][0] for D in Ds]); yaw = np.concatenate([D[c][1] for D in Ds])
        return to_global(E.rows51(xy.mean(0), np.arctan2(np.sin(yaw).mean(0), np.cos(yaw).mean(0))), inp[c]["pose"])
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(job, [(c, meantraj(c)) for c in names], chunksize=4))
    ref = json.load(open(f"{tags[0]}_val_scores.json"))
    log = lambda c: "_".join(c.split("_")[:2]); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(R[c] - ref[c]["edge"])
    L = sorted(by); s = np.array([np.nansum(by[l]) for l in L]); n = np.array([len(by[l]) for l in L]); rs = np.random.RandomState(0)
    bs = [s[i].sum() / n[i].sum() for i in (rs.randint(0, len(L), len(L)) for _ in range(5000))]
    print(f"pool {'+'.join(t.split('/')[-1] for t in tags)}: {np.nanmean([R[c] for c in names]):.4f} vs {tags[0].split('/')[-1]} alone "
          f"{np.mean([ref[c]['edge'] for c in names]):.4f}: {s.sum()/n.sum():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
