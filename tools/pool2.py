# pool2.py TAG1 TAG2 ...: score the mean of the union of K samples on val, paired log-bootstrap vs TAG1 alone.
import sys, json, collections, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
from multiprocessing import Pool
TAGS = sys.argv[1:]
def init():
    global E, to_global, Ds, inp
    import edge_score as E0, edge_pool as P; from plan_traj import to_global as tg
    E = E0; to_global = tg; Ds = [P.load(t) for t in TAGS]; inp = json.load(open("data/edge_inputs_validation.json"))
def sc(c):
    xy = np.concatenate([D[c][0] for D in Ds]); yaw = np.concatenate([D[c][1] for D in Ds])
    T = to_global(E.rows51(xy.mean(0), np.arctan2(np.sin(yaw).mean(0), np.cos(yaw).mean(0))), inp[c]["pose"])
    r = E.score("validation", c, T); return c, (E.nps(r) if r else np.nan)
if __name__ == "__main__":
    ref = json.load(open(f"{TAGS[0]}_val_scores.json")); names = sorted(ref)
    with Pool(8, initializer=init) as p: R = dict(p.map(sc, names, chunksize=8))
    log = lambda c: "_".join(c.split("_")[:2]); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(R[c] - ref[c]["edge"])
    L = sorted(by); s = np.array([np.nansum(by[l]) for l in L]); n = np.array([len(by[l]) for l in L]); rs = np.random.RandomState(0)
    bs = [s[i].sum() / n[i].sum() for i in (rs.randint(0, len(L), len(L)) for _ in range(5000))]
    print(f"pool {'+'.join(TAGS)}: {np.nanmean(list(R.values())):.4f} (nan {np.isnan(list(R.values())).sum()}) vs {TAGS[0]} {np.mean([ref[c]['edge'] for c in names]):.4f}: "
          f"{s.sum()/n.sum():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
    json.dump(R, open(f"runs/pool_{'_'.join(t.split('/')[-1] for t in TAGS)}.json", "w"))
