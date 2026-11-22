"""agg_test.py MODE TAG1 [TAG2 ...]: score an alternative aggregation of the pooled K samples on val, paired log
bootstrap vs runs/edge_e2_2400_dupfix (e2a, mean of 6). Modes: mean | median (per-point xy median, circular-mean yaw of
the same samples) | trim1 / trim2 (drop the 1 / 2 samples farthest (mean L2 over time) from the sample mean, average
the rest). Writes runs/agg_<mode>_<tags>.json (per-clip NPS)."""
import sys, json, collections, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
from multiprocessing import Pool
MODE, TAGS = sys.argv[1], sys.argv[2:]
def init():
    global E, to_global, Ds, inp
    import edge_score as E0, edge_pool as P; from plan_traj import to_global as tg
    E = E0; to_global = tg; Ds = [P.load(t) for t in TAGS]; inp = json.load(open("data/edge_inputs_validation.json"))
def agg(xy, yaw):
    if MODE == "mean": keep = np.arange(len(xy))
    elif MODE == "median":
        k = np.arange(len(xy)); return np.median(xy, 0), np.arctan2(np.sin(yaw[k]).mean(0), np.cos(yaw[k]).mean(0))
    elif MODE.startswith("trim"):
        d = np.linalg.norm(xy - xy.mean(0), axis=-1).mean(-1); keep = np.argsort(d)[:len(xy) - int(MODE[4:])]
    return xy[keep].mean(0), np.arctan2(np.sin(yaw[keep]).mean(0), np.cos(yaw[keep]).mean(0))
def sc(c):
    xy = np.concatenate([D[c][0] for D in Ds]); yaw = np.concatenate([D[c][1] for D in Ds]); m, y = agg(xy, yaw)
    T = to_global(E.rows51(m, y), inp[c]["pose"]); r = E.score("validation", c, T); return c, (E.nps(r) if r else np.nan)
if __name__ == "__main__":
    ref = json.load(open("runs/edge_e2_2400_dupfix_val_scores.json")); names = sorted(ref)
    with Pool(8, initializer=init) as p: R = dict(p.map(sc, names, chunksize=8))
    log = lambda c: "_".join(c.split("_")[:2]); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(R[c] - ref[c]["edge"])
    L = sorted(by); s = np.array([np.nansum(by[l]) for l in L]); n = np.array([len(by[l]) for l in L]); rs = np.random.RandomState(0)
    bs = [s[i].sum() / n[i].sum() for i in (rs.randint(0, len(L), len(L)) for _ in range(5000))]
    h = rs.permutation(L)[:len(L) // 2]; hs = set(h); da = [s[i] / n[i] for i in range(len(L))]
    ga = sum(s[i] for i in range(len(L)) if L[i] in hs) / sum(n[i] for i in range(len(L)) if L[i] in hs)
    gb = sum(s[i] for i in range(len(L)) if L[i] not in hs) / sum(n[i] for i in range(len(L)) if L[i] not in hs)
    print(f"{MODE} {'+'.join(t.split('/')[-1] for t in TAGS)}: {np.nanmean(list(R.values())):.4f} vs e2a 0.7390: "
          f"{s.sum()/n.sum():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}] halves {ga:+.4f} {gb:+.4f}", flush=True)
    json.dump(R, open(f"runs/agg_{MODE}_{'_'.join(t.split('/')[-1] for t in TAGS)}.json", "w"))
