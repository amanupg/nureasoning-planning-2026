# dec_oracle_edge.py TAG [TAG2 ...]: ceiling of decision-steered aggregation of Edge K samples on val.
# True longitudinal class per clip from the human 5 s future (DriveMA thresholds on 1 Hz points: stop if v_max <= 1.333
# and 5 s distance <= 1.952 m; else speed slope k_v <= -0.448 decel, >= 0.429 accel, else keep). Each sample gets the
# same class from its own trajectory. Aggregations (weighted mean of samples, weights from a class "prediction"):
#   hard: only samples of the predicted class (all if none); soft q: weight q on predicted-class samples, 1-q on the rest.
# Predictions: true class (accuracy 1.0) or a simulated predictor of accuracy a (wrong -> random other class, seed 0).
# Paired log bootstrap vs the plain K-mean of TAG (runs/<TAG>_val_scores.json "edge").
import sys, json, collections, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
from multiprocessing import Pool
TAGS = sys.argv[1:]
CL = ["stop", "decel", "keep", "accel"]


def cls5(p):  # p: (5, 2) ego-frame positions at 1..5 s
    q = np.vstack([[0, 0], p]); v = np.linalg.norm(np.diff(q, axis=0), axis=1)  # speeds at 0.5..4.5 s
    if v.max() <= 1.333 and np.linalg.norm(p[-1]) <= 1.952: return 0
    k = np.polyfit(np.arange(5) + 0.5, v, 1)[0]
    return 1 if k <= -0.448 else 3 if k >= 0.429 else 2


def init():
    global E, to_global, Ds, inp
    import edge_score as E0, edge_pool as P; from plan_traj import to_global as tg
    E = E0; to_global = tg; Ds = [P.load(t) for t in TAGS]; inp = json.load(open("data/edge_inputs_validation.json"))


def sc(a):
    c, w = a
    xy = np.concatenate([D[c][0] for D in Ds]); yaw = np.concatenate([D[c][1] for D in Ds]); w = np.asarray(w, float)[:, None]
    m = (xy * w[..., None]).sum(0) / w.sum(); my = np.arctan2((np.sin(yaw) * w).sum(0), (np.cos(yaw) * w).sum(0))
    r = E.score("validation", c, to_global(E.rows51(m, my), inp[c]["pose"])); return c, (E.nps(r) if r else np.nan)


if __name__ == "__main__":
    import edge_pool as P
    ref = json.load(open(f"{TAGS[0]}_val_scores.json")); names = sorted(ref)
    gt = {}
    for l in open("data/plan_rows/planning_validation.jsonl"):
        r = json.loads(l); gt[r["clip"]] = np.asarray(r["gt5"], float)
    Ds = [P.load(t) for t in TAGS]
    tc = {c: cls5(gt[c][:, :2]) for c in names}
    sc_ = {c: np.array([cls5(x[[9, 19, 29, 39, 49]]) for x in np.concatenate([D[c][0] for D in Ds])]) for c in names}
    print("true class counts", collections.Counter(CL[tc[c]] for c in names))
    print("coverage: true class among samples", np.mean([tc[c] in sc_[c] for c in names]).round(3),
          "| majority-sample class == true", np.mean([np.bincount(sc_[c], minlength=4).argmax() == tc[c] for c in names]).round(3))
    rs = np.random.RandomState(0)
    def pred(acc):
        out = {}
        for c in names:
            out[c] = tc[c] if rs.rand() < acc else rs.choice([k for k in range(4) if k != tc[c]])
        return out
    cfgs = []
    for acc in (1.0, 0.8, 0.65):
        pr = pred(acc)
        cfgs.append((f"hard acc{acc}", {c: (sc_[c] == pr[c]).astype(float) if (sc_[c] == pr[c]).any() else np.ones(len(sc_[c])) for c in names}))
        for q in (0.8,):
            cfgs.append((f"soft q{q} acc{acc}", {c: np.where(sc_[c] == pr[c], q, 1 - q) for c in names}))
    log = lambda c: "_".join(c.split("_")[:2])
    with Pool(8, initializer=init) as p:
        for name, W in cfgs:
            R = dict(p.map(sc, [(c, W[c]) for c in names], chunksize=8))
            by = collections.defaultdict(list)
            for c in names: by[log(c)].append(R[c] - ref[c]["edge"])
            L = sorted(by); s = np.array([np.nansum(by[l]) for l in L]); n = np.array([len(by[l]) for l in L]); r2 = np.random.RandomState(0)
            bs = [s[i].sum() / n[i].sum() for i in (r2.randint(0, len(L), len(L)) for _ in range(3000))]
            print(f"{name}: {np.nanmean(list(R.values())):.4f} vs K-mean {np.mean([ref[c]['edge'] for c in names]):.4f}: "
                  f"{s.sum()/n.sum():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]", flush=True)
