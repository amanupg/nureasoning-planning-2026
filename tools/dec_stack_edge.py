# dec_stack_edge.py TAG [--test TESTTAG]: decision stacker + decision-steered aggregation of Edge K samples on val.
# Target: DriveMA 4-class longitudinal decision of the recorded 5 s future (nureasoning/artifacts/meta_labels_validation.json c4;
# 0 stop, 1 decel, 2 keep, 3 accel). Features per clip (no test labels, no MCQ options):
#   Edge class shares of the K samples (tools/dec_oracle_edge.cls5), log of the reasoning thread's meta-action VLM probs
#   (Qwen-Drive LoRA, train-only), ego kinematic history (v_m0..5, ax, yaw_rate, route command).
# Multinomial logistic regression, OOF by drive log (GroupKFold 5) on val. Aggregation: weighted K-mean with weight
# P_stack(class of sample)^a, a in A_GRID (inf = only samples of the argmax class, all if none). a chosen on one log half
# and checked on the other (both directions); full-set paired log bootstrap vs the plain K-mean (selected on same data).
# --test: fit on all val, write runs/dec_stack_test_probs.json (+ weights applied by tools/dec_apply_test.py).
import sys, json, collections, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
from multiprocessing import Pool
from dec_oracle_edge import cls5, init as _init
import dec_oracle_edge as DO
A = "/path/to/workspace/nureasoning/artifacts/"
CL = ["stop", "decel", "keep", "accel"]; CMD = ["GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT"]
A_GRID = [0.5, 1.0, 2.0, float("inf")]
log_of = lambda c: "_".join(c.split("_")[:2])


def feats(names, shares, split):
    meta = json.load(open(A + f"meta_vlm_probs_{split}.json")); kin = json.load(open(A + f"meta_labels_{split}.json"))
    X = []
    for c in names:
        k = kin[c]; m = meta[c]["vlm_meta"]
        X.append(list(shares[c]) + [np.log(max(m[x], 1e-6)) for x in CL] +
                 [k[f"v_m{i}"] for i in range(6)] + [k["ax"], k["yaw_rate"]] + [float(k["command"] == x) for x in CMD])
    return np.array(X, float)


def model():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def boot(d, names):
    by = collections.defaultdict(list)
    for c in names: by[log_of(c)].append(d[c])
    L = sorted(by); s = np.array([np.nansum(by[l]) for l in L]); n = np.array([len(by[l]) for l in L]); r = np.random.RandomState(0)
    bs = [s[i].sum() / n[i].sum() for i in (r.randint(0, len(L), len(L)) for _ in range(3000))]
    return s.sum() / n.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def init_tag(tag):
    DO.TAGS = [tag]; _init()


def weights(P, sc, a):
    if a == float("inf"):
        h = (sc == P.argmax()).astype(float); return h if h.any() else np.ones(len(sc))
    return np.maximum(P[sc], 1e-9) ** a


if __name__ == "__main__":
    import edge_pool as EP
    from sklearn.model_selection import GroupKFold
    TAG = sys.argv[1]; ref = json.load(open(f"{TAG}_val_scores.json")); names = sorted(ref)
    D = EP.load(TAG); lab = json.load(open(A + "meta_labels_validation.json"))
    y = np.array([lab[c]["c4"] for c in names])
    sc = {c: np.array([cls5(x[[9, 19, 29, 39, 49]]) for x in D[c][0]]) for c in names}
    shares = {c: np.bincount(sc[c], minlength=4) / len(sc[c]) for c in names}
    X = feats(names, shares, "validation"); g = np.array([log_of(c) for c in names])
    P = np.zeros((len(names), 4))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = model().fit(X[tr], y[tr]); P[te] = m.predict_proba(X[te])
    maj = np.array([shares[c].argmax() for c in names])
    meta = json.load(open(A + "meta_vlm_probs_validation.json"))
    vm = np.array([[meta[c]["vlm_meta"][x] for x in CL] for c in names]).argmax(1)
    print(f"accuracy: Edge majority {np.mean(maj == y):.3f}, meta VLM {np.mean(vm == y):.3f}, stacker OOF {np.mean(P.argmax(1) == y):.3f}; "
          f"stacker NLL {-np.mean(np.log(P[np.arange(len(y)), y] + 1e-9)):.3f}")
    for k in range(4): print(f"  {CL[k]}: recall Edge {np.mean(maj[y == k] == k):.2f} stacker {np.mean(P.argmax(1)[y == k] == k):.2f}")
    if "--abl" in sys.argv:  # stacker without the meta VLM probs: does the reasoning model add information?
        X2 = np.delete(X, range(4, 8), 1); P2 = np.zeros_like(P)
        for tr, te in GroupKFold(5).split(X2, y, g): P2[te] = model().fit(X2[tr], y[tr]).predict_proba(X2[te])
        print(f"  ablation without meta VLM probs: acc {np.mean(P2.argmax(1) == y):.3f}")
    Pi = {c: P[i] for i, c in enumerate(names)}
    rs = np.random.RandomState(0); logs = sorted(set(g)); half = set(rs.choice(logs, len(logs) // 2, replace=False))
    R = {}
    with Pool(8, initializer=init_tag, initargs=(TAG,)) as p:
        for a in A_GRID:
            r = dict(p.map(DO.sc, [(c, weights(Pi[c], sc[c], a)) for c in names], chunksize=8))
            R[a] = {c: r[c] - ref[c]["edge"] for c in names}
            m, lo, hi = boot(R[a], names)
            hA = np.nanmean([R[a][c] for c in names if log_of(c) in half]); hB = np.nanmean([R[a][c] for c in names if log_of(c) not in half])
            print(f"a={a}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (selected on same data); half A {hA:+.4f}, half B {hB:+.4f}", flush=True)
    json.dump({c: Pi[c].tolist() for c in names}, open("runs/dec_stack_val_oof.json", "w"))
